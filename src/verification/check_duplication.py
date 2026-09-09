# -*- coding: utf-8 -*-
"""Task 3 Group D item 11 - Near-duplicate detection at the SFT/DPO OUTPUT level.

Input : data/adapted/sft_task2.jsonl   (--sft-in)
        data/adapted/dpo_task2.jsonl   (--dpo-in)
Output: data/processed/duplication_report.json   (always written, even at zero findings)
        optionally a deduplicated copy (--apply), never in place

Reuses `data_engineering/dedup.py` rather than restating it
--------------------------------------------------------
`canonical_text`, `sha256_of`, `shingles`, `build_minhash`, `true_jaccard`,
`lsh_threshold_for`, `NUM_PERM` and `DEFAULT_THRESHOLD` are all imported. In particular
the LSH-as-candidate-generator design is inherited whole: the index is built at a LOOSER
threshold than the decision threshold, because LSH banding is probabilistic and a pair
sitting exactly at the index threshold has only ~50% chance of being emitted. Every
candidate is then confirmed against the true Jaccard. That subtlety was found by
dedup.py's own self-test; it is not re-derived here.

Why this stage exists when dedup.py already runs
------------------------------------------------
dedup.py deduplicates SOURCE DOCUMENTS before chunking. It cannot see the generated data:
Task 2 produces many records from each surviving chunk (4,645 SFT records from 394
chunks, ~11.8 per chunk), and nothing upstream compares those records to each other. Two
chunks that were legitimately distinct can still yield near-identical instructions, and a
single chunk can yield the same question phrased twice.

THE ONE CHANGED PARAMETER, AND WHY - shingle size 5 -> 3
--------------------------------------------------------
This is the only constant deliberately not inherited, because dedup.py's value was tuned
for a different regime and carrying it over would have produced a silently weaker check.
dedup.py shingles whole documents of thousands of words. These records are short:

  field             min   p10   median   p90   max     (words, whitespace-collapsed)
  SFT response       2    10      17      27    53
  SFT instruction    3     4       6       9    19
  DPO chosen         3    10      17      27    46
  DPO rejected       2    11      17      28    49

`shingles()` returns the whole text as ONE shingle when the text is shorter than the
shingle size. At size 5 that is not a rare edge case here - it is **27.2% of SFT
instructions** (1,262 of 4,645), with 64.0% producing fewer than three shingles. For that
quarter of the field the "near-duplicate" check silently degrades into an exact-match
check, and reports a clean near-duplicate result it never actually performed. At size 3
the degenerate share is 0.0%.

Both sizes were run against the real data and both return the same verdict on it (zero
near-duplicate pairs at 0.80), so this change did not manufacture the result - it makes
the check able to detect one. `--shingle-size` overrides it, and `degenerate_texts` is
reported per field so a future dataset with even shorter text cannot hide the same way.

Short text is also intrinsically more edit-sensitive, which is a property of the data and
not a defect to correct. Measured, tail edit of k words at 5-shingles:

  words   k=1   k=2   k=3   k=5   k=10
  20      0.88  0.78  0.68  0.52  0.29
  120     0.98  0.97  0.95  0.92  0.88

So at the median length of 17 words a genuine 3-word paraphrase scores well under 0.80.
A near-duplicate threshold on short text is therefore a check for COPIES, not for
paraphrases; §4 of the report records this rather than implying broader coverage.

Four checks, because they answer different questions
----------------------------------------------------
  EXACT_DUP        byte-identical after whitespace collapse, within one field
  NEAR_DUP         Jaccard >= threshold within one field
  PAIR_COLLAPSE    a DPO record whose `rejected` is a near-copy of its own `chosen`.
                   This is the one that matters most and no per-field scan would find
                   it: both texts can be unique in the corpus and the pair still carries
                   no preference signal, because the two sides say the same thing.
  CROSS_FILE       a DPO `chosen` that is also an SFT `response`. Expected by
                   construction - Task 2 built the pairs from the SFT records - but it is
                   the mechanism by which an independent SFT/DPO split leaks, so it is
                   counted here and consumed by check_leakage.py.

Flagged, not deleted
--------------------
The standing project rule is "never discard automatically" (dedup.py: near-duplicates are
flagged, only byte-identical documents are excluded). This module follows it. `--apply`
exists for when removal is actually wanted, and it writes a NEW file, never in place,
keeping the first member of each group in a deterministic order. The report is written
either way.

Limitations, stated rather than implied
---------------------------------------
- Near-duplicate detection on short text finds copies, not paraphrases (above).
- Jaccard over word shingles is orthography-sensitive. `canonical_text` collapses
  whitespace only; it does NOT fold alef/ya. Two spellings of one word are two different
  shingles. dedup.py could rely on `cleaned_text` having been folded upstream; the
  adapted records are training text and are deliberately NOT folded, so a
  spelling-variant near-duplicate is invisible to this check. Folding here was rejected
  for the reason clean.py gives: the fold is right for matching and wrong for text you
  keep, and emitting a "duplicate" verdict computed on a form that is not the shipped
  form would be reporting on a text that does not exist.
- PAIR_COLLAPSE uses the same threshold as NEAR_DUP by default. A pair at 0.75 is not
  reported yet is still a weak training signal; the report includes the full distribution
  so the threshold can be argued with rather than merely trusted.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_engineering'))

from dedup import (canonical_text, sha256_of, shingles, build_minhash,      # noqa: E402
                   true_jaccard, lsh_threshold_for, NUM_PERM, DEFAULT_THRESHOLD)
from datasketch import MinHashLSH                                           # noqa: E402

SFT_IN = 'data/adapted/sft_task2.jsonl'
DPO_IN = 'data/adapted/dpo_task2.jsonl'
REPORT_PATH = 'data/processed/duplication_report.json'

# See the module docstring. NOT inherited from dedup.py, and that is the point.
SHORT_TEXT_SHINGLE = 3

# Per-finding detail is SAMPLED in the written report, never enumerated in full.
# CROSS_FILE fires 2,485 times on the current data because Task 2 builds the DPO pairs
# from the SFT records - the count is the finding, and 2,485 index pairs is not something
# anyone reads. Enumerating them produced a 17,712-line, 466KB report against a repo where
# every other report is 21-62 lines, and it would churn completely on every regeneration.
# The full COUNT is always reported; only the per-item list is capped. `run()` returns the
# uncapped findings in memory, so --apply and any caller still see everything.
REPORT_SAMPLE = 50

SFT_FIELDS = ('instruction', 'response')
DPO_FIELDS = ('prompt', 'chosen', 'rejected')

EXACT_DUP, NEAR_DUP, PAIR_COLLAPSE, CROSS_FILE = (
    'EXACT_DUP', 'NEAR_DUP', 'PAIR_COLLAPSE', 'CROSS_FILE')


# --------------------------------------------------------------------------- machinery

def _prepared(texts, size):
    """Canonical text and its shingle set, plus how many degenerated to one shingle."""
    canon = [canonical_text(t) for t in texts]
    shs = [shingles(c, size) for c in canon]
    degenerate = sum(1 for c in canon if len(c.split()) < size)
    return canon, shs, degenerate


def exact_groups(canon):
    """Indices grouped by SHA-256 of the canonical text, groups of 2+ only."""
    by_hash = defaultdict(list)
    for i, c in enumerate(canon):
        by_hash[sha256_of(c)].append(i)
    return {h: idx for h, idx in by_hash.items() if len(idx) > 1}


def near_pairs(shs, threshold, num_perm=NUM_PERM):
    """(jaccard, i, j) for every pair at or above `threshold`, i < j.

    LSH is a candidate generator only - indexed at the looser threshold that dedup.py
    derives, then every candidate is confirmed on the true Jaccard. An empty shingle set
    is skipped rather than indexed: it matches nothing and MinHash over it is undefined.
    """
    lsh = MinHashLSH(threshold=lsh_threshold_for(threshold), num_perm=num_perm)
    mins = {}
    for i, s in enumerate(shs):
        if not s:
            continue
        m = build_minhash(s, num_perm)
        mins[i] = m
        lsh.insert(str(i), m)
    out = []
    for i, m in mins.items():
        for key in lsh.query(m):
            j = int(key)
            if j <= i:
                continue
            score = true_jaccard(shs[i], shs[j])
            if score >= threshold:
                out.append((score, i, j))
    return sorted(out, reverse=True)


def _finding(code, field, detail):
    d = {'code': code, 'field': field}
    d.update(detail)
    return d


# ------------------------------------------------------------------------- the checks

def check_field_duplication(rows, field, threshold, size):
    """EXACT_DUP + NEAR_DUP within one field of one file."""
    canon, shs, degenerate = _prepared([r.get(field, '') for r in rows], size)
    findings = []

    for _h, idx in sorted(exact_groups(canon).items(), key=lambda kv: kv[1][0]):
        findings.append(_finding(EXACT_DUP, field, {
            'indices': idx,
            'redundant': len(idx) - 1,
            'chunk_ids': sorted({rows[i].get('source_chunk_id', '?') for i in idx}),
            'text': canon[idx[0]][:120]}))

    for score, i, j in near_pairs(shs, threshold):
        findings.append(_finding(NEAR_DUP, field, {
            'indices': [i, j],
            'jaccard': round(score, 4),
            'chunk_ids': sorted({rows[i].get('source_chunk_id', '?'),
                                 rows[j].get('source_chunk_id', '?')}),
            'text': [canon[i][:120], canon[j][:120]]}))

    return findings, {'records': len(rows),
                      'degenerate_texts': degenerate,
                      'degenerate_pct': round(100.0 * degenerate / len(rows), 2) if rows else 0.0}


def check_pair_collapse(dpo_rows, threshold, size):
    """A DPO pair whose two sides say the same thing carries no preference signal."""
    findings, scores = [], []
    for i, r in enumerate(dpo_rows):
        a = shingles(canonical_text(r.get('chosen', '')), size)
        b = shingles(canonical_text(r.get('rejected', '')), size)
        score = true_jaccard(a, b)
        identical = canonical_text(r.get('chosen', '')) == canonical_text(r.get('rejected', ''))
        scores.append(score)
        if identical or score >= threshold:
            findings.append(_finding(PAIR_COLLAPSE, 'chosen|rejected', {
                'index': i,
                'jaccard': round(score, 4),
                'identical': identical,
                'rejection_type': r.get('rejection_type'),
                'chunk_id': r.get('source_chunk_id'),
                'text': [canonical_text(r.get('chosen', ''))[:120],
                         canonical_text(r.get('rejected', ''))[:120]]}))
    return findings, scores


def check_cross_file(sft_rows, dpo_rows):
    """DPO `chosen` values that are byte-identical to some SFT `response`.

    Not a defect on its own - Task 2 builds the pairs from the SFT records, so overlap is
    the expected shape. It is reported because it is the exact mechanism by which
    splitting the two files independently leaks text across partitions; check_leakage.py
    consumes this count.
    """
    sft_hashes = {}
    for i, r in enumerate(sft_rows):
        sft_hashes.setdefault(sha256_of(canonical_text(r.get('response', ''))), i)
    findings = []
    for i, r in enumerate(dpo_rows):
        h = sha256_of(canonical_text(r.get('chosen', '')))
        if h in sft_hashes:
            findings.append(_finding(CROSS_FILE, 'dpo.chosen==sft.response', {
                'dpo_index': i,
                'sft_index': sft_hashes[h],
                'chunk_id': r.get('source_chunk_id')}))
    return findings


# ----------------------------------------------------------------------------- removal

def removal_indices(findings):
    """Indices to drop, keeping the FIRST member of each exact/near group.

    Deterministic: groups are processed in index order and the survivor is the lowest
    index, so two runs over the same file remove the same records. Near-duplicate pairs
    are resolved transitively through a union of overlapping pairs, because A~B and B~C
    should collapse to one survivor rather than two.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for f in findings:
        if f['code'] == EXACT_DUP:
            idx = f['indices']
            for other in idx[1:]:
                union(idx[0], other)
        elif f['code'] == NEAR_DUP:
            union(f['indices'][0], f['indices'][1])
    return sorted(x for x in parent if find(x) != x)


# ------------------------------------------------------------------------------- run

def run(sft_rows, dpo_rows, threshold=DEFAULT_THRESHOLD, size=SHORT_TEXT_SHINGLE):
    report = {'threshold': threshold,
              'lsh_index_threshold': lsh_threshold_for(threshold),
              'shingle_size': size,
              'shingle_size_source': 'check_duplication.SHORT_TEXT_SHINGLE '
                                     '(deliberately not dedup.SHINGLE_SIZE=5; see docstring)',
              'num_perm': NUM_PERM,
              'sft': {'records': len(sft_rows), 'fields': {}, 'findings': []},
              'dpo': {'records': len(dpo_rows), 'fields': {}, 'findings': []}}

    for field in SFT_FIELDS:
        f, stats = check_field_duplication(sft_rows, field, threshold, size)
        report['sft']['fields'][field] = stats
        report['sft']['findings'] += f

    for field in DPO_FIELDS:
        f, stats = check_field_duplication(dpo_rows, field, threshold, size)
        report['dpo']['fields'][field] = stats
        report['dpo']['findings'] += f

    collapse, scores = check_pair_collapse(dpo_rows, threshold, size)
    report['dpo']['findings'] += collapse
    if scores:
        s = sorted(scores, reverse=True)
        report['pair_similarity'] = {
            'max': round(s[0], 4),
            'p90': round(s[int(0.10 * len(s))], 4),
            'median': round(s[len(s) // 2], 4),
            'at_or_above': {str(t): sum(1 for v in s if v >= t)
                            for t in (0.95, 0.90, 0.80, 0.70, 0.60)}}

    cross = check_cross_file(sft_rows, dpo_rows)
    report['cross_file'] = {
        'dpo_chosen_also_an_sft_response': len(cross),
        'pct_of_dpo': round(100.0 * len(cross) / len(dpo_rows), 2) if dpo_rows else 0.0}
    report['cross_file_findings'] = cross

    report['counts'] = Counter(f['code'] for f in
                               report['sft']['findings'] + report['dpo']['findings'])
    report['counts'][CROSS_FILE] = len(cross)
    report['counts'] = dict(report['counts'])
    return report


# ------------------------------------------------------------------------- self-test

def run_self_test():
    """Discrimination first: a check that cannot fail on bad input has not been tested."""
    ok = True

    def sft(instr, resp, cid='asas_albalagha_c0001'):
        return {'instruction': instr, 'response': resp, 'source_chunk_id': cid}

    def dpo(prompt, chosen, rejected, cid='asas_albalagha_c0001', rt='factual_errors'):
        return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected,
                'source_chunk_id': cid, 'rejection_type': rt}

    A = 'ططط ظظظ ضضض ذذذ ثثث خخخ صصص قققق ففف غغغ'
    B = 'حححح جججج دددد سسسس شششش زززز طططط كككك للل ممم'

    # 1. clean input must produce NOTHING. A checker that always fires is useless.
    rep = run([sft('س ١', A), sft('س ٢', B)], [dpo('ب', A, B)])
    if rep['counts'].get(EXACT_DUP) or rep['counts'].get(NEAR_DUP):
        print('  [FAIL] distinct texts reported as duplicates:', rep['counts'])
        ok = False

    # 2. an exact duplicate must be caught, and only whitespace should be ignored
    rep = run([sft('س ١', A), sft('س ٢', '  ' + A.replace(' ', '   ') + ' ')], [])
    ex = [f for f in rep['sft']['findings'] if f['code'] == EXACT_DUP and f['field'] == 'response']
    if len(ex) != 1 or ex[0]['redundant'] != 1:
        print('  [FAIL] whitespace-only variant not caught as EXACT_DUP:', ex)
        ok = False

    # 3. a near-duplicate must be caught at the decision threshold
    near = A + ' ننن'
    rep = run([sft('س ١', A), sft('س ٢', near)], [])
    if not [f for f in rep['sft']['findings'] if f['code'] == NEAR_DUP]:
        print('  [FAIL] one-word extension not caught as NEAR_DUP')
        ok = False

    # 4. PAIR_COLLAPSE must fire on identical sides even when both texts are unique in
    #    the corpus - the case no per-field scan can see.
    rep = run([], [dpo('ب', A, A)])
    pc = [f for f in rep['dpo']['findings'] if f['code'] == PAIR_COLLAPSE]
    if len(pc) != 1 or not pc[0]['identical']:
        print('  [FAIL] identical chosen/rejected not caught as PAIR_COLLAPSE:', pc)
        ok = False
    if [f for f in rep['dpo']['findings'] if f['code'] in (EXACT_DUP, NEAR_DUP)]:
        print('  [FAIL] a collapsed pair was also reported as a field duplicate')
        ok = False

    # 5. a genuine preference pair must NOT be reported
    rep = run([], [dpo('ب', A, B)])
    if [f for f in rep['dpo']['findings'] if f['code'] == PAIR_COLLAPSE]:
        print('  [FAIL] a genuinely different rejected side was flagged as collapsed')
        ok = False

    # 6. CROSS_FILE fires exactly when a DPO chosen equals an SFT response
    rep = run([sft('س', A)], [dpo('ب', A, B), dpo('ب', B, A)])
    if rep['cross_file']['dpo_chosen_also_an_sft_response'] != 1:
        print('  [FAIL] cross-file overlap miscounted:', rep['cross_file'])
        ok = False

    # 7. DEGENERATE SHINGLING must be visible, not silent. This is the pin for the one
    #    parameter this module changes: at size 5 a 3-word text is a single shingle and
    #    the near-duplicate check quietly becomes an exact-match check.
    short = [sft('س', 'ططط ظظظ ضضض'), sft('س', 'ططط ظظظ ذذذ')]
    at5 = run(short, [], size=5)['sft']['fields']['response']
    at3 = run(short, [], size=3)['sft']['fields']['response']
    if at5['degenerate_texts'] != 2:
        print('  [FAIL] degenerate texts not counted at size 5:', at5)
        ok = False
    if at3['degenerate_texts'] != 0:
        print('  [FAIL] size 3 should not degenerate on a 3-word text:', at3)
        ok = False

    # 8. removal keeps the FIRST member and is transitive across chained pairs
    findings = [{'code': EXACT_DUP, 'field': 'response', 'indices': [2, 5, 7]},
                {'code': NEAR_DUP, 'field': 'response', 'indices': [5, 9]},
                {'code': NEAR_DUP, 'field': 'response', 'indices': [11, 3]}]
    # {2,5,7} keeps 2; {5,9} merges into that set so 9 goes too; (11,3) is given in
    # DESCENDING order on purpose - the survivor is the lower index regardless of the
    # order the pair was emitted in, so 3 stays and 11 goes.
    got = removal_indices(findings)
    if got != [5, 7, 9, 11]:
        print('  [FAIL] removal set wrong: expected [5, 7, 9, 11], got %r' % got)
        ok = False

    # 9. the written report is SAMPLED but its COUNTS are not. A capped report that also
    #    capped the totals would understate the findings, which is the failure mode that
    #    makes sampling dangerous in the first place.
    fake = {'sft': {'records': 3, 'fields': {}, 'findings': [{'code': EXACT_DUP}] * 130},
            'dpo': {'records': 0, 'fields': {}, 'findings': []},
            'cross_file_findings': [{'code': CROSS_FILE}] * 2485}
    out = sampled_for_report(fake, limit=50)
    if len(out['sft']['findings']) != 50 or out['sft']['findings_total'] != 130:
        print('  [FAIL] sft findings not capped/counted correctly:',
              len(out['sft']['findings']), out['sft'].get('findings_total'))
        ok = False
    if out['sft']['findings_truncated'] != 80:
        print('  [FAIL] truncation count wrong:', out['sft']['findings_truncated'])
        ok = False
    if len(out['cross_file_findings']) != 50 or out['cross_file_findings_total'] != 2485:
        print('  [FAIL] cross-file list not capped/counted correctly')
        ok = False
    if len(fake['cross_file_findings']) != 2485 or len(fake['sft']['findings']) != 130:
        print('  [FAIL] sampling mutated the in-memory report; --apply would under-remove')
        ok = False

    # 10. the imports really are dedup.py's, not a local copy. If someone pastes these
    #    helpers in later, this fails.
    import dedup as _d
    for fn in (canonical_text, sha256_of, shingles, build_minhash, true_jaccard):
        if getattr(_d, fn.__name__, None) is not fn:
            print('  [FAIL] %s is not dedup.py\'s - it has been restated locally' % fn.__name__)
            ok = False

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def sampled_for_report(report, limit=REPORT_SAMPLE):
    """A copy of `report` with per-finding lists capped for writing to disk.

    Counts are computed before the cap and are never reduced, so the written report
    always states the true totals; `*_truncated` records how many were omitted. The
    in-memory report is left alone - removal and any programmatic caller need the full
    lists.
    """
    out = dict(report)
    for side in ('sft', 'dpo'):
        if side in out and isinstance(out[side], dict):
            f = out[side].get('findings', [])
            out[side] = dict(out[side])
            out[side]['findings_total'] = len(f)
            out[side]['findings_truncated'] = max(0, len(f) - limit)
            out[side]['findings'] = f[:limit]
    cf = out.get('cross_file_findings', [])
    out['cross_file_findings_total'] = len(cf)
    out['cross_file_findings_truncated'] = max(0, len(cf) - limit)
    out['cross_file_findings'] = cf[:limit]
    out['report_sample_limit'] = limit
    return out


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--sft-in', default=SFT_IN)
    ap.add_argument('--dpo-in', default=DPO_IN)
    ap.add_argument('--report', default=REPORT_PATH)
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--shingle-size', type=int, default=SHORT_TEXT_SHINGLE)
    ap.add_argument('--apply', action='store_true',
                    help='write deduplicated copies (NEVER in place) alongside the report')
    ap.add_argument('--out-dir', default='data/adapted',
                    help='where --apply writes its *_dedup.jsonl copies')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    sft = _load(args.sft_in) if os.path.exists(args.sft_in) else []
    dpo = _load(args.dpo_in) if os.path.exists(args.dpo_in) else []
    if not sft and not dpo:
        print('[dup] neither input exists: %s / %s' % (args.sft_in, args.dpo_in))
        sys.exit(2)

    rep = run(sft, dpo, args.threshold, args.shingle_size)
    rep['inputs'] = {'sft': args.sft_in, 'dpo': args.dpo_in}

    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as fh:
        json.dump(sampled_for_report(rep), fh, ensure_ascii=False, indent=2)

    print('[dup] %s: %d records | %s: %d records' % (args.sft_in, len(sft), args.dpo_in, len(dpo)))
    print('[dup] threshold=%.2f (LSH index %.2f)  shingle=%d words  num_perm=%d'
          % (rep['threshold'], rep['lsh_index_threshold'], rep['shingle_size'], rep['num_perm']))
    for side in ('sft', 'dpo'):
        for field, st in rep[side][side and 'fields'].items():
            flag = '' if st['degenerate_texts'] == 0 else \
                '   <- %d texts shorter than the shingle: exact-match only' % st['degenerate_texts']
            print('        %-4s %-12s %5d records, %d degenerate (%.1f%%)%s'
                  % (side, field, st['records'], st['degenerate_texts'],
                     st['degenerate_pct'], flag))
    print('[dup] findings:')
    for code in (EXACT_DUP, NEAR_DUP, PAIR_COLLAPSE, CROSS_FILE):
        print('        %-14s %d' % (code, rep['counts'].get(code, 0)))
    if 'pair_similarity' in rep:
        ps = rep['pair_similarity']
        print('[dup] chosen~rejected similarity: median=%.3f p90=%.3f max=%.3f'
              % (ps['median'], ps['p90'], ps['max']))
        print('        at or above: ' + '  '.join('%s:%d' % (t, n)
                                                  for t, n in sorted(ps['at_or_above'].items(),
                                                                     reverse=True)))
    print('[dup] cross-file: %d of %d DPO chosen values (%.1f%%) are also an SFT response'
          % (rep['cross_file']['dpo_chosen_also_an_sft_response'], len(dpo),
             rep['cross_file']['pct_of_dpo']))
    print('[dup] wrote %s' % args.report)

    if args.apply:
        os.makedirs(args.out_dir, exist_ok=True)
        for side, rows, name in (('sft', sft, 'sft_task2_dedup.jsonl'),
                                 ('dpo', dpo, 'dpo_task2_dedup.jsonl')):
            drop = set(removal_indices(rep[side]['findings']))
            out = os.path.join(args.out_dir, name)
            with open(out, 'w', encoding='utf-8') as fh:
                for i, r in enumerate(rows):
                    if i not in drop:
                        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            print('[dup] --apply: %s -> %d kept, %d removed (input untouched)'
                  % (out, len(rows) - len(drop), len(drop)))
    else:
        n = len(removal_indices(rep['sft']['findings'])) + len(removal_indices(rep['dpo']['findings']))
        print('[dup] flagged only; %d record(s) would be removed by --apply '
              '(project rule: never discard automatically)' % n)


if __name__ == '__main__':
    main()
