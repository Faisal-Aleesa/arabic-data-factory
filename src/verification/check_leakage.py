# -*- coding: utf-8 -*-
"""Task 3 Group D - Train/val/test leakage checker.

Implements docs/SPLIT_POLICY.md, which was written before this module and specifies what
it must verify. That file said "No leakage-checking module exists yet - when one is
built, it implements this and this file moves next to it or is superseded by its
docstring." This is that module; SPLIT_POLICY.md remains the rationale, and this docstring
does not repeat its argument for why the split unit is the document.

    THE RULE: split by the parent DOCUMENT of `source_chunk_id`, never by chunk.
              No document's derived records may appear in more than one partition.

Two modes, and the difference matters
-------------------------------------
  VERIFY       --partition NAME=PATH (repeatable). Checks real released partition files.
               This is the mode the policy asks for: "the check runs on the RELEASED
               files, not on an intermediate manifest", because partitions are assembled
               and rewritten after the split is decided.

  FEASIBILITY  no --partition given. Reports the splittable-unit inventory of the input
               and says whether the policy admits a split at all. It CANNOT and does not
               certify anything about leakage - there are no partitions to check.

FAIL-CLOSED: feasibility mode exits 2 and says explicitly that no partition check was
performed. A run that checked nothing must never read as a clean result. This is the same
discipline as audit_staged_arabic.py refusing to return 0 with no corpus loaded.

What is checked, and why each one can actually happen
-----------------------------------------------------
  UNPARSEABLE_CHUNK_ID   `source_chunk_id` does not match `<doc_id>_c<NNNN>`. Fails
                         LOUDLY by policy: an unparseable id silently defaults to "not
                         the same document", so every leak involving it passes. This is
                         the fail-open case the policy names explicitly.
  UNKNOWN_DOC_ID         parses, but the doc_id is not in docs/license_manifest.csv. The
                         policy fixes the manifest as the doc_id vocabulary, so that the
                         split unit and the licence-tracked unit are the same thing. A
                         mismatch means one of the two is wrong and the split unit is not
                         actually licence-tracked.
  DOC_IN_MULTIPLE_PARTITIONS   the rule itself.
  PAIR_SPLIT             a DPO pair whose two halves were assigned to different
                         partitions. Both derive from one chunk, so the pair belongs
                         wholly to its document's partition.
  TEXT_CROSSES_PARTITIONS      identical or near-identical text present in two
                         partitions. This survives a correct document split, because the
                         same text can be reached through different files - see below.

Why the text check is not redundant with the document check
-----------------------------------------------------------
A document-level split makes the *documents* disjoint. It does not make the *files*
disjoint. Measured on the current adapted data by check_duplication.py: **2,485 of 3,000
DPO `chosen` values (82.8%) are byte-identical to an SFT `response`.** That is expected -
Task 2 builds the pairs from the SFT records - but it means splitting the SFT file and the
DPO file independently puts the same sentence in SFT-train and DPO-test while every
document rule is still satisfied. The document check would report clean. Hence the text
check, which reuses check_duplication.py rather than restating MinHash/LSH.

Limitations
-----------
- The text check compares partitions pairwise; cost grows with the square of partition
  count, which is fine for the two or three partitions this policy can produce.
- SHORT shared strings are NOT filtered out. A two-word instruction appearing in two
  partitions is reported like any other crossing, and a human will dismiss some of those.
  That is deliberate: to be effective here a minimum-length filter would have to sit above
  the median instruction length (27.2% of instructions are under five words), so it would
  suppress a quarter of the field from a LEAK check to buy tidier output. Every crossing
  finding carries a `tokens` count instead, so triage stays possible without the checker
  deciding in advance what is too short to matter.
- Near-duplicate sensitivity is inherited from check_duplication.py and is a check for
  copies, not paraphrases, on text this short. Its docstring has the measured numbers.
- A partition file is trusted to be what it claims. This module does not verify that
  train/val/test are the complete and disjoint cover of some original set; that is
  validate_release.py's territory.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_engineering'))

import check_duplication as cd                                              # noqa: E402
from dedup import canonical_text, sha256_of, shingles, true_jaccard         # noqa: E402

MANIFEST = 'docs/license_manifest.csv'
REPORT_PATH = 'data/processed/leakage_report.json'

# From SPLIT_POLICY.md, "Deriving the document from a record". Verified there against all
# 731 chunk ids of the two original corpora with zero mismatches.
PARENT_DOC = re.compile(r'^(.*)_c\d{4}$')

# Fields whose text is compared across partitions. `instruction`/`prompt` are included
# because a question repeated across partitions leaks just as effectively as an answer.
TEXT_FIELDS = ('instruction', 'response', 'prompt', 'chosen', 'rejected')

(UNPARSEABLE_CHUNK_ID, UNKNOWN_DOC_ID, DOC_IN_MULTIPLE_PARTITIONS,
 PAIR_SPLIT, TEXT_CROSSES_PARTITIONS) = (
    'UNPARSEABLE_CHUNK_ID', 'UNKNOWN_DOC_ID', 'DOC_IN_MULTIPLE_PARTITIONS',
    'PAIR_SPLIT', 'TEXT_CROSSES_PARTITIONS')


# --------------------------------------------------------------------------- helpers

def parent_doc(chunk_id):
    """`<doc_id>_c<NNNN>` -> doc_id, or None. None is a finding, never a default."""
    m = PARENT_DOC.match(chunk_id or '')
    return m.group(1) if m else None


def known_doc_ids(manifest_path=MANIFEST):
    """The doc_id vocabulary, from the licence manifest. Empty set if unreadable."""
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path, encoding='utf-8') as fh:
        return {row['doc_id'].strip() for row in csv.DictReader(fh) if row.get('doc_id')}


def _texts_of(rec):
    for f in TEXT_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            yield f, v


# ---------------------------------------------------------------------------- checks

def check_ids(records, known, where):
    """UNPARSEABLE_CHUNK_ID and UNKNOWN_DOC_ID. Both fail loudly, by policy."""
    findings, docs = [], Counter()
    for i, r in enumerate(records):
        cid = r.get('source_chunk_id')
        doc = parent_doc(cid)
        if doc is None:
            findings.append({'code': UNPARSEABLE_CHUNK_ID, 'partition': where,
                             'index': i, 'source_chunk_id': cid,
                             'why': 'does not match <doc_id>_c<NNNN>; cannot be assigned '
                                    'to a split unit, so every leak involving it would '
                                    'pass unnoticed'})
            continue
        docs[doc] += 1
        if known and doc not in known:
            findings.append({'code': UNKNOWN_DOC_ID, 'partition': where,
                             'index': i, 'doc_id': doc, 'source_chunk_id': cid,
                             'why': 'not a doc_id in %s; the split unit and the '
                                    'licence-tracked unit are supposed to be the same '
                                    'set (SPLIT_POLICY.md, "Related")' % MANIFEST})
    return findings, docs


def check_document_disjoint(doc_counts_by_partition):
    """The rule: no doc_id in more than one partition."""
    where = defaultdict(list)
    for part, counts in doc_counts_by_partition.items():
        for doc in counts:
            where[doc].append(part)
    findings = []
    for doc, parts in sorted(where.items()):
        if len(parts) > 1:
            findings.append({
                'code': DOC_IN_MULTIPLE_PARTITIONS, 'doc_id': doc,
                'partitions': sorted(parts),
                'records': {p: doc_counts_by_partition[p][doc] for p in sorted(parts)}})
    return findings


def check_pairs_intact(partitions):
    """A DPO pair must live wholly in one partition.

    Two ways it can break. Within a file, `chosen` and `rejected` are one record, so the
    only way to separate them is a malformed record missing one side - reported, because
    a half pair is not a pair. Across files, the same chunk's pairs appearing in two
    partitions is already DOC_IN_MULTIPLE_PARTITIONS; what is checked here is the
    stricter statement the policy makes about the pair itself.
    """
    findings = []
    seen_chunk = {}
    for part, rows in partitions.items():
        for i, r in enumerate(rows):
            if 'chosen' not in r and 'rejected' not in r:
                continue
            if not (r.get('chosen') and r.get('rejected')):
                findings.append({'code': PAIR_SPLIT, 'partition': part, 'index': i,
                                 'source_chunk_id': r.get('source_chunk_id'),
                                 'why': 'half a preference pair: chosen=%s rejected=%s'
                                        % (bool(r.get('chosen')), bool(r.get('rejected')))})
                continue
            cid = r.get('source_chunk_id')
            if cid in seen_chunk and seen_chunk[cid] != part:
                findings.append({'code': PAIR_SPLIT, 'partition': part, 'index': i,
                                 'source_chunk_id': cid,
                                 'also_in': seen_chunk[cid],
                                 'why': 'pairs from one chunk span two partitions'})
            else:
                seen_chunk.setdefault(cid, part)
    return findings


def check_text_crosses(partitions, threshold, size):
    """Identical or near-identical text in two partitions.

    Exact matches are found by hash across all partitions at once. Near matches are found
    per partition PAIR, by indexing one side and querying with the other - reusing
    check_duplication.near_pairs on the concatenation and discarding same-partition hits,
    so the LSH-as-candidate-generator behaviour is inherited rather than re-implemented.
    """
    findings = []

    by_hash = defaultdict(list)
    for part, rows in partitions.items():
        for i, r in enumerate(rows):
            for field, text in _texts_of(r):
                by_hash[sha256_of(canonical_text(text))].append((part, i, field))
    for _h, hits in by_hash.items():
        parts = {p for p, _, _ in hits}
        if len(parts) > 1:
            _t = canonical_text(partitions[hits[0][0]][hits[0][1]][hits[0][2]])
            findings.append({'code': TEXT_CROSSES_PARTITIONS, 'match': 'exact',
                             'partitions': sorted(parts),
                             'tokens': len(_t.split()),
                             'occurrences': [{'partition': p, 'index': i, 'field': f}
                                             for p, i, f in hits[:6]],
                             'text': canonical_text(
                                 [r for p, i, f in hits[:1]
                                  for r in [partitions[p][i][f]]][0])[:120]})

    names = sorted(partitions)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pa, pb = names[a], names[b]
            flat = []
            for part in (pa, pb):
                for i, r in enumerate(partitions[part]):
                    for field, text in _texts_of(r):
                        flat.append((part, i, field, text))
            if not flat:
                continue
            shs = [shingles(canonical_text(t), size) for _, _, _, t in flat]
            for score, i, j in cd.near_pairs(shs, threshold):
                if flat[i][0] == flat[j][0]:
                    continue
                if score >= 1.0:
                    continue          # already reported as an exact match
                findings.append({
                    'code': TEXT_CROSSES_PARTITIONS, 'match': 'near',
                    'jaccard': round(score, 4), 'partitions': [pa, pb],
                    'tokens': min(len(canonical_text(flat[i][3]).split()),
                                  len(canonical_text(flat[j][3]).split())),
                    'occurrences': [{'partition': flat[i][0], 'index': flat[i][1],
                                     'field': flat[i][2]},
                                    {'partition': flat[j][0], 'index': flat[j][1],
                                     'field': flat[j][2]}],
                    'text': [canonical_text(flat[i][3])[:120],
                             canonical_text(flat[j][3])[:120]]})
    return findings


# ------------------------------------------------------------------------------- run

def verify(partitions, known=None, threshold=cd.DEFAULT_THRESHOLD,
           size=cd.SHORT_TEXT_SHINGLE):
    known = known_doc_ids() if known is None else known
    findings, docs_by_part = [], {}
    for part, rows in sorted(partitions.items()):
        f, docs = check_ids(rows, known, part)
        findings += f
        docs_by_part[part] = docs
    findings += check_document_disjoint(docs_by_part)
    findings += check_pairs_intact(partitions)
    findings += check_text_crosses(partitions, threshold, size)
    return {
        'mode': 'verify',
        'partitions': {p: {'records': len(r), 'documents': dict(docs_by_part[p])}
                       for p, r in sorted(partitions.items())},
        'known_doc_ids': sorted(known),
        'threshold': threshold,
        'shingle_size': size,
        'findings': findings,
        'counts': dict(Counter(f['code'] for f in findings)),
        'clean': not findings,
    }


def feasibility(records_by_name, known=None):
    """No partitions to check - report whether the policy admits a split at all."""
    known = known_doc_ids() if known is None else known
    findings, docs = [], Counter()
    for name, rows in sorted(records_by_name.items()):
        f, d = check_ids(rows, known, name)
        findings += f
        docs += d
    units = len(docs)
    return {
        'mode': 'feasibility',
        'checked_partitions': False,
        'inputs': {n: len(r) for n, r in sorted(records_by_name.items())},
        'splittable_units': units,
        'unit_sizes': dict(docs.most_common()),
        'largest_unit_share': round(100.0 * max(docs.values()) / sum(docs.values()), 1) if docs else 0.0,
        'three_way_split_possible': units >= 3,
        'two_way_split_possible': units >= 2,
        'known_doc_ids': sorted(known),
        'findings': findings,
        'counts': dict(Counter(f['code'] for f in findings)),
    }


# ------------------------------------------------------------------------- self-test

def run_self_test():
    ok = True
    KNOWN = {'asas_albalagha', 'dialect_dict_najdi'}

    # Each fixture gets a DISTINCT instruction/prompt. An earlier version defaulted
    # every record to the same one-word instruction, and the cross-partition text check
    # correctly flagged it - the probe was wrong, not the checker. Sharing a field by
    # accident is exactly what this module exists to catch.
    _n = [0]

    def _uniq(tag):
        _n[0] += 1
        return '%s %d ســؤال مختلف' % (tag, _n[0])

    def sft(cid, resp, instr=None):
        return {'source_chunk_id': cid,
                'instruction': instr or _uniq('س'), 'response': resp}

    def dpo(cid, chosen, rejected):
        return {'source_chunk_id': cid, 'prompt': _uniq('ب'),
                'chosen': chosen, 'rejected': rejected}

    A = 'ططط ظظظ ضضض ذذذ ثثث خخخ صصص قققق ففف غغغ'
    B = 'حححح جججج دددد سسسس شششش زززز طططط كككك للل ممم'
    C = 'نننن هههه وووو يييي ءءءء ئئئئ ؤؤؤؤ ٱٱٱٱ ىىىى ةةةة'

    # 1. a POLICY-CONFORMING split must produce nothing. If this fires, the checker is
    #    unusable regardless of what else it detects.
    clean = {'train': [sft('asas_albalagha_c0001', A)],
             'test':  [sft('dialect_dict_najdi_c0001', B)]}
    r = verify(clean, KNOWN)
    if not r['clean']:
        print('  [FAIL] a conforming split was reported as leaking:', r['counts'])
        ok = False

    # 2. the rule itself
    bad = {'train': [sft('asas_albalagha_c0001', A)],
           'test':  [sft('asas_albalagha_c0002', B)]}
    r = verify(bad, KNOWN)
    if r['counts'].get(DOC_IN_MULTIPLE_PARTITIONS) != 1:
        print('  [FAIL] one document across two partitions not caught:', r['counts'])
        ok = False

    # 3. FAIL LOUDLY on an unparseable id. The policy names this as the fail-open case:
    #    an id that cannot be assigned defaults to "different document" and hides a leak.
    r = verify({'train': [sft('no_chunk_suffix', A)],
                'test':  [sft('asas_albalagha_c0002', B)]}, KNOWN)
    if r['counts'].get(UNPARSEABLE_CHUNK_ID) != 1:
        print('  [FAIL] unparseable chunk id not reported:', r['counts'])
        ok = False
    #    ... and it must not be silently treated as its own document
    if r['clean']:
        print('  [FAIL] a run containing an unparseable id reported clean')
        ok = False

    # 4. a doc_id absent from the manifest. This is the real defect the first live run
    #    found: the split unit and the licence-tracked unit were different strings.
    r = verify({'train': [sft('some_other_book_c0001', A)]}, KNOWN)
    if r['counts'].get(UNKNOWN_DOC_ID) != 1:
        print('  [FAIL] doc_id outside the manifest not reported:', r['counts'])
        ok = False

    # 5. identical text across partitions, WITH the document rule satisfied. This is the
    #    82.8% cross-file case; a document-only checker reports clean here.
    r = verify({'train': [sft('asas_albalagha_c0001', A)],
                'test':  [dpo('dialect_dict_najdi_c0001', A, B)]}, KNOWN)
    xs = [f for f in r['findings'] if f['code'] == TEXT_CROSSES_PARTITIONS]
    if not xs or xs[0]['match'] != 'exact':
        print('  [FAIL] identical text across partitions not caught:', r['counts'])
        ok = False
    if r['counts'].get(DOC_IN_MULTIPLE_PARTITIONS):
        print('  [FAIL] the document rule should be SATISFIED in this case')
        ok = False

    # 6. near-identical, not identical, across partitions
    r = verify({'train': [sft('asas_albalagha_c0001', A)],
                'test':  [sft('dialect_dict_najdi_c0001', A + ' نننن')]}, KNOWN)
    near = [f for f in r['findings']
            if f['code'] == TEXT_CROSSES_PARTITIONS and f['match'] == 'near']
    if not near:
        print('  [FAIL] near-duplicate across partitions not caught:', r['counts'])
        ok = False

    # 7. genuinely different text across partitions must NOT be flagged
    r = verify({'train': [sft('asas_albalagha_c0001', A)],
                'test':  [sft('dialect_dict_najdi_c0001', C)]}, KNOWN)
    if [f for f in r['findings'] if f['code'] == TEXT_CROSSES_PARTITIONS]:
        print('  [FAIL] unrelated texts flagged as crossing partitions')
        ok = False

    # 8. half a preference pair is not a pair
    r = verify({'train': [{'source_chunk_id': 'asas_albalagha_c0001',
                           'chosen': A, 'rejected': ''}]}, KNOWN)
    if r['counts'].get(PAIR_SPLIT) != 1:
        print('  [FAIL] half a DPO pair not reported:', r['counts'])
        ok = False

    # 9. FAIL-CLOSED: feasibility mode must never look clean. It checked no partitions.
    f = feasibility({'sft': [sft('asas_albalagha_c0001', A)]}, KNOWN)
    if f.get('clean'):
        print('  [FAIL] feasibility mode reported a clean verdict it did not earn')
        ok = False
    if f['checked_partitions'] is not False:
        print('  [FAIL] feasibility mode must record that it checked no partitions')
        ok = False
    if f['splittable_units'] != 1 or f['three_way_split_possible']:
        print('  [FAIL] feasibility miscounted units:', f['splittable_units'])
        ok = False
    f2 = feasibility({'sft': [sft('asas_albalagha_c0001', A),
                              sft('dialect_dict_najdi_c0001', B)]}, KNOWN)
    if f2['splittable_units'] != 2 or f2['three_way_split_possible']:
        print('  [FAIL] two units must not admit a three-way split:', f2['splittable_units'])
        ok = False

    # 10. the parent-document regex, on the shapes the policy names
    for cid, want in (('dialect_dict_najdi_c0083', 'dialect_dict_najdi'),
                      ('asas_albalagha_c0001', 'asas_albalagha'),
                      ('majam_alkalimat_alshaabia_najd_c0000',
                       'majam_alkalimat_alshaabia_najd'),
                      ('asas_albalagha_c001', None),      # three digits, not four
                      ('asas_albalagha', None),
                      ('', None)):
        if parent_doc(cid) != want:
            print('  [FAIL] parent_doc(%r) = %r, expected %r' % (cid, parent_doc(cid), want))
            ok = False

    # 11. the duplicate machinery really is check_duplication's
    if cd.near_pairs.__module__ != 'check_duplication':
        print('  [FAIL] near_pairs has been restated locally')
        ok = False

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def _load(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--partition', action='append', default=[], metavar='NAME=PATH',
                    help='a released partition file; repeat for train/val/test')
    ap.add_argument('--sft-in', default=cd.SFT_IN)
    ap.add_argument('--dpo-in', default=cd.DPO_IN)
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--report', default=REPORT_PATH)
    ap.add_argument('--threshold', type=float, default=cd.DEFAULT_THRESHOLD)
    ap.add_argument('--shingle-size', type=int, default=cd.SHORT_TEXT_SHINGLE)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    known = known_doc_ids(args.manifest)

    if args.partition:
        parts = {}
        for spec in args.partition:
            if '=' not in spec:
                print('[leak] --partition needs NAME=PATH, got %r' % spec)
                sys.exit(2)
            name, path = spec.split('=', 1)
            if not os.path.exists(path):
                print('[leak] partition %r: no such file %s' % (name, path))
                sys.exit(2)
            parts[name] = _load(path)
        rep = verify(parts, known, args.threshold, args.shingle_size)
    else:
        inputs = {}
        if os.path.exists(args.sft_in):
            inputs['sft'] = _load(args.sft_in)
        if os.path.exists(args.dpo_in):
            inputs['dpo'] = _load(args.dpo_in)
        if not inputs:
            print('[leak] no partitions given and no adapted input found')
            sys.exit(2)
        rep = feasibility(inputs, known)

    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    if rep['mode'] == 'verify':
        print('[leak] VERIFY mode - %d partition(s)' % len(rep['partitions']))
        for p, info in rep['partitions'].items():
            print('        %-10s %5d records, documents: %s'
                  % (p, info['records'], ', '.join(sorted(info['documents'])) or '-'))
        print('[leak] findings:')
        for code in (UNPARSEABLE_CHUNK_ID, UNKNOWN_DOC_ID, DOC_IN_MULTIPLE_PARTITIONS,
                     PAIR_SPLIT, TEXT_CROSSES_PARTITIONS):
            print('        %-28s %d' % (code, rep['counts'].get(code, 0)))
        for f in rep['findings'][:10]:
            print('        - %s %s' % (f['code'], json.dumps(
                {k: v for k, v in f.items() if k not in ('code', 'text', 'occurrences')},
                ensure_ascii=False)[:150]))
        print('[leak] wrote %s' % args.report)
        print('[leak] %s' % ('PASS - no leakage under SPLIT_POLICY.md' if rep['clean']
                             else 'FAIL - see findings above'))
        sys.exit(0 if rep['clean'] else 1)

    print('[leak] FEASIBILITY mode - NO PARTITIONS WERE CHECKED')
    print('        inputs: %s' % ', '.join('%s=%d' % kv for kv in rep['inputs'].items()))
    print('[leak] splittable units (parent documents): %d' % rep['splittable_units'])
    for doc, n in rep['unit_sizes'].items():
        print('        %-38s %6d records' % (doc, n))
    print('        largest unit holds %.1f%% of all records' % rep['largest_unit_share'])
    print('[leak] under SPLIT_POLICY.md: two-way split %s, three-way split %s'
          % ('POSSIBLE' if rep['two_way_split_possible'] else 'IMPOSSIBLE',
             'POSSIBLE' if rep['three_way_split_possible'] else 'IMPOSSIBLE'))
    for code in (UNPARSEABLE_CHUNK_ID, UNKNOWN_DOC_ID):
        if rep['counts'].get(code):
            print('[leak] %-24s %d' % (code, rep['counts'][code]))
            for f in [x for x in rep['findings'] if x['code'] == code][:3]:
                print('        %s' % json.dumps(
                    {k: v for k, v in f.items() if k != 'code'}, ensure_ascii=False)[:200])
    print('[leak] wrote %s' % args.report)
    print('[leak] EXIT 2 - no partition check was performed. This is not a pass.')
    sys.exit(2)


if __name__ == '__main__':
    main()
