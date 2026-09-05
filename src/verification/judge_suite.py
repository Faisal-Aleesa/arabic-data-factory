# -*- coding: utf-8 -*-
"""Task 3, Group B - synthetic suite runner for the LLM judges.

Loads the hand-built probes in `tests/fixtures/judge_*_classical_lexicon.jsonl`, resolves
each one against the real corpus chunk it names, runs it through `judge_sft.py` or
`judge_dpo.py`, and scores the answers against the expectation recorded in the fixture.

مشغّل مجموعة الاختبارات اليدوية للحَكَم، ومقارنة النتائج بالتوقعات المسجّلة.

Modes, and which of them need a model
-------------------------------------
    --validate    fixtures well-formed, chunk ids resolve, expectations in vocabulary
    --dry-run     build every prompt and report size + projected cost
    --run         judge the suite through a backend and print the matrix
    --self-test   the runner's own logic

Only `--run` against a configured backend needs a credential. The other three are the
reason this module exists now rather than after the endpoint arrives: the fixtures, the
prompt assembly, the scoring and the cost projection are all model-independent, and
leaving them until a credential exists would mean writing them under time pressure with
real money running.

Fixtures are CLASSICAL only, and that is a licence decision
-----------------------------------------------------------
Every probe is built on `asas_albalagha` chunks, which are CC BY-SA 4.0, so these
fixtures are committable. A probe built on the dialect corpus would inherit its
rights-pending status and would have to be gitignored like the five existing
`tests/fixtures/*saudi_dialect*` files.

The cost is a real coverage gap and is not papered over: `wrong_register` on the DIALECT
side means dialectal-versus-MSA register, which is a different axis from the
classical-versus-casual register these probes test. Dialect register probes need to be
built separately, locally, and gitignored, before the judge is trusted on dialect pairs.

What a green run here does and does not establish
-------------------------------------------------
It establishes that prompts assemble, that answers parse, that verdicts fold correctly,
that position bias is detected, and that every failure path abstains rather than passing.

It does NOT establish that any model judges Arabic register or organisation well. That
needs a real model, and on `wrong_register` and `weak_organization` there is no automated
check to cross-validate against even then - see `judge_dpo.py`. Those two types need
human spot-checks before the judge is trusted at scale on them.
"""

from __future__ import unicode_literals

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge_client as jc              # noqa: E402
import judge_sft as jsft               # noqa: E402
import judge_dpo as jdpo               # noqa: E402
import check_similarity as csim        # noqa: E402
import check_dpo as cd                 # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SFT_FIXTURE = os.path.join(REPO, 'tests', 'fixtures',
                           'judge_sft_classical_lexicon.jsonl')
DPO_FIXTURE = os.path.join(REPO, 'tests', 'fixtures',
                           'judge_dpo_classical_lexicon.jsonl')

VALID_SFT_EXPECT = (jc.JUDGE_PASS, jc.JUDGE_FAIL, jc.JUDGE_REVIEW)
VALID_DPO_EXPECT = (jc.JUDGE_PASS, jc.JUDGE_FAIL, jc.JUDGE_REVIEW)


def load_fixture(path):
    rows = []
    with io.open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError('%s line %d: %s' % (path, i, exc))
    return rows


def load_corpus():
    """chunk_id -> record, for whichever corpora are present on this machine.

    Reuses check_similarity's CHUNKS map and loader rather than restating either. The
    dialect file is gitignored, so on a fresh clone only the classical corpus loads -
    which is exactly what these fixtures need.
    """
    chunks = {}
    missing = []
    for corpus, rel in sorted(csim.CHUNKS.items()):
        path = os.path.join(REPO, rel)
        if os.path.exists(path):
            chunks.update(csim.load_chunks(path))
        else:
            missing.append(corpus)
    return chunks, missing


# ------------------------------------------------------------------------------ validate

def validate(sft_rows, dpo_rows, chunks):
    """Structural checks that need no model. Returns a list of problem strings."""
    problems = []

    for r in sft_rows:
        lbl = r.get('label', '?')
        for k in ('label', 'source_chunk_id', 'instruction', 'response',
                  'format_type', 'expect_verdict'):
            if not r.get(k):
                problems.append('SFT %s: missing %r' % (lbl, k))
        if r.get('expect_verdict') not in VALID_SFT_EXPECT:
            problems.append('SFT %s: expect_verdict %r outside the vocabulary'
                            % (lbl, r.get('expect_verdict')))
        cid = r.get('source_chunk_id')
        if chunks and cid not in chunks:
            problems.append('SFT %s: chunk %r not in the corpus' % (lbl, cid))
        for d in r.get('expect_fail_dimensions', []):
            if d not in jsft.DIMENSIONS:
                problems.append('SFT %s: unknown dimension %r' % (lbl, d))
        # An expectation must be internally consistent, or the matrix means nothing.
        fails = r.get('expect_fail_dimensions', [])
        if r.get('expect_verdict') == jc.JUDGE_PASS and fails:
            problems.append('SFT %s: expects PASS but lists failing dimensions %s'
                            % (lbl, fails))
        if r.get('expect_verdict') == jc.JUDGE_FAIL and not fails:
            problems.append('SFT %s: expects FAIL but names no failing dimension' % lbl)

    for r in dpo_rows:
        lbl = r.get('label', '?')
        for k in ('label', 'prompt', 'chosen', 'rejected', 'source_chunk_id',
                  'rejection_type', 'expect_verdict'):
            if not r.get(k):
                problems.append('DPO %s: missing %r' % (lbl, k))
        if r.get('expect_verdict') not in VALID_DPO_EXPECT:
            problems.append('DPO %s: expect_verdict %r outside the vocabulary'
                            % (lbl, r.get('expect_verdict')))
        rt = r.get('rejection_type')
        if rt not in cd.REJECTION_TYPES:
            problems.append('DPO %s: %r is not a charter rejection type' % (lbl, rt))
        exp_dim = r.get('expect_corroborating_dimension')
        if rt in jdpo.CORROBORATING_DIMENSION and \
                exp_dim != jdpo.CORROBORATING_DIMENSION[rt]:
            problems.append('DPO %s: fixture says %r corroborates %r, module says %r'
                            % (lbl, rt, exp_dim, jdpo.CORROBORATING_DIMENSION[rt]))
        cid = r.get('source_chunk_id')
        if chunks and cid not in chunks:
            problems.append('DPO %s: chunk %r not in the corpus' % (lbl, cid))

    # The suite must actually cover the two types nothing else can check.
    covered = set(r.get('rejection_type') for r in dpo_rows)
    for t in jdpo.NO_AUTOMATED_COVERAGE:
        n = sum(1 for r in dpo_rows if r.get('rejection_type') == t)
        if n < 2:
            problems.append('DPO: %r has %d probe(s); it has ZERO automated coverage '
                            'elsewhere and needs at least 2' % (t, n))
    if not covered:
        problems.append('DPO: no rejection types present at all')

    # There must be at least one probe whose expected answer is not a pass, or a judge
    # that says PASS to everything would score 100%.
    if not any(r.get('expect_verdict') != jc.JUDGE_PASS for r in sft_rows):
        problems.append('SFT: every probe expects PASS - the suite cannot discriminate')
    if not any(r.get('expect_verdict') != jc.JUDGE_PASS for r in dpo_rows):
        problems.append('DPO: every probe expects PASS - the suite cannot discriminate')
    return problems


# ------------------------------------------------------------------------------- dry run

def dry_run(sft_rows, dpo_rows, chunks, price_in=None, price_out=None):
    """Build every prompt; report size and projected cost without calling anything."""
    calls = 0
    in_tok = 0
    for r in sft_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        s, u = jsft.build_prompt(r, chunk.get('chunk_text', ''))
        in_tok += jc._approx_tokens(s) + jc._approx_tokens(u)
        calls += 1
    for r in dpo_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        pos = jdpo.chosen_position(r['source_chunk_id'])
        s, u = jdpo.build_prompt(r, chunk.get('chunk_text', ''), pos)
        n = jc._approx_tokens(s) + jc._approx_tokens(u)
        passes = 2 if r.get('rejection_type') in jdpo.NO_AUTOMATED_COVERAGE else 1
        in_tok += n * passes
        calls += passes
    out_tok = calls * 300      # a filled verdict object, measured against the rubric
    cost = jc.estimate_cost(in_tok, out_tok, price_in, price_out)
    return {'calls': calls, 'input_tokens': in_tok, 'output_tokens_est': out_tok,
            'estimated_cost': cost,
            'cost_basis': ('per-MTok in=%s out=%s' % (price_in, price_out)
                           if cost is not None else
                           'UNKNOWN - supply --price-in/--price-out for the chosen model')}


# ----------------------------------------------------------------------------- run + score

def run_suite(sft_rows, dpo_rows, chunks, backend, ledger=None):
    results = []
    for r in sft_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        res = jsft.judge_sft_record(r, chunk.get('chunk_text', ''), backend,
                                    ledger=ledger, probe_id=r['label'].split()[0])
        results.append(('SFT', r, res))
    for r in dpo_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        res = jdpo.judge_dpo_pair(r, chunk.get('chunk_text', ''), backend,
                                  ledger=ledger, probe_id=r['label'].split()[0])
        results.append(('DPO', r, res))
    return results


def score(results):
    """Compare each verdict against the fixture's expectation.

    An ABSTAIN is never scored as a match, even against an expected REVIEW: they are
    different outcomes and conflating them would hide a broken backend as agreement.
    """
    rows = []
    hits = 0
    scored = 0
    abstained = 0
    for kind, fix, res in results:
        got = res.get('judge_verdict')
        want = fix.get('expect_verdict')
        if got == jc.JUDGE_ABSTAIN:
            abstained += 1
            status = 'ABSTAIN'
        else:
            scored += 1
            if got == want:
                hits += 1
                status = 'match'
            else:
                status = 'MISS'
        rows.append({'kind': kind, 'label': fix.get('label'), 'expect': want,
                     'got': got, 'status': status,
                     'reason': res.get('reason'),
                     'abstain_reason': res.get('abstain_reason'),
                     'position_biased': res.get('position_biased'),
                     'corroborated': res.get('corroborated')})
    return {'rows': rows, 'scored': scored, 'hits': hits, 'abstained': abstained,
            'accuracy': (float(hits) / scored) if scored else None}


def print_matrix(sc, ledger=None):
    out = sys.stdout
    out.write('\n%-4s %-58s %-12s %-12s %s\n'
              % ('kind', 'probe', 'expected', 'got', 'status'))
    out.write('-' * 104 + '\n')
    for r in sc['rows']:
        out.write('%-4s %-58s %-12s %-12s %s\n'
                  % (r['kind'], (r['label'] or '')[:58],
                     (r['expect'] or '').replace('JUDGE_', ''),
                     (r['got'] or '').replace('JUDGE_', ''), r['status']))
        if r['status'] == 'MISS':
            out.write('     -> %s\n' % (r['reason'] or ''))
        if r['status'] == 'ABSTAIN':
            out.write('     -> %s: %s\n' % (r['abstain_reason'], r['reason'] or ''))
    out.write('-' * 104 + '\n')
    if sc['accuracy'] is None:
        out.write('NO PROBE WAS SCORED - every call abstained (%d). '
                  'This is a backend or prompt failure, not a judge result.\n'
                  % sc['abstained'])
    else:
        out.write('scored %d/%d probes, %d matched (%.0f%%), %d abstained\n'
                  % (sc['scored'], len(sc['rows']), sc['hits'],
                     100.0 * sc['accuracy'], sc['abstained']))
    if ledger is not None:
        s = ledger.summary()
        out.write('usage: %d calls, %d in / %d out tokens, cost %s (%s)\n'
                  % (s['calls'], s['input_tokens'], s['output_tokens'],
                     s['estimated_cost'], s['cost_basis']))


# ------------------------------------------------------------------------------ self-test

def run_self_test():
    ok = True
    sft = load_fixture(SFT_FIXTURE)
    dpo = load_fixture(DPO_FIXTURE)
    chunks, missing = load_corpus()

    if len(sft) < 10:
        print('  [FAIL] SFT suite has %d probes, the brief asks for 10-15' % len(sft))
        ok = False

    problems = validate(sft, dpo, chunks)
    for p in problems:
        print('  [FAIL] %s' % p)
        ok = ok and False
    if problems:
        ok = False

    # Scoring must not count an abstention as agreement with an expected REVIEW.
    fake = [('SFT', {'label': 'x', 'expect_verdict': jc.JUDGE_REVIEW},
             {'judge_verdict': jc.JUDGE_ABSTAIN})]
    sc = score(fake)
    if sc['hits'] != 0 or sc['abstained'] != 1 or sc['accuracy'] is not None:
        print('  [FAIL] an abstention was scored as a match against expected REVIEW')
        ok = False

    # An all-abstain run must report no accuracy rather than 0% or 100%.
    nb = jc.make_backend('qwen')
    led = jc.UsageLedger()
    res = run_suite(sft, dpo, chunks, nb, ledger=led)
    sc = score(res)
    if sc['accuracy'] is not None:
        print('  [FAIL] unconfigured backend produced an accuracy figure')
        ok = False
    if sc['abstained'] != len(res):
        print('  [FAIL] unconfigured backend did not abstain on every probe (%d/%d)'
              % (sc['abstained'], len(res)))
        ok = False

    # A scripted backend that always says "pass" must NOT score 100% - the suite has to
    # contain probes that such a judge gets wrong, or it measures nothing.
    always_pass_sft = json.dumps(
        {'dimensions': {d: {'verdict': 'pass', 'evidence': ''} for d in jsft.DIMENSIONS},
         'confidence': 'high'})
    by_key = {}
    for r in sft:
        by_key[r['label'].split()[0]] = always_pass_sft
    for r in dpo:
        pid = r['label'].split()[0]
        pos = jdpo.chosen_position(r['source_chunk_id'])
        payload = json.dumps({'dimensions': {d: pos for d in jdpo.DIMENSIONS},
                              'overall': pos, 'evidence': '', 'confidence': 'high'})
        by_key[pid] = payload
        by_key[pid + '_swap'] = payload      # same slot both times => position bias
    be = jc.ScriptedBackend(by_key=by_key)
    sc = score(run_suite(sft, dpo, chunks, be))
    if sc['accuracy'] is None or sc['accuracy'] > 0.75:
        print('  [FAIL] a judge that passes everything scored %s - the suite does not '
              'discriminate' % sc['accuracy'])
        ok = False

    # That same run must have caught position bias on every zero-coverage probe.
    biased = [r for r in sc['rows'] if r['position_biased']]
    n_zero = sum(1 for r in dpo if r.get('rejection_type') in jdpo.NO_AUTOMATED_COVERAGE)
    if len(biased) != n_zero:
        print('  [FAIL] position bias caught on %d of %d zero-coverage probes'
              % (len(biased), n_zero))
        ok = False

    # Dry run must refuse to invent a cost.
    d = dry_run(sft, dpo, chunks)
    if d['estimated_cost'] is not None or 'UNKNOWN' not in d['cost_basis']:
        print('  [FAIL] dry run reported a cost with no pricing supplied')
        ok = False
    if d['calls'] != len(sft) + len(dpo) + n_zero:
        print('  [FAIL] dry run call count %d does not match the swap-check plan'
              % d['calls'])
        ok = False

    if missing:
        print('  [note] corpora not present on this machine: %s' % ', '.join(missing))
    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Synthetic suite runner for the LLM judges.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--backend', default='qwen')
    ap.add_argument('--transcript', help='record/replay JSONL for offline reruns')
    ap.add_argument('--replay-only', action='store_true')
    ap.add_argument('--price-in', type=float)
    ap.add_argument('--price-out', type=float)
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if run_self_test() else 1)

    sft_rows = load_fixture(SFT_FIXTURE)
    dpo_rows = load_fixture(DPO_FIXTURE)
    chunks, missing = load_corpus()
    if missing:
        sys.stderr.write('note: corpora not on this machine: %s\n' % ', '.join(missing))

    if a.validate:
        probs = validate(sft_rows, dpo_rows, chunks)
        for p in probs:
            print('PROBLEM: %s' % p)
        print('%d SFT + %d DPO probes, %d problem(s)'
              % (len(sft_rows), len(dpo_rows), len(probs)))
        sys.exit(1 if probs else 0)

    if a.dry_run:
        print(json.dumps(dry_run(sft_rows, dpo_rows, chunks, a.price_in, a.price_out),
                         ensure_ascii=False, indent=2))
        sys.exit(0)

    if a.run:
        backend = jc.make_backend(a.backend)
        if a.transcript:
            backend = jc.CachingBackend(backend, a.transcript,
                                        replay_only=a.replay_only)
        led = jc.UsageLedger(a.price_in, a.price_out)
        sc = score(run_suite(sft_rows, dpo_rows, chunks, backend, ledger=led))
        print_matrix(sc, led)
        sys.exit(0)

    ap.print_help()
