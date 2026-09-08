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

Two fixture sets, split by licence
----------------------------------
The CLASSICAL probes are built on `asas_albalagha` chunks (CC BY-SA 4.0) and are
committed. The DIALECT probes quote the rights-pending corpus and are gitignored by
`tests/fixtures/*saudi_dialect*`, the same rule holding back the fact-checker's dialect
fixtures. They load when present and are skipped when absent, so a fresh clone runs the
classical suite and says what it could not load. `validate()` enforces the split as an
invariant: a probe citing a `dialect_dict_*` chunk that sits in a non-ignored fixture is
a reported problem, not a filename convention.

Why the dialect set is not redundant: the register axis is different
--------------------------------------------------------------------
The classical probes test classical-versus-casual register. The dialect probes test
dialectal-versus-MSA, and that axis has a failure mode the classical set structurally
cannot reach, tested in both directions:

  direction 1  the metalanguage collapses into the object language - the explanation is
               written in dialect rather than describing dialect. Analogous to the
               classical case.
  direction 2  OVER-MSA-IFICATION, and this is the one that matters. The response
               rewrites the attested dialectal form into fluent MSA. It reads as BETTER
               Arabic and is lexicographically worthless, because the attested form is
               precisely the data a dialect dictionary exists to record.

A judge applying the ordinary heuristic "more formal Arabic is better" scores direction 2
backwards. On the classical corpus that heuristic is correct, so no classical probe can
expose it. REG-D-03, REG-D-04 and SFT-D-03 are built for exactly this, and REG-D-06 is a
negative control: both sides in identical register, differing only in coverage, so a
judge that routes every difference through register corroborates on the wrong dimension.

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

# Dialect probes. Present only on a machine that has the rights-pending corpus: both
# files are gitignored by `tests/fixtures/*saudi_dialect*`, the same rule that holds back
# the fact-checker's dialect fixtures. They load when present and are skipped silently
# when absent, exactly like the dialect corpus itself - so a fresh clone runs the
# classical suite and reports what it could not load, rather than failing.
SFT_FIXTURE_DIALECT = os.path.join(REPO, 'tests', 'fixtures',
                                   'judge_sft_saudi_dialect.jsonl')
DPO_FIXTURE_DIALECT = os.path.join(REPO, 'tests', 'fixtures',
                                   'judge_dpo_saudi_dialect.jsonl')

# A probe naming a dialect chunk quotes rights-pending text and may live ONLY in a file
# the ignore rule covers. validate() enforces this rather than trusting the filename.
DIALECT_CHUNK_PREFIX = 'dialect_dict_'
RIGHTS_HELD_MARKER = 'saudi_dialect'

VALID_SFT_EXPECT = (jc.JUDGE_PASS, jc.JUDGE_FAIL, jc.JUDGE_REVIEW)
VALID_DPO_EXPECT = (jc.JUDGE_PASS, jc.JUDGE_FAIL, jc.JUDGE_REVIEW)


def load_fixture(path, optional=False):
    """Probe rows, each tagged with the file it came from.

    `_fixture` is stamped on every row so validate() can check the rights invariant: a
    probe quoting a dialect chunk must have come from a gitignored file. With `optional`,
    a missing file yields no rows instead of raising - that is how the dialect probes stay
    invisible on a clone that does not have them.
    """
    if optional and not os.path.exists(path):
        return []
    rows = []
    with io.open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise ValueError('%s line %d: %s' % (path, i, exc))
            rec['_fixture'] = os.path.basename(path)
            rows.append(rec)
    return rows


def load_all():
    """(sft_rows, dpo_rows, loaded_names, skipped_names) across classical + dialect."""
    loaded, skipped = [], []
    sft, dpo = [], []
    for path, target in ((SFT_FIXTURE, sft), (DPO_FIXTURE, dpo)):
        target.extend(load_fixture(path))
        loaded.append(os.path.basename(path))
    for path, target in ((SFT_FIXTURE_DIALECT, sft), (DPO_FIXTURE_DIALECT, dpo)):
        rows = load_fixture(path, optional=True)
        (loaded if rows else skipped).append(os.path.basename(path))
        target.extend(rows)
    return sft, dpo, loaded, skipped


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


# ------------------------------------------------- the merged DPO path (src/dpo)
#
# The probe suite has to validate whatever actually ships. After the LLM_Judge delivery
# was integrated, the shipping DPO judge is `src/dpo/llm_judge.judge_pair()` behind
# `judge_pipeline.adjudicate()` - NOT the standalone `judge_dpo.judge_dpo_pair()` these
# probes were originally written against. A test asset pointed at a module that no longer
# runs in production is worse than no test: it reports green about code nobody executes.
#
# So `--dpo-path merged` (the default) drives the delivery's judge, and `legacy` keeps the
# standalone one available for comparison. The gameability and discrimination probes are
# SFT-side and always run through `judge_sft.py`, which the delivery did not replace.

DPO_PATH_MERGED = 'merged'
DPO_PATH_LEGACY = 'legacy'


def _merged_modules():
    """Import the delivery package lazily; it is optional for a classical-only run."""
    p = os.path.join(REPO, 'src', 'dpo')
    if p not in sys.path:
        sys.path.insert(0, p)
    import judge_contract as jcon
    import llm_judge as lj
    import judge_provider as jprov
    return jcon, lj, jprov


def _to_our_verdict(res):
    """Map the delivery's JudgeResult onto this suite's verdict vocabulary.

    `unavailable/PositionBias` maps to REVIEW rather than ABSTAIN on purpose: the judge
    did answer twice, and the refusal is a finding about the judge, not a failure to
    reach it. That matches what the standalone judge_dpo.py reported for the same
    condition, so a matrix produced before and after the merge stays comparable.
    """
    if res.status == 'ok':
        a = res.judge_output.assessment
        if a == 'chosen_better':
            return jc.JUDGE_PASS
        if a == 'rejected_better':
            return jc.JUDGE_FAIL
        return jc.JUDGE_REVIEW
    if res.error_type == 'PositionBias':
        return jc.JUDGE_REVIEW
    return jc.JUDGE_ABSTAIN


def judge_dpo_merged(pair, chunk_text, backend, ledger=None, probe_id=None):
    """Judge one fixture pair through the delivery's judge, over our HTTP backend."""
    jcon, lj, jprov = _merged_modules()
    client = jprov.BackendJudgeClient(
        jprov.ClientConfig.from_env(provider='suite', model=getattr(backend, 'model_id',
                                                                   'unknown')),
        backend=backend)
    ji = jcon.JudgeInput(
        prompt=pair.get('prompt', ''),
        chosen=pair.get('chosen', ''),
        rejected=pair.get('rejected', ''),
        source_chunk_id=pair.get('source_chunk_id', ''),
        source_region=pair.get('source_region', ''),
        rejection_type=pair.get('rejection_type', ''),
        model_version=pair.get('model_version', 'suite'),
        format_type=pair.get('format_type'),
        source_text=chunk_text or '',
        deterministic_signals={},
    )
    before = getattr(backend, '_suite_calls', 0)
    res = lj.judge_pair(ji, client, case_label=probe_id)
    if ledger is not None:
        ledger.record(None, abstained=(res.status != 'ok'))
    return {'judge_verdict': _to_our_verdict(res),
            'reason': res.reason or (res.judge_output.assessment
                                     if res.judge_output else None),
            'error_type': res.error_type,
            'position_biased': (res.error_type == 'PositionBias') or None,
            'corroborated': (res.judge_output.rejection_type_agrees_with_record
                             if res.judge_output else None),
            'passes': 2 if pair.get('rejection_type') in lj.SWAP_CHECK_TYPES else 1,
            'dpo_path': DPO_PATH_MERGED}


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

    # RIGHTS INVARIANT. A probe naming a dialect chunk quotes rights-pending source text,
    # so it may live only in a fixture the ignore rule covers. Checked here rather than
    # left to the filename, because the failure is silent and expensive: a dialect probe
    # added to the classical fixture would be committed on the next commit, and the
    # licence audit would only catch it if the quoted run happened to be long enough.
    for r in list(sft_rows) + list(dpo_rows):
        cid = r.get('source_chunk_id') or ''
        fx = r.get('_fixture') or '(unknown)'
        if cid.startswith(DIALECT_CHUNK_PREFIX) and RIGHTS_HELD_MARKER not in fx:
            problems.append(
                'RIGHTS: %s cites dialect chunk %r but lives in %r, which the ignore '
                'rule tests/fixtures/*saudi_dialect* does NOT cover - it would be '
                'committed' % (r.get('label', '?'), cid, fx))

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

def run_suite(sft_rows, dpo_rows, chunks, backend, ledger=None,
              dpo_path=DPO_PATH_MERGED):
    results = []
    for r in sft_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        res = jsft.judge_sft_record(r, chunk.get('chunk_text', ''), backend,
                                    ledger=ledger, probe_id=r['label'].split()[0])
        results.append(('SFT', r, res))
    for r in dpo_rows:
        chunk = chunks.get(r['source_chunk_id'], {})
        pid = r['label'].split()[0]
        if dpo_path == DPO_PATH_MERGED:
            res = judge_dpo_merged(r, chunk.get('chunk_text', ''), backend,
                                   ledger=ledger, probe_id=pid)
        else:
            res = jdpo.judge_dpo_pair(r, chunk.get('chunk_text', ''), backend,
                                      ledger=ledger, probe_id=pid)
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
    sft, dpo, loaded, skipped = load_all()
    chunks, missing = load_corpus()

    if len(load_fixture(SFT_FIXTURE)) < 10:
        print('  [FAIL] classical SFT suite has %d probes, the brief asks for 10-15'
              % len(load_fixture(SFT_FIXTURE)))
        ok = False

    # The rights guard must actually fire. Simulate a dialect probe smuggled into the
    # committed classical fixture - the exact mistake it exists to catch.
    smuggled = [{'label': 'X', 'source_chunk_id': 'dialect_dict_najdi_c0065',
                 'instruction': 'i', 'response': 'r', 'format_type': 'dictionary_entry',
                 'expect_verdict': jc.JUDGE_PASS, 'expect_fail_dimensions': [],
                 '_fixture': 'judge_sft_classical_lexicon.jsonl'}]
    if not any('RIGHTS:' in p for p in validate(smuggled, [], {})):
        print('  [FAIL] the rights invariant did not fire on a dialect probe placed in '
              'a committed fixture')
        ok = False
    # ...and must NOT fire when the same probe sits in the gitignored fixture.
    okfile = [dict(smuggled[0], _fixture='judge_sft_saudi_dialect.jsonl')]
    if any('RIGHTS:' in p for p in validate(okfile, [], {})):
        print('  [FAIL] the rights invariant fired on a correctly-ignored fixture')
        ok = False

    # If the dialect probes are present, they must cover BOTH register directions.
    if any(r.get('_fixture', '').find(RIGHTS_HELD_MARKER) >= 0 for r in dpo):
        dreg = [r for r in dpo if r.get('rejection_type') == 'wrong_register'
                and RIGHTS_HELD_MARKER in r.get('_fixture', '')]
        if len(dreg) < 4:
            print('  [FAIL] dialect register probes: %d, expected at least 4 (two per '
                  'direction)' % len(dreg))
            ok = False
        if not any('TRAP' in r.get('label', '') for r in dreg):
            print('  [FAIL] no over-MSA-ification TRAP probe among the dialect register '
                  'set - the failure mode the classical probes cannot reach is untested')
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
    sc = score(run_suite(sft, dpo, chunks, be, dpo_path=DPO_PATH_LEGACY))
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

    # The MERGED path is what ships, so the suite must be able to drive it and must still
    # catch position bias through it. Scripted in the DELIVERY's wire format, which is
    # different from the legacy block above - that difference is the whole reason this
    # assertion exists separately rather than being folded into it.
    import json as _json
    zero_cov = [r for r in dpo if r.get('rejection_type') in jdpo.NO_AUTOMATED_COVERAGE]
    if zero_cov:
        r0 = zero_cov[0]
        pid = r0['label'].split()[0]
        same_slot = _json.dumps({
            'schema_version': 'judge-1', 'judge_model_version': 'suite/scripted',
            'assessment': 'chosen_better', 'weakness_present': False,
            'rejection_type': None, 'rejection_type_agrees_with_record': None,
            'type_scores': {}, 'confidence': 0.9, 'evidence': [], 'notes': None})
        be_m = jc.ScriptedBackend(by_key={pid: same_slot, pid + '_swap': same_slot})
        res = judge_dpo_merged(r0, chunks.get(r0['source_chunk_id'], {}).get('chunk_text', ''),
                               be_m, probe_id=pid)
        if not res.get('position_biased') or res['judge_verdict'] != jc.JUDGE_REVIEW:
            print('  [FAIL] merged DPO path did not catch position bias on %r: %s'
                  % (pid, res))
            ok = False
        if res.get('dpo_path') != DPO_PATH_MERGED:
            print('  [FAIL] merged path did not label itself')
            ok = False
        # a covered type must still cost one call through the merged path
        covered = [r for r in dpo
                   if r.get('rejection_type') not in jdpo.NO_AUTOMATED_COVERAGE]
        if covered:
            rc = covered[0]
            pidc = rc['label'].split()[0]
            good = _json.dumps({
                'schema_version': 'judge-1', 'judge_model_version': 'suite/scripted',
                'assessment': 'chosen_better', 'weakness_present': False,
                'rejection_type': None, 'rejection_type_agrees_with_record': None,
                'type_scores': {}, 'confidence': 0.9, 'evidence': [], 'notes': None})
            be_c = jc.ScriptedBackend(by_key={pidc: good})
            rr = judge_dpo_merged(rc, chunks.get(rc['source_chunk_id'], {}).get('chunk_text', ''),
                                  be_c, probe_id=pidc)
            if rr['judge_verdict'] != jc.JUDGE_PASS or rr.get('passes') != 1:
                print('  [FAIL] merged path on a covered type: %s' % rr)
                ok = False

    if missing:
        print('  [note] corpora not present on this machine: %s' % ', '.join(missing))
    print('  [note] fixtures loaded: %s' % ', '.join(loaded))
    if skipped:
        print('  [note] fixtures absent (gitignored): %s' % ', '.join(skipped))
    print('  [note] %d SFT + %d DPO probes in this run' % (len(sft), len(dpo)))
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
    ap.add_argument('--dpo-path', choices=[DPO_PATH_MERGED, DPO_PATH_LEGACY],
                    default=DPO_PATH_MERGED,
                    help='which DPO judge to exercise; merged = the shipping src/dpo path')
    ap.add_argument('--transcript', help='record/replay JSONL for offline reruns')
    ap.add_argument('--replay-only', action='store_true')
    ap.add_argument('--price-in', type=float)
    ap.add_argument('--price-out', type=float)
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if run_self_test() else 1)

    sft_rows, dpo_rows, loaded, skipped = load_all()
    chunks, missing = load_corpus()
    sys.stderr.write('fixtures loaded: %s\n' % ', '.join(loaded))
    if skipped:
        sys.stderr.write('fixtures NOT on this machine (gitignored, expected on a '
                         'fresh clone): %s\n' % ', '.join(skipped))
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
        sc = score(run_suite(sft_rows, dpo_rows, chunks, backend, ledger=led,
                             dpo_path=a.dpo_path))
        sys.stderr.write('DPO path: %s\\n' % a.dpo_path)
        print_matrix(sc, led)
        sys.exit(0)

    ap.print_help()
