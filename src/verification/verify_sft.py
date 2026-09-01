# -*- coding: utf-8 -*-
"""Task 3 - Combined SFT verifier.

Composes the three checks already built into one consolidated verdict:

    extract_facts.py    ground-truth fact tables (built offline, read here)
    check_facts.py      fact-level contradiction / support
    check_similarity.py semantic grounding against the source chunk

Input : an SFT record - instruction, response, source_chunk_id, source_region,
        format_type, model_version
Output: a verdict object carrying BOTH component signals and a suggested verdict

Combination logic, and why it is not an average
-----------------------------------------------
The two signals catch different failures and are not commensurable, so averaging them
would let a high similarity score dilute a hard factual error. The rules are asymmetric
on purpose:

  CONTRADICTED on a PRECISE fact -> FAIL_CONTRADICTED, at any similarity. A wrong year
  is wrong however fluent and however well-grounded the surrounding prose reads.
  Similarity is recorded but cannot rescue it. This is the single most important rule
  here: the dangerous response is the one that looks grounded and is not. Verified
  directly - two composite probes with identical prose and identical similarity
  (pct 98.3) split PASS / FAIL_CONTRADICTED on one changed digit.

  CONTRADICTED on region_claim ALONE -> REVIEW/fail, not a hard failure. Region claims
  are matched against cross_dialect_reference facts, whose extraction recall is poor: a
  probe naming two regions the chunk genuinely discusses was hard-failed because the
  extractor had recorded those references only as truncated fragments. A hard reject
  should rest on precise evidence, not on a heuristic built over a lossy field.

  SUPPORTED + high similarity -> PASS. Both signals agree.

  UNSUPPORTED / PARTIAL, or the two signals disagreeing -> REVIEW, never a hard reject.
  Both underlying tools already treat their own uncertainty this way - check_facts.py
  routes its low-confidence tail to a review CSV, check_similarity.py documents that a
  low score is evidence of drift rather than proof - and it would be incoherent for the
  wrapper to be more decisive than the evidence it is built on. REVIEW carries a
  `leaning` field so a human queue can be ordered.

  No extractable facts -> NO_FACT_COVERAGE, which is deliberately NOT a pass or a fail.
  check_facts.py had nothing to check, so the fact axis is silent, not satisfied. Folding
  that into PASS would launder a coverage gap into an endorsement; folding it into FAIL
  would punish a response for the extractor's recall. It is reported as its own outcome
  with the similarity score attached.

Thresholds are parameters, not constants
----------------------------------------
No pass/fail cutoff is baked into the logic. The similarity bands live in
SimilarityBands, default to provisional values derived from 12 hand-written probes, are
overridable per call and on the command line, and are echoed into every verdict
(`bands_used`) so no score can be read without the boundaries that classified it.
They are a starting point for discussion, not a calibrated threshold - see
check_similarity.py's caveats, especially that a correct paraphrase with no shared
wording scored pct 96 on one corpus and pct 20 on the other.

Built for DPO reuse
-------------------
`verify_response()` is the unit of work and knows nothing about SFT records: it takes a
response string plus a chunk id and returns a verdict. `verify_sft_record()` is a thin
adapter that validates the SFT schema and delegates. `compare_responses()` runs the same
per-response function over two candidates and reports which is better and why, which is
what the DPO stage needs for chosen-vs-rejected. Building the pipeline around a
per-response function rather than SFT plumbing is the whole reason DPO will not need a
rewrite.

A VerificationContext holds the facts index, the chunk texts, the embedder and a
fixed-seed baseline sample. It is built once and reused across every candidate, which
matters because the baseline embedding is the expensive part.

What this CANNOT do yet
-----------------------
- NO FORMAT VALIDATION. `format_type` is carried through and checked for presence only.
  Whether the response actually conforms to its declared format is a separate task that
  has not been built. A malformed response with good facts and good similarity will pass
  here.
- It inherits every limitation of the tools it wraps:
  * extraction recall - a fact extract_facts.py missed is not in the table, so a response
    asserting it reads as UNSUPPORTED. UNSUPPORTED counts must be read against
    docs/citation_review_*.csv, not as a hallucination rate.
  * drift and hallucination are not separable by similarity; both read as ungrounded.
  * on the dialect corpus similarity partly tracks lexical overlap, so a correct
    reworded paraphrase can score like a hallucination.
- Similarity requires torch (requirements-verification.txt). Without it the module still
  runs, but every verdict is marked `similarity_available: false` and no PASS is issued -
  a single-signal result is reported as REVIEW rather than quietly presented as a pass.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_facts as cf                                        # noqa: E402
import check_similarity as csim                                 # noqa: E402

CHUNKS = csim.CHUNKS
FACTS = cf.FACTS

# Verdicts
PASS = 'PASS'
FAIL_CONTRADICTED = 'FAIL_CONTRADICTED'
REVIEW = 'REVIEW'
NO_FACT_COVERAGE = 'NO_FACT_COVERAGE'
INVALID_RECORD = 'INVALID_RECORD'

SFT_FIELDS = ('instruction', 'response', 'source_chunk_id', 'source_region',
              'format_type', 'model_version')


class SimilarityBands(object):
    """Provisional percentile bands. NOT calibrated - see the module docstring.

    Derived from 12 hand-written probes: grounded landed at pct 99-100 on both corpora,
    hallucinated at 1-40, with drift and ambiguous scattered between. The middle band is
    deliberately wide and routes to REVIEW. Widening it is the safe direction to adjust;
    narrowing it manufactures confidence the data does not support.

    `high` was 90 and is now 95, on evidence. A composite probe with the correct year but
    entirely wrong subject matter (falconry against a chunk about speech verbs) scored
    pct 90.8 and passed - a false PASS. The same text WITHOUT the year clause scored 31.9,
    so one short formulaic sentence moved it nearly 60 points: this measure is sensitive
    to shared boilerplate, and a boundary at 90 sat inside that noise. 95 excludes the
    observed false pass while leaving every genuinely grounded probe (98.3-100) above it.
    That is one data point, not a calibration - it is still a parameter, and real
    reconstructed output should decide it.
    """

    def __init__(self, high=95.0, low=50.0):
        if not 0 <= low <= high <= 100:
            raise ValueError('bands must satisfy 0 <= low <= high <= 100')
        self.high, self.low = float(high), float(low)

    def band(self, pct):
        if pct is None:
            return 'unavailable'
        if pct >= self.high:
            return 'high'
        if pct < self.low:
            return 'low'
        return 'middle'

    def as_dict(self):
        return {'high': self.high, 'low': self.low}


class VerificationContext(object):
    """Everything the per-response check needs, built once and reused.

    Holds the fact table, the chunk texts, the embedder and a fixed-seed baseline sample
    of chunks used to turn a raw cosine into a percentile. The baseline is the expensive
    part - embedding it once and reusing it is what makes per-candidate checks cheap, and
    it is what lets DPO score two candidates against an identical reference.
    """

    def __init__(self, corpus, facts_path=None, chunks_path=None, embedder=None,
                 baseline_n=120, bands=None, seed=0):
        self.corpus = corpus
        self.facts = cf.load_index(facts_path or FACTS[corpus])
        self.chunks = csim.load_chunks(chunks_path or CHUNKS[corpus])
        self.bands = bands or SimilarityBands()
        self.embedder = embedder
        self.index = csim.WindowIndex(embedder) if embedder is not None else None
        rng = random.Random(seed)
        ids = sorted(self.chunks)
        self.baseline_ids = rng.sample(ids, min(baseline_n, len(ids)))

    @property
    def similarity_available(self):
        return self.index is not None

    def similarity(self, response, chunk_id):
        """Raw cosine plus percentile/margin against the fixed baseline."""
        if not self.similarity_available:
            return None
        rec = self.chunks.get(chunk_id)
        if rec is None:
            return None
        own = self.index.score(response, chunk_id, rec['chunk_text'])
        base = [self.index.score(response, b, self.chunks[b]['chunk_text'])['max']
                for b in self.baseline_ids if b != chunk_id]
        base_sorted = sorted(base)
        n = len(base_sorted)
        median = base_sorted[n // 2] if n else float('nan')
        pct = 100.0 * sum(1 for x in base_sorted if x < own['max']) / n if n else None
        return {'max': own['max'], 'mean': own['mean'], 'n_windows': own['n_windows'],
                'baseline_n': n, 'baseline_median': median,
                'margin': own['max'] - median if n else None,
                'percentile': pct}


# ------------------------------------------------------------------ the core decision

# Fact types whose contradiction is PRECISE evidence. A year, a page number or a lemma
# either matches the source or does not.
#
# `region_claim` is deliberately excluded. It is a heuristic built on
# cross_dialect_reference facts, and that extraction has poor recall: a chunk discussing
# other regions may record them only as truncated fragments. Testing hit exactly this -
# a correct response naming two regions the chunk genuinely discusses was hard-failed,
# because the extractor had stored those references truncated. Escalating an extraction
# recall gap into the most severe verdict is the wrong trade, so a region-only
# contradiction routes to REVIEW instead, with the cause named.
PRECISE_CONTRADICTION_TYPES = {
    'hijri_year', 'gregorian_year', 'year_unmarked_era', 'page_reference',
    'entry_headword', 'entry_root', 'citation_authority',
}


def combine(fact_verdict, sim_band, similarity_available, contradicted_types=()):
    """Pure decision function: two signals in, (verdict, leaning, reason) out.

    Kept free of IO and of the embedder so the whole matrix can be self-tested without a
    model, and so the rule can be read in one place rather than inferred from control
    flow scattered through the caller.
    """
    if fact_verdict == 'UNKNOWN_CHUNK':
        return INVALID_RECORD, None, 'source_chunk_id not present in the fact table'

    # Rule 1: a contradicted fact is a failure at any similarity - but only when the
    # contradiction rests on a precise fact type. See PRECISE_CONTRADICTION_TYPES.
    if fact_verdict == cf.CONTRADICTED:
        types = set(contradicted_types or ())
        precise = types & PRECISE_CONTRADICTION_TYPES
        if precise or not types:
            return (FAIL_CONTRADICTED, 'fail',
                    'a response fact conflicts with the source (%s); similarity cannot '
                    'rescue this' % (', '.join(sorted(precise)) if precise else 'unspecified'))
        return (REVIEW, 'fail',
                'only heuristic contradiction(s) (%s), which depend on cross-reference '
                'extraction recall - flagged rather than hard-failed'
                % ', '.join(sorted(types)))

    if fact_verdict == 'NO_CHECKABLE_CLAIMS':
        if not similarity_available:
            return (NO_FACT_COVERAGE, None,
                    'no checkable facts and no similarity signal: nothing was verified')
        lean = {'high': 'pass', 'low': 'fail'}.get(sim_band)
        return (NO_FACT_COVERAGE, lean,
                'no extractable facts to check; similarity band=%s. The fact axis is '
                'silent, not satisfied.' % sim_band)

    if not similarity_available:
        # One signal only. Never a PASS - that would present a half-check as a full one.
        return (REVIEW, 'pass' if fact_verdict == cf.SUPPORTED else 'fail',
                'similarity unavailable (torch not installed); fact check alone says %s'
                % fact_verdict)

    if fact_verdict == cf.SUPPORTED:
        if sim_band == 'high':
            return PASS, None, 'facts supported and similarity in the high band'
        if sim_band == 'low':
            return (REVIEW, 'fail',
                    'signals disagree: facts supported but similarity in the low band - '
                    'possible correct-facts-wrong-substance drift, or a reworded '
                    'paraphrase the measure cannot see')
        return REVIEW, 'pass', 'facts supported, similarity mid-band'

    # UNSUPPORTED / PARTIAL
    if sim_band == 'low':
        return (REVIEW, 'fail',
                'facts %s and similarity in the low band - likely ungrounded, flagged '
                'rather than rejected because both underlying checks treat this as '
                'uncertain' % fact_verdict)
    if sim_band == 'high':
        return (REVIEW, 'pass',
                'facts %s but similarity high - often a correct paraphrase the fact '
                'matcher could not align' % fact_verdict)
    return REVIEW, None, 'facts %s and similarity mid-band' % fact_verdict


# --------------------------------------------------------------- per-response check

def verify_response(response, source_chunk_id, ctx):
    """Verify ONE response against ONE chunk. The DPO-reusable unit.

    Deliberately knows nothing about SFT records: give it a string and a chunk id.
    """
    facts = cf.check_response(response, source_chunk_id, ctx.facts, ctx.corpus)
    sim = ctx.similarity(response, source_chunk_id)
    pct = sim['percentile'] if sim else None
    band = ctx.bands.band(pct)
    contradicted_types = [a['type'] for a in facts.get('assertions', [])
                          if a.get('verdict') == cf.CONTRADICTED]
    verdict, leaning, reason = combine(facts['verdict'], band,
                                       ctx.similarity_available and sim is not None,
                                       contradicted_types)
    return {
        'source_chunk_id': source_chunk_id,
        'verdict': verdict,
        'leaning': leaning,
        'reason': reason,
        'fact_verdict': facts['verdict'],
        'fact_assertions': facts.get('assertions', []),
        'similarity': sim,
        'similarity_band': band,
        'similarity_available': bool(ctx.similarity_available and sim is not None),
        'bands_used': ctx.bands.as_dict(),
    }


def verify_sft_record(record, ctx):
    """Validate the SFT schema, then delegate to verify_response."""
    missing = [f for f in SFT_FIELDS if f not in record]
    if missing:
        return {'verdict': INVALID_RECORD, 'leaning': None,
                'reason': 'missing required field(s): %s' % ', '.join(missing),
                'source_chunk_id': record.get('source_chunk_id')}
    out = verify_response(record['response'], record['source_chunk_id'], ctx)
    out['model_version'] = record.get('model_version')
    out['format_type'] = record.get('format_type')
    out['format_validated'] = False        # explicit: no format check exists yet

    # Cheap record-level consistency check: the record's declared region must match the
    # chunk it cites. A mismatch is a record construction error, not a model error, and
    # would otherwise be invisible.
    chunk = ctx.chunks.get(record['source_chunk_id'])
    if chunk is not None and record.get('source_region') != chunk.get('region'):
        out['record_warnings'] = [
            'source_region %r does not match the chunk region %r'
            % (record.get('source_region'), chunk.get('region'))]
    return out


# Verdict ordering for DPO preference. Facts dominate; similarity only breaks ties
# between equal verdicts. Kept as a pure function so the ordering is testable without a
# model, exactly as combine() is.
VERDICT_RANK = {PASS: 0, NO_FACT_COVERAGE: 1, REVIEW: 2, FAIL_CONTRADICTED: 3,
                INVALID_RECORD: 4}


def rank_key(result):
    pct = (result.get('similarity') or {}).get('percentile')
    return (VERDICT_RANK.get(result.get('verdict'), 9),
            -(pct if pct is not None else -1))


def compare_responses(response_a, response_b, source_chunk_id, ctx):
    """DPO helper: score two candidates for the same chunk and rank them.

    Uses the identical per-response function and the identical baseline, so the two are
    genuinely comparable. Ordering is by verdict severity first, then by similarity
    percentile - facts dominate, similarity breaks ties.
    """
    a = verify_response(response_a, source_chunk_id, ctx)
    b = verify_response(response_b, source_chunk_id, ctx)
    ka, kb = rank_key(a), rank_key(b)
    if ka < kb:
        preferred, why = 'a', 'candidate a has the stronger verdict/similarity'
    elif kb < ka:
        preferred, why = 'b', 'candidate b has the stronger verdict/similarity'
    else:
        preferred, why = None, 'candidates are indistinguishable on both signals'
    return {'preferred': preferred, 'reason': why, 'a': a, 'b': b}


# ------------------------------------------------------------------------- self-test
#
# The decision rule is tested directly, with no model and no corpus, so the full matrix
# is exercised cheaply and deterministically. Arabic is not needed here at all - the
# inputs are verdict labels.

MATRIX = [
    # (fact_verdict, sim_band, sim_available, expected verdict, expected leaning,
    #  contradicted_types)
    (cf.CONTRADICTED,        'high',        True,  FAIL_CONTRADICTED, 'fail'),
    (cf.CONTRADICTED,        'low',         True,  FAIL_CONTRADICTED, 'fail'),
    (cf.CONTRADICTED,        'middle',      True,  FAIL_CONTRADICTED, 'fail'),
    (cf.CONTRADICTED,        'unavailable', False, FAIL_CONTRADICTED, 'fail'),
    (cf.SUPPORTED,           'high',        True,  PASS,              None),
    (cf.SUPPORTED,           'middle',      True,  REVIEW,            'pass'),
    (cf.SUPPORTED,           'low',         True,  REVIEW,            'fail'),
    (cf.UNSUPPORTED,         'low',         True,  REVIEW,            'fail'),
    (cf.UNSUPPORTED,         'high',        True,  REVIEW,            'pass'),
    (cf.UNSUPPORTED,         'middle',      True,  REVIEW,            None),
    (cf.PARTIAL,             'low',         True,  REVIEW,            'fail'),
    (cf.PARTIAL,             'high',        True,  REVIEW,            'pass'),
    ('NO_CHECKABLE_CLAIMS',  'high',        True,  NO_FACT_COVERAGE,  'pass'),
    ('NO_CHECKABLE_CLAIMS',  'low',         True,  NO_FACT_COVERAGE,  'fail'),
    ('NO_CHECKABLE_CLAIMS',  'middle',      True,  NO_FACT_COVERAGE,  None),
    ('NO_CHECKABLE_CLAIMS',  'unavailable', False, NO_FACT_COVERAGE,  None),
    (cf.SUPPORTED,           'unavailable', False, REVIEW,            'pass'),
    (cf.UNSUPPORTED,         'unavailable', False, REVIEW,            'fail'),
    ('UNKNOWN_CHUNK',        'high',        True,  INVALID_RECORD,    None),
]

# Contradiction severity depends on the fact type that contradicted.
CONTRADICTION_TYPE_MATRIX = [
    # (contradicted_types, expected verdict)
    (['hijri_year'],                  FAIL_CONTRADICTED),
    (['page_reference'],              FAIL_CONTRADICTED),
    (['citation_authority'],          FAIL_CONTRADICTED),
    (['entry_headword'],              FAIL_CONTRADICTED),
    (['region_claim'],                REVIEW),            # heuristic alone -> review
    (['region_claim', 'hijri_year'],  FAIL_CONTRADICTED),  # any precise type -> fail
    ([],                              FAIL_CONTRADICTED),  # unknown -> fail closed
]


def run_self_test():
    ok = True
    for fv, band, avail, want_v, want_lean in MATRIX:
        got_v, got_lean, _ = combine(fv, band, avail)
        if got_v != want_v or got_lean != want_lean:
            print('  [FAIL] %-20s + %-11s avail=%-5s -> %s/%s, expected %s/%s'
                  % (fv, band, avail, got_v, got_lean, want_v, want_lean))
            ok = False

    # A precise contradicted fact must fail at EVERY band - the rule that matters most.
    for band in ('high', 'middle', 'low', 'unavailable'):
        v = combine(cf.CONTRADICTED, band, band != 'unavailable', ['hijri_year'])[0]
        if v != FAIL_CONTRADICTED:
            print('  [FAIL] precise CONTRADICTED escaped failure at band=%s' % band)
            ok = False

    # Contradiction severity by fact type.
    for types, want in CONTRADICTION_TYPE_MATRIX:
        got = combine(cf.CONTRADICTED, 'high', True, types)[0]
        if got != want:
            print('  [FAIL] contradiction types %s -> %s, expected %s'
                  % (types, got, want))
            ok = False

    # A region-only contradiction must never be a hard fail, at any band.
    for band in ('high', 'middle', 'low'):
        if combine(cf.CONTRADICTED, band, True, ['region_claim'])[0] == FAIL_CONTRADICTED:
            print('  [FAIL] region-only contradiction hard-failed at band=%s' % band)
            ok = False

    # No-facts must never be reported as PASS or as a contradiction failure.
    for band in ('high', 'middle', 'low'):
        v = combine('NO_CHECKABLE_CLAIMS', band, True)[0]
        if v in (PASS, FAIL_CONTRADICTED):
            print('  [FAIL] NO_CHECKABLE_CLAIMS collapsed into %s at band=%s' % (v, band))
            ok = False

    # Without similarity, nothing may be issued as a PASS.
    for fv in (cf.SUPPORTED, cf.PARTIAL, cf.UNSUPPORTED):
        if combine(fv, 'unavailable', False)[0] == PASS:
            print('  [FAIL] single-signal result issued as PASS for %s' % fv)
            ok = False

    # Bands
    b = SimilarityBands(high=90, low=50)
    for pct, want in ((100, 'high'), (90, 'high'), (89.9, 'middle'), (50, 'middle'),
                      (49.9, 'low'), (0, 'low'), (None, 'unavailable')):
        if b.band(pct) != want:
            print('  [FAIL] band(%s) = %s, expected %s' % (pct, b.band(pct), want))
            ok = False
    try:
        SimilarityBands(high=10, low=90)
        print('  [FAIL] inverted bands were accepted')
        ok = False
    except ValueError:
        pass

    # DPO preference ordering
    def R(v, pct=None):
        return {'verdict': v, 'similarity': ({'percentile': pct} if pct is not None else None)}
    order_cases = [
        (R(PASS, 98), R(FAIL_CONTRADICTED, 98), 'a', 'facts dominate at equal similarity'),
        (R(PASS, 50), R(REVIEW, 100), 'a', 'a better verdict beats a better score'),
        (R(NO_FACT_COVERAGE, 100), R(NO_FACT_COVERAGE, 32), 'a',
         'similarity breaks a tie between equal verdicts'),
        (R(REVIEW, 70), R(REVIEW, 70), None, 'identical results must tie'),
        (R(FAIL_CONTRADICTED, 99), R(INVALID_RECORD, 99), 'a',
         'a bad response still ranks above an unusable record'),
    ]
    for ra, rb, want, note in order_cases:
        ka, kb = rank_key(ra), rank_key(rb)
        got = 'a' if ka < kb else ('b' if kb < ka else None)
        if got != want:
            print('  [FAIL] ordering (%s): got %r expected %r' % (note, got, want))
            ok = False

    # SFT schema validation
    bad = verify_sft_record({'response': 'x'}, _StubCtx())
    if bad['verdict'] != INVALID_RECORD or 'instruction' not in bad['reason']:
        print('  [FAIL] missing SFT fields not reported: %r' % bad)
        ok = False

    print('passed: %s' % ok)
    return ok


class _StubCtx(object):
    """Minimal context for schema-validation tests; never reaches the checks."""
    corpus = 'saudi_dialect'
    facts = {}
    chunks = {}
    bands = SimilarityBands()
    index = None
    similarity_available = False

    def similarity(self, response, chunk_id):
        return None


# ------------------------------------------------------------------------------ main

def _fmt_sim(sim):
    if not sim:
        return 'unavailable'
    return ('pct=%5.1f%% margin=%+.3f max=%.3f' %
            (sim['percentile'], sim['margin'], sim['max']))


def main():
    ap = argparse.ArgumentParser(description='Combined SFT verifier (Task 3).')
    ap.add_argument('--corpus', choices=sorted(CHUNKS), help='required')
    ap.add_argument('--in', dest='inp', help='SFT records .jsonl')
    ap.add_argument('--no-similarity', action='store_true',
                    help='skip the embedding check (fact signal only)')
    ap.add_argument('--model', default=csim.DEFAULT_MODEL)
    ap.add_argument('--baseline', type=int, default=120)
    ap.add_argument('--sim-high', type=float, default=95.0,
                    help='percentile at or above which similarity counts as high')
    ap.add_argument('--sim-low', type=float, default=50.0,
                    help='percentile below which similarity counts as low')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)
    if not args.corpus:
        sys.stderr.write('[verify] REFUSING to run: --corpus is required.\n')
        sys.exit(1)
    if not args.inp:
        sys.stderr.write('[verify] --in <records.jsonl> is required\n')
        sys.exit(1)

    embedder = None
    if not args.no_similarity:
        try:
            embedder = csim.SentenceTransformerEmbedder(args.model)
        except ImportError:
            sys.stderr.write(
                '[verify] sentence-transformers not installed; running fact-only.\n'
                '         No PASS will be issued. Install:\n'
                '           python -m pip install -r requirements-verification.txt\n')

    ctx = VerificationContext(args.corpus, embedder=embedder, baseline_n=args.baseline,
                              bands=SimilarityBands(args.sim_high, args.sim_low))
    records = [json.loads(l) for l in open(args.inp, encoding='utf-8') if l.strip()]

    print('[verify] corpus=%s  similarity=%s  bands: high>=%.0f low<%.0f  baseline=%d'
          % (args.corpus, 'on' if ctx.similarity_available else 'OFF',
             ctx.bands.high, ctx.bands.low, len(ctx.baseline_ids)))
    print()

    results = []
    for rec in records:
        r = verify_sft_record(rec, ctx)
        results.append(r)
        if args.json:
            continue
        print('-' * 78)
        print('[%s%s] %s' % (r['verdict'],
                             '/' + r['leaning'] if r.get('leaning') else '',
                             rec.get('label', '')))
        print('  facts      : %s' % r.get('fact_verdict'))
        print('  similarity : %s  band=%s' % (_fmt_sim(r.get('similarity')),
                                              r.get('similarity_band')))
        print('  reason     : %s' % r['reason'])
        for w in r.get('record_warnings', []):
            print('  WARNING    : %s' % w)

    if args.json:
        for rec, r in zip(records, results):
            print(json.dumps({'label': rec.get('label'), **r}, ensure_ascii=False))
    else:
        print('-' * 78)
        counts = {}
        for r in results:
            counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
        print('%d record(s): %s' % (len(results), '  '.join(
            '%s=%d' % kv for kv in sorted(counts.items()))))
        print('NOTE: format_type is carried through but NOT validated - no format '
              'checker exists yet.')


if __name__ == '__main__':
    main()
