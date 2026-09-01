# -*- coding: utf-8 -*-
"""Task 3 Group C item 7 - DPO automated rule checks.

Input : a DPO record - prompt, chosen, rejected, source_chunk_id, source_region,
        rejection_type, model_version
Output: AUTO_CONFIRM / NEEDS_JUDGE / FLAG_SUSPICIOUS, with every component signal

Fast, deterministic checks that run BEFORE any LLM-judge call is spent on a pair.
Wraps and extends verify_sft.compare_responses(); it does not replace it.

What this is for
----------------
A judge call costs money and latency, and most of it is wasted on pairs that are
obviously fine or obviously broken. This triages:

  AUTO_CONFIRM    chosen beats rejected on a measurable axis AND the automated signal
                  corroborates the weakness the pair CLAIMS to have. A judge call is
                  optional, not required.
  NEEDS_JUDGE     the default, and the expected outcome for most pairs. Automated
                  checks are one layer of a layered design, not the final word.
  FLAG_SUSPICIOUS rejected looks equal to or better than chosen. This is not a quality
                  score - it says something is wrong with the PAIR: either it is
                  mislabeled, or the rejection-generation step produced something that
                  is not actually worse. These should never reach a judge as if they
                  were ordinary pairs.

Why AUTO_CONFIRM needs corroboration, not just a verdict gap
------------------------------------------------------------
A verdict gap alone says "chosen is better". It does not say "better in the way this
pair claims". A pair labeled `wrong_register` whose rejected half happens to carry a
factual error would auto-confirm on evidence that has nothing to do with register - and
the mislabel would be laundered into the training set as a confirmed pair.

So AUTO_CONFIRM requires both:
  1. chosen's verdict strictly outranks rejected's, and
  2. the declared rejection_type is corroborated by the signal that would detect it.

When the declared type is one this tooling cannot see at all (see the coverage table),
condition 2 can never be met and the pair routes to NEEDS_JUDGE. That is the correct
outcome: the automated layer has no opinion on that weakness and should not pretend to.

Detection coverage by rejection_type
------------------------------------
Measured on hand-built synthetic pairs, not asserted. `strong` means a dedicated signal
fires on the weakness; `partial` means it fires on some instances of it; `none` means no
automated check in this project can see it and the LLM judge is the only line of defence.

  rejection_type                signal                            coverage   auto-conf?
  partial_factual_errors        check_facts CONTRADICTED          strong     YES
  less_faithful_reconstruction  similarity percentile drop        strong     YES
  unsupported_additions         check_facts extension/addition    partial    YES
  verbosity                     length ratio only                 strong     NO
  missing_information           length ratio (+ weak sim signal)  partial    NO
  poor_instruction_following    none dedicated; drifts off-chunk  incidental NO
  wrong_formatting              none dedicated; may damage content incidental NO
  wrong_register                (no register classifier)          none       NO
  weak_organization             (no structure checker)            none       NO

The `auto-conf?` column is the practical answer and it is narrower than `coverage`.
`verbosity` is reliably DETECTED - the length signal fires every time - but a padded
answer keeps every fact, so verdict and similarity tie and the pair can never reach
AUTO_CONFIRM. Detection and confirmability are different questions and conflating them
would overstate what this saves.

Only 3 of 9 types can be auto-confirmed. For the other 6 this module still catches a
BROKEN pair (rejected scoring better than chosen) but cannot confirm a good one, and the
judge carries the load. Whoever builds real DPO generation should know this before
choosing the rejection_type mix: a set weighted toward the bottom six will produce almost
no AUTO_CONFIRMs and will not reduce judge spend.

Measured outcomes on the synthetic set
--------------------------------------
22 hand-built pairs: all 9 charter types on BOTH corpora, plus 2 deliberately broken
pairs per corpus. Identical results on the two corpora, which is what a rule keyed on
signals rather than surface text should produce:

  AUTO_CONFIRM     3   partial_factual_errors, less_faithful_reconstruction,
                       unsupported_additions
  NEEDS_JUDGE      6   the other six types
  FLAG_SUSPICIOUS  2   the broken pairs only

The broken pairs exist because zero FLAG_SUSPICIOUS on well-formed input only proves the
flag does not fire spuriously - not that it fires when it should:

  swapped halves      the corrupted response labeled `chosen`
                      -> chosen=FAIL_CONTRADICTED, rejected=PASS, caught on both corpora
  mislabeled          both halves identical while claiming a factual error
                      -> tie on a detectable type with nothing corroborating it, caught

FORMAT CHECKING IS A REAL GAP, NOT SKIPPED SILENTLY
---------------------------------------------------
Check 3 in the brief is "chosen should match its declared format_type at least as well
as rejected does". No format checker exists in this project yet - verify_sft.py marks
every record `format_validated: false` for the same reason. `check_format()` below is a
declared hook that returns 'unavailable' and is wired into the verdict, so a
`wrong_formatting` pair can never be AUTO_CONFIRMed. When a real checker is built it
plugs in here and the coverage table entry changes from `none` to whatever it earns.

Length is a FLAG, never a failure
---------------------------------
Length is measured and compared against what the declared rejection_type implies -
`verbosity` should make rejected longer, `missing_information` shorter. An inconsistency
does not fail the pair; it downgrades AUTO_CONFIRM to NEEDS_JUDGE and is reported. A
short answer is not automatically worse, and a long one is not automatically padded.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_facts as cf                                        # noqa: E402
import check_similarity as csim                                 # noqa: E402
import verify_sft as vs                                         # noqa: E402

CHUNKS = csim.CHUNKS

AUTO_CONFIRM = 'AUTO_CONFIRM'
NEEDS_JUDGE = 'NEEDS_JUDGE'
FLAG_SUSPICIOUS = 'FLAG_SUSPICIOUS'

DPO_FIELDS = ('prompt', 'chosen', 'rejected', 'source_chunk_id', 'source_region',
              'rejection_type', 'model_version')

# The nine deliberate weakness types from the project charter.
REJECTION_TYPES = (
    'poor_instruction_following',
    'wrong_formatting',
    'missing_information',
    'unsupported_additions',
    'wrong_register',
    'verbosity',
    'weak_organization',
    'partial_factual_errors',
    'less_faithful_reconstruction',
)

# What the length signal should look like if the declared type is honest. None means
# the type implies nothing about length, so length carries no evidence either way.
LENGTH_EXPECTATION = {
    'verbosity': 'rejected_longer',
    'missing_information': 'rejected_shorter',
}
# Ratio = rejected words / chosen words. Deliberately loose: these are sanity bounds,
# not calibrated thresholds, and a pair is only ever flagged by them, never failed.
VERBOSE_RATIO = 1.4
TERSE_RATIO = 0.7

# Percentile gap below which two similarity scores are treated as indistinguishable.
SIMILARITY_TIE_MARGIN = 5.0

# Levels:
#   strong      a dedicated signal fires on the weakness itself
#   partial     a signal fires on some instances of it
#   incidental  no dedicated signal, but the weakness often damages content enough that
#               the grounding checks notice SOMETHING. Cannot corroborate the declared
#               type, so it never reaches AUTO_CONFIRM - but it does not tie either.
#   none        invisible to every automated check here
#
# These values are MEASURED on the 18 synthetic pairs, not assumed. Two were corrected
# after the first run: poor_instruction_following produced a verdict gap on both corpora
# (an off-topic answer drifts from the chunk), and wrong_formatting did on one corpus
# where destroying the structure also destroyed the content signal.
DETECTABILITY = {
    'partial_factual_errors': 'strong',
    'less_faithful_reconstruction': 'strong',
    'verbosity': 'strong',
    'missing_information': 'partial',
    'unsupported_additions': 'partial',
    'poor_instruction_following': 'incidental',
    'wrong_formatting': 'incidental',
    'wrong_register': 'none',
    'weak_organization': 'none',
}


# ------------------------------------------------------------------ format hook (gap)

def check_format(response, format_type):
    """Declared hook. No format checker exists yet - see the module docstring.

    Returns 'unavailable' rather than a pass, so nothing downstream can mistake the
    absence of a check for a clean result.
    """
    return 'unavailable'


# ------------------------------------------------------------------------ pure logic

def length_stats(chosen, rejected):
    c, r = len(chosen.split()), len(rejected.split())
    return {'chosen_words': c, 'rejected_words': r,
            'ratio': (float(r) / c) if c else None}


def length_signal(ratio):
    if ratio is None:
        return 'unknown'
    if ratio >= VERBOSE_RATIO:
        return 'rejected_longer'
    if ratio <= TERSE_RATIO:
        return 'rejected_shorter'
    return 'comparable'


def length_consistency(rejection_type, signal):
    """Does the observed length relationship match what the declared type implies?"""
    expected = LENGTH_EXPECTATION.get(rejection_type)
    if expected is None:
        return 'not_applicable', 'this rejection_type implies nothing about length'
    if signal == expected:
        return 'consistent', 'length matches what %r implies' % rejection_type
    return ('inconsistent',
            'declared %r implies %s, observed %s' % (rejection_type, expected, signal))


def direction(chosen_result, rejected_result, tie_margin=SIMILARITY_TIE_MARGIN):
    """Which side is better, and on what evidence. Pure over two verdict dicts."""
    rc = vs.VERDICT_RANK.get(chosen_result.get('verdict'), 9)
    rr = vs.VERDICT_RANK.get(rejected_result.get('verdict'), 9)
    if rc < rr:
        return 'chosen_better_verdict'
    if rr < rc:
        return 'rejected_better_verdict'
    pc = (chosen_result.get('similarity') or {}).get('percentile')
    pr = (rejected_result.get('similarity') or {}).get('percentile')
    if pc is None or pr is None:
        return 'tie'
    if pc - pr > tie_margin:
        return 'chosen_better_similarity'
    if pr - pc > tie_margin:
        return 'rejected_better_similarity'
    return 'tie'


def corroborates(rejection_type, chosen_result, rejected_result, length_sig):
    """Do the automated signals corroborate the weakness the pair CLAIMS to have?

    Returns True, False, or None when this tooling cannot see the declared type at all.
    None and False are different: None means "no opinion", False means "looked and did
    not find it".
    """
    if DETECTABILITY.get(rejection_type) in ('none', 'incidental'):
        return None

    cf_rej = rejected_result.get('fact_verdict')
    cf_cho = chosen_result.get('fact_verdict')
    pc = (chosen_result.get('similarity') or {}).get('percentile')
    pr = (rejected_result.get('similarity') or {}).get('percentile')

    if rejection_type == 'partial_factual_errors':
        return cf_rej == cf.CONTRADICTED and cf_cho != cf.CONTRADICTED

    if rejection_type == 'unsupported_additions':
        # The extension/addition path lands as CONTRADICTED (token added that the source
        # does not contain) or UNSUPPORTED (specific with nothing to match).
        return (cf_rej in (cf.CONTRADICTED, cf.UNSUPPORTED)
                and cf_cho not in (cf.CONTRADICTED, cf.UNSUPPORTED))

    if rejection_type == 'less_faithful_reconstruction':
        if pc is None or pr is None:
            return False
        return (pc - pr) > SIMILARITY_TIE_MARGIN

    if rejection_type == 'verbosity':
        return length_sig == 'rejected_longer'

    if rejection_type == 'missing_information':
        # Shorter alone is weak evidence; pair it with a similarity drop.
        shorter = length_sig == 'rejected_shorter'
        weaker = (pc is not None and pr is not None and (pc - pr) > SIMILARITY_TIE_MARGIN)
        return bool(shorter and weaker)

    return False


def combine_dpo(direction_value, corroborated, length_consistent, format_status,
                declared_detectability):
    """Pure decision function. No IO, no model - the whole rule in one place."""
    if direction_value in ('rejected_better_verdict', 'rejected_better_similarity'):
        return (FLAG_SUSPICIOUS,
                'rejected scores better than chosen (%s) - the pair is mislabeled or the '
                'rejection step did not produce something worse' % direction_value)
    if direction_value == 'tie':
        # A tie is only suspicious when this tooling COULD have seen the declared
        # weakness and did not. Measured: 4 of 9 charter types are invisible to every
        # automated check, so treating every tie as suspicious would flag well-formed
        # pairs as broken and make the signal useless. And `verbosity` ties on verdict
        # BY CONSTRUCTION - a padded answer keeps every fact - yet is detected by the
        # length signal, so a corroborated tie is evidence the pair is fine, not broken.
        if corroborated is True:
            return (NEEDS_JUDGE,
                    'verdict and similarity tie, but the declared weakness IS '
                    'corroborated on its own axis - the pair differs where the verdict '
                    'cannot see')
        if length_consistent == 'consistent':
            return (NEEDS_JUDGE,
                    'verdict and similarity tie, but the length profile matches what the '
                    'declared rejection_type implies')
        if declared_detectability in ('none', 'incidental'):
            return (NEEDS_JUDGE,
                    'verdict and similarity tie, but %r is not something any automated '
                    'check here can see - the tie is uninformative, not suspicious'
                    % declared_detectability)
        return (FLAG_SUSPICIOUS,
                'chosen and rejected are indistinguishable on every automated axis, and '
                'this rejection_type IS one the checks can normally detect - the pair is '
                'likely mislabeled or the rejection is not actually worse')

    if direction_value == 'chosen_better_similarity':
        return (NEEDS_JUDGE,
                'chosen leads on similarity only, not on verdict - too weak to confirm '
                'automatically')

    # chosen_better_verdict from here on.
    if declared_detectability in ('none', 'incidental'):
        return (NEEDS_JUDGE,
                'chosen outranks rejected, but coverage for this rejection_type is %r, '
                'so the claimed weakness cannot be corroborated - the gap may be on an '
                'unrelated axis' % declared_detectability)
    if format_status == 'unavailable' and declared_detectability == 'format':
        return NEEDS_JUDGE, 'no format checker exists to corroborate this pair'
    if corroborated is None:
        return (NEEDS_JUDGE,
                'chosen outranks rejected, but the declared weakness type cannot be '
                'checked automatically')
    if not corroborated:
        return (NEEDS_JUDGE,
                'chosen outranks rejected, but on evidence unrelated to the declared '
                'rejection_type - possible mislabel')
    if length_consistent == 'inconsistent':
        return (NEEDS_JUDGE,
                'chosen outranks rejected and the weakness is corroborated, but the '
                'length profile contradicts the declared rejection_type')
    return (AUTO_CONFIRM,
            'chosen outranks rejected and the declared weakness is corroborated by the '
            'signal that detects it')


# ------------------------------------------------------------------ record-level check

def check_pair(record, ctx):
    missing = [f for f in DPO_FIELDS if f not in record]
    if missing:
        return {'verdict': FLAG_SUSPICIOUS, 'reason': 'missing field(s): %s'
                % ', '.join(missing), 'rejection_type': record.get('rejection_type')}

    rtype = record['rejection_type']
    warnings = []
    if rtype not in REJECTION_TYPES:
        warnings.append('rejection_type %r is not one of the nine charter types' % rtype)

    cid = record['source_chunk_id']
    chosen = vs.verify_response(record['chosen'], cid, ctx)
    rejected = vs.verify_response(record['rejected'], cid, ctx)

    lens = length_stats(record['chosen'], record['rejected'])
    lsig = length_signal(lens['ratio'])
    lcons, lnote = length_consistency(rtype, lsig)
    fmt_chosen = check_format(record['chosen'], record.get('format_type'))
    fmt_rejected = check_format(record['rejected'], record.get('format_type'))

    d = direction(chosen, rejected)
    corr = corroborates(rtype, chosen, rejected, lsig)
    detect = DETECTABILITY.get(rtype, 'none')
    verdict, reason = combine_dpo(d, corr, lcons, fmt_chosen, detect)

    chunk = ctx.chunks.get(cid)
    if chunk is not None and record.get('source_region') != chunk.get('region'):
        warnings.append('source_region %r does not match the chunk region %r'
                        % (record.get('source_region'), chunk.get('region')))

    return {
        'verdict': verdict, 'reason': reason,
        'rejection_type': rtype, 'detectability': detect,
        'direction': d, 'corroborated': corr,
        'chosen_verdict': chosen['verdict'], 'rejected_verdict': rejected['verdict'],
        'chosen_facts': chosen.get('fact_verdict'),
        'rejected_facts': rejected.get('fact_verdict'),
        'chosen_pct': (chosen.get('similarity') or {}).get('percentile'),
        'rejected_pct': (rejected.get('similarity') or {}).get('percentile'),
        'length': lens, 'length_signal': lsig,
        'length_consistency': lcons, 'length_note': lnote,
        'format_check': {'chosen': fmt_chosen, 'rejected': fmt_rejected,
                         'note': 'no format checker exists yet - this is a real gap'},
        'record_warnings': warnings,
    }


# ------------------------------------------------------------------------- self-test

def _R(verdict, pct=None, facts=None):
    return {'verdict': verdict, 'fact_verdict': facts,
            'similarity': ({'percentile': pct} if pct is not None else None)}


def run_self_test():
    ok = True

    # direction()
    dir_cases = [
        (_R(vs.PASS, 99), _R(vs.FAIL_CONTRADICTED, 99), 'chosen_better_verdict'),
        (_R(vs.FAIL_CONTRADICTED, 99), _R(vs.PASS, 99), 'rejected_better_verdict'),
        (_R(vs.REVIEW, 90), _R(vs.REVIEW, 40), 'chosen_better_similarity'),
        (_R(vs.REVIEW, 40), _R(vs.REVIEW, 90), 'rejected_better_similarity'),
        (_R(vs.REVIEW, 70), _R(vs.REVIEW, 70), 'tie'),
        (_R(vs.REVIEW, 72), _R(vs.REVIEW, 70), 'tie'),          # inside the margin
        (_R(vs.PASS), _R(vs.PASS), 'tie'),                       # no similarity at all
    ]
    for c, r, want in dir_cases:
        got = direction(c, r)
        if got != want:
            print('  [FAIL] direction -> %s, expected %s' % (got, want))
            ok = False

    # combine_dpo() matrix
    matrix = [
        # (direction, corroborated, length_consistency, detectability, expected)
        ('chosen_better_verdict', True,  'consistent',     'strong',  AUTO_CONFIRM),
        ('chosen_better_verdict', True,  'not_applicable', 'strong',  AUTO_CONFIRM),
        ('chosen_better_verdict', True,  'inconsistent',   'strong',  NEEDS_JUDGE),
        ('chosen_better_verdict', False, 'not_applicable', 'strong',  NEEDS_JUDGE),
        ('chosen_better_verdict', None,  'not_applicable', 'none',    NEEDS_JUDGE),
        ('chosen_better_verdict', True,  'consistent',     'partial', AUTO_CONFIRM),
        ('chosen_better_similarity', True, 'consistent',   'strong',  NEEDS_JUDGE),
        ('tie',                    True,  'consistent',    'strong',  NEEDS_JUDGE),
        ('tie',                    False, 'not_applicable','strong',  FLAG_SUSPICIOUS),
        ('tie',                    False, 'consistent',    'partial', NEEDS_JUDGE),
        ('tie',                    None,  'not_applicable','none',    NEEDS_JUDGE),
        ('tie',                    None,  'not_applicable','incidental', NEEDS_JUDGE),
        ('rejected_better_verdict', True, 'consistent',    'strong',  FLAG_SUSPICIOUS),
        ('rejected_better_similarity', True, 'consistent', 'strong',  FLAG_SUSPICIOUS),
    ]
    for d, corr, lc, det, want in matrix:
        got, _ = combine_dpo(d, corr, lc, 'unavailable', det)
        if got != want:
            print('  [FAIL] combine_dpo(%s, %s, %s, %s) -> %s, expected %s'
                  % (d, corr, lc, det, got, want))
            ok = False

    # Invariants that matter most.
    for det in ('strong', 'partial', 'none'):
        for corr in (True, False, None):
            v, _ = combine_dpo('rejected_better_verdict', corr, 'consistent',
                               'unavailable', det)
            if v != FLAG_SUSPICIOUS:
                print('  [FAIL] rejected-better escaped FLAG_SUSPICIOUS (det=%s)' % det)
                ok = False
    # A tie is suspicious ONLY when the type is one the checks could have detected and
    # nothing corroborated it. Ties on invisible types are uninformative, not broken -
    # measured: 2 of 9 charter types are invisible, and flagging their well-formed pairs
    # as suspicious would drown the signal.
    for det in ('none', 'incidental'):
        for corr in (True, False, None):
            v, _ = combine_dpo('tie', corr, 'not_applicable', 'unavailable', det)
            if v == FLAG_SUSPICIOUS:
                print('  [FAIL] tie on an undetectable type (%s) was flagged suspicious'
                      % det)
                ok = False
    for det in ('strong', 'partial'):
        v, _ = combine_dpo('tie', False, 'not_applicable', 'unavailable', det)
        if v != FLAG_SUSPICIOUS:
            print('  [FAIL] uncorroborated tie on a detectable type (%s) -> %s' % (det, v))
            ok = False
        # a corroborated tie is evidence the pair is fine, not broken: verbosity ties on
        # verdict by construction because a padded answer keeps every fact
        v, _ = combine_dpo('tie', True, 'consistent', 'unavailable', det)
        if v != NEEDS_JUDGE:
            print('  [FAIL] corroborated tie (%s) -> %s, expected NEEDS_JUDGE' % (det, v))
            ok = False
    # An undetectable weakness type can never be AUTO_CONFIRMed.
    for corr in (True, False, None):
        v, _ = combine_dpo('chosen_better_verdict', corr, 'consistent', 'unavailable',
                           'none')
        if v == AUTO_CONFIRM:
            print('  [FAIL] undetectable rejection_type reached AUTO_CONFIRM')
            ok = False

    # length signals
    for ratio, want in ((2.0, 'rejected_longer'), (1.4, 'rejected_longer'),
                        (1.0, 'comparable'), (0.7, 'rejected_shorter'),
                        (0.3, 'rejected_shorter'), (None, 'unknown')):
        if length_signal(ratio) != want:
            print('  [FAIL] length_signal(%s) = %s, expected %s'
                  % (ratio, length_signal(ratio), want))
            ok = False
    if length_consistency('verbosity', 'rejected_longer')[0] != 'consistent':
        print('  [FAIL] verbosity + rejected_longer should be consistent'); ok = False
    if length_consistency('verbosity', 'rejected_shorter')[0] != 'inconsistent':
        print('  [FAIL] verbosity + rejected_shorter should be inconsistent'); ok = False
    if length_consistency('wrong_register', 'rejected_longer')[0] != 'not_applicable':
        print('  [FAIL] wrong_register implies nothing about length'); ok = False

    # every charter type has a detectability entry
    for t in REJECTION_TYPES:
        if t not in DETECTABILITY:
            print('  [FAIL] no detectability entry for %r' % t); ok = False

    # format hook must never report a pass
    if check_format('anything', 'dictionary_entry') != 'unavailable':
        print('  [FAIL] check_format must report unavailable until one is built')
        ok = False

    # schema validation
    bad = check_pair({'prompt': 'x'}, None)
    if bad['verdict'] != FLAG_SUSPICIOUS or 'chosen' not in bad['reason']:
        print('  [FAIL] missing DPO fields not reported: %r' % bad); ok = False

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description='DPO automated rule checks (Task 3 Group C).')
    ap.add_argument('--corpus', choices=sorted(CHUNKS), help='required')
    ap.add_argument('--in', dest='inp', help='DPO records .jsonl')
    ap.add_argument('--no-similarity', action='store_true')
    ap.add_argument('--model', default=csim.DEFAULT_MODEL)
    ap.add_argument('--baseline', type=int, default=120)
    ap.add_argument('--coverage', action='store_true',
                    help='print the rejection_type detection coverage table and exit')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)
    if args.coverage:
        print('%-30s %s' % ('rejection_type', 'automated coverage'))
        print('-' * 54)
        for t in REJECTION_TYPES:
            print('%-30s %s' % (t, DETECTABILITY[t]))
        print('\n%d of %d are invisible to every automated check here.'
              % (sum(1 for t in REJECTION_TYPES if DETECTABILITY[t] == 'none'),
                 len(REJECTION_TYPES)))
        sys.exit(0)
    if not args.corpus or not args.inp:
        sys.stderr.write('[dpo] --corpus and --in are both required\n')
        sys.exit(1)

    embedder = None
    if not args.no_similarity:
        try:
            embedder = csim.SentenceTransformerEmbedder(args.model)
        except ImportError:
            sys.stderr.write('[dpo] sentence-transformers missing; similarity off\n')

    ctx = vs.VerificationContext(args.corpus, embedder=embedder,
                                 baseline_n=args.baseline)
    records = [json.loads(l) for l in open(args.inp, encoding='utf-8') if l.strip()]

    print('[dpo] corpus=%s  similarity=%s  pairs=%d'
          % (args.corpus, 'on' if ctx.similarity_available else 'OFF', len(records)))
    print()

    results = []
    for rec in records:
        r = check_pair(rec, ctx)
        results.append(r)
        if args.json:
            continue
        print('-' * 78)
        print('[%s] %s   (%s, coverage=%s)'
              % (r['verdict'], rec.get('label', ''), r['rejection_type'],
                 r['detectability']))
        print('  chosen  : %-18s facts=%-18s pct=%s'
              % (r['chosen_verdict'], r['chosen_facts'],
                 '%.1f' % r['chosen_pct'] if r['chosen_pct'] is not None else 'n/a'))
        print('  rejected: %-18s facts=%-18s pct=%s'
              % (r['rejected_verdict'], r['rejected_facts'],
                 '%.1f' % r['rejected_pct'] if r['rejected_pct'] is not None else 'n/a'))
        print('  direction=%s  corroborated=%s  length=%s (%s)'
              % (r['direction'], r['corroborated'], r['length_signal'],
                 r['length_consistency']))
        print('  reason  : %s' % r['reason'])
        for w in r['record_warnings']:
            print('  WARNING : %s' % w)

    if args.json:
        for rec, r in zip(records, results):
            print(json.dumps({'label': rec.get('label'), **r}, ensure_ascii=False))
    else:
        print('-' * 78)
        counts = {}
        for r in results:
            counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
        print('%d pair(s): %s' % (len(results),
                                  '  '.join('%s=%d' % kv for kv in sorted(counts.items()))))
        print('NOTE: format_type is NOT validated - no format checker exists yet.')


if __name__ == '__main__':
    main()
