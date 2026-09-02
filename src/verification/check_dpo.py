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
  wrong_formatting              check_format.py conformance       strong     YES
  unsupported_additions         check_facts extension/addition    partial    YES
  verbosity                     length ratio only                 strong     NO
  missing_information           length ratio (+ weak sim signal)  partial    NO
  poor_instruction_following    none dedicated; drifts off-chunk  incidental NO
  wrong_register                (no register classifier)          none       NO
  weak_organization             (no structure checker)            none       NO

The `auto-conf?` column is the practical answer and it is narrower than `coverage`.
`verbosity` is reliably DETECTED - the length signal fires every time - but a padded
answer keeps every fact, so verdict and similarity tie and the pair can never reach
AUTO_CONFIRM. Detection and confirmability are different questions and conflating them
would overstate what this saves.

4 of 9 types can be auto-confirmed, up from 3 once check_format.py replaced the stub.
For the other 5 this module still catches a BROKEN pair (rejected scoring better than
chosen) but cannot confirm a good one, and the judge carries the load. A rejection_type
mix weighted toward the bottom five will produce almost no AUTO_CONFIRMs and will not
reduce judge spend.

Format coverage is NOT uniform across format_types. MEASURED: both corpora are 100%
`dictionary_entry`, so that is the only value validated against real corpus content.
`prose`, `verse`, `list` and `footnote_block` are implemented against chunk.py's own
assignment rules and exercised only by synthetic cases - weaker evidence, and it should
not be reported as if it were equal.

Measured outcomes on the synthetic set
--------------------------------------
22 hand-built pairs: all 9 charter types on BOTH corpora, plus 2 deliberately broken
pairs per corpus. Identical results on the two corpora, which is what a rule keyed on
signals rather than surface text should produce:

  AUTO_CONFIRM     4   partial_factual_errors, less_faithful_reconstruction,
                       unsupported_additions, wrong_formatting
  NEEDS_JUDGE      5   the other five types
  FLAG_SUSPICIOUS  2   the broken pairs only

Plus 4 format probes per corpus, also identical across the two: 3 AUTO_CONFIRM and 1
FLAG_SUSPICIOUS - the last being the control, where both halves are equally well-formed
and the declared formatting weakness therefore cannot be corroborated.

The broken pairs exist because zero FLAG_SUSPICIOUS on well-formed input only proves the
flag does not fire spuriously - not that it fires when it should:

  swapped halves      the corrupted response labeled `chosen`
                      -> chosen=FAIL_CONTRADICTED, rejected=PASS, caught on both corpora
  mislabeled          both halves identical while claiming a factual error
                      -> tie on a detectable type with nothing corroborating it, caught

Format checking, and why it needed its own direction rule
--------------------------------------------------------
`check_format()` now delegates to check_format.py, which tests structural conformance
against the pipeline's own format vocabulary. It replaced a stub returning 'unavailable'.

A pure formatting failure ties on verdict BY CONSTRUCTION: the malformed half carries
identical facts and identical vocabulary, so fact and similarity checks see nothing.
Requiring a verdict gap would have capped wrong_formatting at NEEDS_JUDGE forever,
defeating the point of building the checker. Hence `axis_win`: a dedicated, deterministic
check showing chosen strictly better on the declared axis can carry a tie to
AUTO_CONFIRM. Format qualifies; the length signal deliberately does not, being a ratio
against a hand-picked bound rather than a binary structural fact. `axis_win` can never
override a rejected-better pair.

INDEPENDENCE, measured rather than assumed: on the classical corpus a probe whose halves
differ ONLY by entry markers scored PASS/SUPPORTED/pct 100.0 on BOTH sides - fact verdict
and similarity percentile identical - and only the format check separated them. On the
dialect corpus the same probe is partially COUPLED: check_facts detects lexical items
only when marked, so stripping the markers also degrades the fact axis. The independence
is real but corpus-dependent.

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
import check_format as cfmt                                     # noqa: E402

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
SIMILARITY_TIE_MARGIN = csim.PERCENTILE_TIE_MARGIN   # single source of truth

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
# ---------------------------------------------- chosen/rejected distinctness floor
#
# أرضية التمايز: يجب ألا يكون المرفوض نسخة شبه طبق الأصل من المختار.
#
# A DPO pair whose two halves are near-identical carries no contrastive signal - there is
# nothing for the model to learn a preference from. But the floor CANNOT be a single
# number, and the measurement says why.
#
# MEASURED, chosen-vs-rejected difflib token ratio across the 22 synthetic pairs:
#
#   1.000        the two BROKEN "halves identical" pairs        <- genuinely degenerate
#   0.974/0.967  LEGITIMATE partial_factual_errors (one digit)  <- minimal edit by design
#   0.884/0.857  LEGITIMATE unsupported_additions               <- minimal addition
#   0.685 ->     verbosity, missing_information, wrong_formatting, wrong_register,
#   0.033        weak_organization, less_faithful_reconstruction, poor_instruction_following
#
# A flat floor at 0.95 would flag the legitimate minimal-edit pairs, which are arguably
# the MOST valuable kind: they isolate one defect and hold everything else constant. So
# the floor is rejection_type aware. For the two minimal-edit types only an essentially
# exact duplicate counts as degenerate; for every other type, text that similar means the
# declared weakness was never actually introduced.
#
# Note what is NOT used: embedding cosine. `wrong_register` pairs say the SAME thing in a
# different register, so they are semantically near-identical BY DESIGN - a semantic floor
# would flag exactly the pairs whose contrast is register. Cosine is reported for
# information and never judged on.
MINIMAL_EDIT_TYPES = frozenset(['partial_factual_errors', 'unsupported_additions'])
DEGENERATE_IDENTICAL = 0.995   # any type: essentially the same text
DEGENERATE_FLOOR = 0.90        # non-minimal-edit types; highest observed legit is 0.685


def pair_distinctness(chosen, rejected, rejection_type, embedder=None):
    """Are the two halves distinct enough to carry a training signal?"""
    sim = csim.pair_similarity(chosen, rejected, embedder)
    lex = sim['lexical']
    floor = (DEGENERATE_IDENTICAL if rejection_type in MINIMAL_EDIT_TYPES
             else DEGENERATE_FLOOR)
    degenerate = lex >= floor
    if degenerate and lex >= DEGENERATE_IDENTICAL:
        why = ('the two halves are essentially the same text (lexical %.3f) - no '
               'contrastive signal at all' % lex)
    elif degenerate:
        why = ('lexical similarity %.3f >= %.2f for %r, which should produce '
               'substantively different text - the declared weakness may never have '
               'been introduced' % (lex, floor, rejection_type))
    else:
        why = 'halves are distinct enough (lexical %.3f < %.2f)' % (lex, floor)
    return {'lexical': lex, 'cosine': sim['cosine'], 'floor': floor,
            'degenerate': degenerate, 'reason': why}


DETECTABILITY = {
    'partial_factual_errors': 'strong',
    'less_faithful_reconstruction': 'strong',
    'verbosity': 'strong',
    'missing_information': 'partial',
    'unsupported_additions': 'partial',
    'poor_instruction_following': 'incidental',
    'wrong_formatting': 'strong',
    'wrong_register': 'none',
    'weak_organization': 'none',
}


# ------------------------------------------------------------------ format hook (gap)

def check_format(response, format_type):
    """Structural format conformance. Delegates to check_format.py.

    Was a stub returning 'unavailable', which made wrong_formatting the one charter
    rejection_type that could never be corroborated. Returns the status string; the full
    feature detail is available from check_format.check_format().
    """
    return cfmt.check_format(response, format_type)['status']


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


def similarity_comparable(chosen_result, rejected_result):
    """Delegates to verify_sft, the canonical implementation.

    يفوّض إلى verify_sft حيث التعريف الأصلي، حتى لا تُصلح القاعدة في موضع وتُترك
    معطلة في الآخر.
    """
    return vs.similarity_comparable(chosen_result, rejected_result)


def direction(chosen_result, rejected_result, tie_margin=SIMILARITY_TIE_MARGIN):
    """Which side is better, and on what evidence. Pure over two verdict dicts."""
    rc = vs.VERDICT_RANK.get(chosen_result.get('verdict'), 9)
    rr = vs.VERDICT_RANK.get(rejected_result.get('verdict'), 9)
    if rc < rr:
        return 'chosen_better_verdict'
    if rr < rc:
        return 'rejected_better_verdict'
    # المقارنة عبر الواجهة المحروسة في وحدة التشابه، التي ترفض الحكم بين نصين غير مسندين.
    # Compare through check_similarity's guarded API rather than by hand: it takes
    # groundedness as a required argument and refuses to rank two ungrounded responses,
    # so this call site cannot reintroduce the bug even if someone edits it later.
    rel = csim.compare_percentiles(
        (chosen_result.get('similarity') or {}).get('percentile'),
        (rejected_result.get('similarity') or {}).get('percentile'),
        vs.is_grounded(chosen_result), vs.is_grounded(rejected_result),
        tie_margin)
    if rel == csim.A_BETTER:
        return 'chosen_better_similarity'
    if rel == csim.B_BETTER:
        return 'rejected_better_similarity'
    return 'tie'          # covers TIE and INCOMPARABLE_UNGROUNDED alike


def corroborates(rejection_type, chosen_result, rejected_result, length_sig,
                 format_relation=None):
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

    if rejection_type == 'wrong_formatting':
        # Corroborated only when chosen conforms and rejected does not. Both-match or
        # both-mismatch means the declared weakness is not present as declared.
        return format_relation == 'chosen_better'

    if rejection_type == 'verbosity':
        return length_sig == 'rejected_longer'

    if rejection_type == 'missing_information':
        # Shorter alone is weak evidence; pair it with a similarity drop.
        shorter = length_sig == 'rejected_shorter'
        weaker = (pc is not None and pr is not None and (pc - pr) > SIMILARITY_TIE_MARGIN)
        return bool(shorter and weaker)

    return False


def combine_dpo(direction_value, corroborated, length_consistent, format_status,
                declared_detectability, axis_win=False, degenerate=False):
    """Pure decision function. No IO, no model - the whole rule in one place.

    `axis_win` means a DEDICATED, DETERMINISTIC check shows chosen strictly better on
    the declared axis. Only format conformance qualifies today. It matters because a
    pure formatting failure ties on verdict by construction - the malformed half carries
    the same facts and the same vocabulary, so fact and similarity checks see no
    difference - and without this a correctly detected wrong_formatting pair could never
    be confirmed.

    Length is deliberately NOT an axis_win. Format conformance is a binary structural
    fact with no threshold to tune; the length signal is a ratio against a hand-picked
    bound, and a padded answer is not wrong in the way a malformed one is.
    """
    if degenerate:
        return (FLAG_SUSPICIOUS,
                'DEGENERATE PAIR: chosen and rejected are near-duplicates, so there is '
                'no contrastive signal to train on regardless of which is better')
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
        if axis_win and corroborated is True and declared_detectability == 'strong':
            return (AUTO_CONFIRM,
                    'verdict and similarity tie - as they must for this weakness - but a '
                    'dedicated deterministic check shows chosen strictly better on the '
                    'declared axis')
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
        if axis_win and corroborated is True and declared_detectability == 'strong':
            return (AUTO_CONFIRM,
                    'a dedicated deterministic check shows chosen strictly better on the '
                    'declared axis, and similarity agrees')
        return (NEEDS_JUDGE,
                'chosen leads on similarity only, not on verdict - too weak to confirm '
                'automatically')

    # chosen_better_verdict from here on.
    if declared_detectability in ('none', 'incidental'):
        return (NEEDS_JUDGE,
                'chosen outranks rejected, but coverage for this rejection_type is %r, '
                'so the claimed weakness cannot be corroborated - the gap may be on an '
                'unrelated axis' % declared_detectability)
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
    fmt_chosen = fmt_rejected = None      # filled from the comparison below

    d = direction(chosen, rejected)
    fmt_relation, fmt_c_detail, fmt_r_detail = cfmt.compare_format(
        record['chosen'], record['rejected'], record.get('format_type'))
    corr = corroborates(rtype, chosen, rejected, lsig, fmt_relation)
    detect = DETECTABILITY.get(rtype, 'none')
    dist = pair_distinctness(record['chosen'], record['rejected'], rtype, ctx.embedder)
    axis_win = (rtype == 'wrong_formatting' and fmt_relation == 'chosen_better')
    verdict, reason = combine_dpo(d, corr, lcons, fmt_relation, detect, axis_win,
                                  dist['degenerate'])

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
        'distinctness': dist,
        'length': lens, 'length_signal': lsig,
        'length_consistency': lcons, 'length_note': lnote,
        'format_check': {'chosen': fmt_c_detail['status'],
                         'rejected': fmt_r_detail['status'],
                         'relation': fmt_relation,
                         'chosen_reason': fmt_c_detail.get('reason'),
                         'rejected_reason': fmt_r_detail.get('reason')},
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

    # distinctness floor: degeneracy outranks every other consideration
    deg_cases = [
        # (chosen, rejected, rejection_type, expected degenerate)
        ('نص واحد مكرر تماما بلا اي تغيير', 'نص واحد مكرر تماما بلا اي تغيير',
         'partial_factual_errors', True),        # identical -> degenerate for ANY type
        ('نص واحد مكرر تماما بلا اي تغيير', 'نص واحد مكرر تماما بلا اي تغيير',
         'wrong_register', True),
        # a minimal edit is LEGITIMATE for partial_factual_errors ...
        ('طبع المرجع سنة 1426 وفيه شرح مطول للمادة المعجمية المذكورة',
         'طبع المرجع سنة 1436 وفيه شرح مطول للمادة المعجمية المذكورة',
         'partial_factual_errors', False),
        # ... but the SAME pair under a type that should rewrite the text is degenerate
        ('طبع المرجع سنة 1426 وفيه شرح مطول للمادة المعجمية المذكورة',
         'طبع المرجع سنة 1436 وفيه شرح مطول للمادة المعجمية المذكورة',
         'wrong_register', True),
        # genuinely distinct halves pass under either
        ('اشرح معنى هذه المادة وبيّن أصل اللفظة',
         'تتناول هذه الفقرة موضوعا مختلفا تماما لا صلة له بما سبق إطلاقا',
         'wrong_register', False),
        ('اشرح معنى هذه المادة وبيّن أصل اللفظة',
         'تتناول هذه الفقرة موضوعا مختلفا تماما لا صلة له بما سبق إطلاقا',
         'partial_factual_errors', False),
    ]
    for ch, rj, rt, want in deg_cases:
        got = pair_distinctness(ch, rj, rt)['degenerate']
        if got != want:
            print('  [FAIL] distinctness(%r) -> degenerate=%s, expected %s (lex %.3f)'
                  % (rt, got, want, pair_distinctness(ch, rj, rt)['lexical']))
            ok = False
    # an identical pair must be degenerate under EVERY charter type
    for rt in REJECTION_TYPES:
        if not pair_distinctness('نص مطابق تماما', 'نص مطابق تماما', rt)['degenerate']:
            print('  [FAIL] identical halves not degenerate for %r' % rt); ok = False
    # degeneracy outranks every other signal, including a clean verdict gap
    v, why = combine_dpo('chosen_better_verdict', True, 'consistent', 'chosen_better',
                         'strong', True, degenerate=True)
    if v != FLAG_SUSPICIOUS or 'DEGENERATE' not in why:
        print('  [FAIL] a degenerate pair with a verdict gap was not flagged: %s' % v)
        ok = False

    # similarity between two ungrounded responses is noise, not a direction
    ung_a = {'verdict': vs.NO_FACT_COVERAGE, 'similarity': {'percentile': 20.0}}
    ung_b = {'verdict': vs.NO_FACT_COVERAGE, 'similarity': {'percentile': 90.0}}
    if direction(ung_a, ung_b) != 'tie':
        print('  [FAIL] ungrounded-vs-ungrounded similarity produced a direction')
        ok = False
    if direction(ung_b, ung_a) != 'tie':
        print('  [FAIL] ungrounded direction is not symmetric'); ok = False
    # but a verdict gap is still a direction even when one side is ungrounded
    gr = {'verdict': vs.PASS, 'similarity': {'percentile': 20.0}}
    if direction(gr, ung_b) != 'chosen_better_verdict':
        print('  [FAIL] verdict gap suppressed by the ungrounded guard'); ok = False
    if similarity_comparable(gr, ung_b) is not True:
        print('  [FAIL] one grounded side should make similarity comparable'); ok = False

    # axis_win: a deterministic check can carry a tie to AUTO_CONFIRM
    axis_cases = [
        ('tie', True, 'strong', True, AUTO_CONFIRM),
        ('tie', True, 'strong', False, NEEDS_JUDGE),      # no axis win -> not confirmed
        ('tie', False, 'strong', True, FLAG_SUSPICIOUS),  # axis win but not corroborated
        ('tie', True, 'partial', True, NEEDS_JUDGE),      # only `strong` may carry a tie
        ('chosen_better_similarity', True, 'strong', True, AUTO_CONFIRM),
    ]
    for d, corr, det, aw, want in axis_cases:
        got, _ = combine_dpo(d, corr, 'not_applicable', 'chosen_better', det, aw)
        if got != want:
            print('  [FAIL] axis_win(%s, corr=%s, det=%s, aw=%s) -> %s, expected %s'
                  % (d, corr, det, aw, got, want)); ok = False
    # axis_win must never override a rejected-better pair
    for d in ('rejected_better_verdict', 'rejected_better_similarity'):
        v, _ = combine_dpo(d, True, 'consistent', 'chosen_better', 'strong', True)
        if v != FLAG_SUSPICIOUS:
            print('  [FAIL] axis_win overrode %s' % d); ok = False

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

    # format checking is real now: it must discriminate, and must never pass an
    # unrecognised format_type
    if check_format('الكلمة (خب) وهي أرض مستوية.', 'dictionary_entry') != cfmt.MATCH:
        print('  [FAIL] a well-formed entry should match'); ok = False
    if check_format('خب أرض مستوية بلا بنية', 'dictionary_entry') != cfmt.MISMATCH:
        print('  [FAIL] undifferentiated text should mismatch'); ok = False
    if check_format('أي نص', 'table') == cfmt.MATCH:
        print('  [FAIL] an unrecognised format_type must never match'); ok = False

    # wrong_formatting corroborates only when chosen conforms and rejected does not
    fmt_cases = [('chosen_better', True), ('rejected_better', False),
                 ('both_match', False), ('both_mismatch', False),
                 ('undecidable', False)]
    for rel, want in fmt_cases:
        got = corroborates('wrong_formatting', _R(vs.PASS, 90), _R(vs.PASS, 90),
                           'comparable', rel)
        if got != want:
            print('  [FAIL] wrong_formatting corroboration on %s -> %s, expected %s'
                  % (rel, got, want)); ok = False

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
        print('  distinctness: lexical=%.3f floor=%.2f%s'
              % (r['distinctness']['lexical'], r['distinctness']['floor'],
                 '  DEGENERATE' if r['distinctness']['degenerate'] else ''))
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
        print('NOTE: format conformance IS checked (check_format.py). Only '
              'dictionary_entry is validated against real corpus data - both corpora '
              'are 100% that type; other format_types rest on synthetic cases.')


if __name__ == '__main__':
    main()
