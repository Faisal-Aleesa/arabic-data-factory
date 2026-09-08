# -*- coding: utf-8 -*-
"""Task 3, Group B - LLM judge for DPO pairs.

Confirms that `chosen` is genuinely better than `rejected` across the eight charter
dimensions. For two of the nine charter rejection types this judge is the ONLY
verification that exists anywhere in the project.

حَكَم أزواج التفضيل: التحقق الوحيد لنوعين من أنواع الضعف التسعة.

MEASURED, from check_dpo.py's coverage table:

    strong      wrong_formatting, verbosity, partial_factual_errors,
                less_faithful_reconstruction
    partial     missing_information, unsupported_additions
    incidental  poor_instruction_following
    none        wrong_register, weak_organization      <-- no automated check at all

`wrong_register` and `weak_organization` have no deterministic signal, and cannot get
one: check_dpo.py records that a `wrong_register` pair says the SAME thing in a different
register, so it is semantically near-identical by design - a similarity floor would flag
exactly the pairs whose whole contrast is register. `weak_organization` reorders content
without changing it. Both are invisible to fact checking, similarity, length and format.
So there is nothing to cross-validate this judge against on those two types. That is
stated here rather than discovered later.

THE KEY DECISION: the judge is never told the declared rejection_type
---------------------------------------------------------------------
`judge_dpo_pair()` takes the prompt, the two responses and the source. It does NOT take
`rejection_type`, and must not. Telling a judge "this pair is supposed to differ in
register, which is better?" is a leading question, and its agreement would corroborate
nothing - it would only show the judge can follow an instruction.

Instead the judge scores both responses on all eight dimensions blind, and THIS MODULE
maps the declared rejection_type onto the dimension that should have separated them
(`CORROBORATING_DIMENSION`). Corroboration is then a fact about the judge's independent
scoring, not about the hint it was given. It is the same discipline `check_dpo.py`
already applies when it corroborates a declared type against a deterministic signal, and
the same reason `judge_sft.py` never sees the rule-based verdict.

Position bias, and why the zero-coverage types cost two calls
-------------------------------------------------------------
LLM judges systematically favour whichever response is shown first. That bias is
indistinguishable from a genuine preference in a single call, which would be fatal for
the two types where nothing else can check the answer.

So position is assigned deterministically from `source_chunk_id` (stable across reruns,
not correlated with chosen/rejected), and for the types in `NO_AUTOMATED_COVERAGE` the
pair is judged TWICE with the positions swapped. If the two runs disagree, the verdict is
JUDGE_REVIEW with `position_biased: True` - the judge preferred a slot rather than a
response, and its answer carries no information. A pair that survives both orders is the
only kind of confirmation available for `wrong_register` and `weak_organization`.

This is the discrimination principle from `check_similarity.py` and `check_facts.py`
turned on the judge itself: a check that cannot fail on a bad input has not been tested.

Not calibrated
--------------
No real model has scored a pair through this module. The judging backend is a Qwen
instance whose endpoint is pending. The suite here exercises prompt shape, parsing,
position handling and verdict folding against a scripted backend. It does not and cannot
establish that a real model judges Arabic register well - and on the two zero-coverage
types there will be no automated way to find out, so those verdicts should be spot-checked
by a human reviewer before the type is trusted at scale.

STATUS: NOT THE SHIPPING DPO JUDGE. Kept deliberately - do not delete.
------------------------------------------------------------------------
As of the LLM_Judge integration (commit `3507c67`), production DPO judging runs through
`src/dpo/llm_judge.py` behind `judge_pipeline.adjudicate()`. This module is reachable
only via `judge_suite.py --dpo-path legacy`. That makes it unreferenced by the live path,
which is exactly the condition under which code quietly rots - so its remaining job is
written down here rather than left to be rediscovered.

**It is the reference implementation for the type-priming fix.** The merged judge is told
the record's `rejection_type` before being asked whether it agrees
(`judge_contract.JudgeInput.to_prompt_payload` -> `rejection_type_declared`). That is a
leading question: agreement means "not contradicted", not independent confirmation, and
escalation can therefore under-detect a genuine mislabel. See the KNOWN LIMITATION
section in `src/dpo/judge_contract.py`.

**What the follow-up needs to port from here: `CORROBORATING_DIMENSION`.** This module
never passes the declared type to the model. It asks for a blind score on all eight
charter dimensions, then maps rejection_type -> the dimension that should have separated
the pair and checks THAT. Corroboration becomes a fact about the judge's own independent
scoring instead of about the hint it was given. `judge_dpo_pair()` shows the whole
arrangement end to end, and `run_self_test()` pins it.

Already ported, so do NOT port it twice: the position-bias swap check now lives in
`llm_judge.judge_pair()`, deriving its type set from `check_dpo.DETECTABILITY` the same
way `NO_AUTOMATED_COVERAGE` does here.

هذه الوحدة ليست الحَكَم المستخدم في الإنتاج، وتُحفظ مرجعًا لإصلاح تحيّز النوع المُعلن.
"""

from __future__ import unicode_literals

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge_client as jc              # noqa: E402
import check_dpo as cd                 # noqa: E402
from judge_client import (             # noqa: E402
    JUDGE_PASS, JUDGE_FAIL, JUDGE_REVIEW, JUDGE_ABSTAIN,
    JudgeUnavailable, ABSTAIN_TRANSPORT, ABSTAIN_REFUSED,
)

PROMPT_VERSION = 'dpo-v1'

# The eight charter dimensions.
DIMENSIONS = ('grounding', 'factual_consistency', 'arabic_quality',
              'instruction_following', 'format', 'completeness', 'style', 'relevance')

WIN_A = 'A'
WIN_B = 'B'
WIN_TIE = 'tie'
WIN_VALUES = (WIN_A, WIN_B, WIN_TIE)

CONFIDENCE_VALUES = ('high', 'medium', 'low')
REQUIRED_KEYS = ('dimensions', 'overall', 'confidence')

# Which dimension SHOULD separate the pair if the declared rejection_type is honest.
# Single source of truth for the type vocabulary: check_dpo.REJECTION_TYPES.
CORROBORATING_DIMENSION = {
    'poor_instruction_following': 'instruction_following',
    'wrong_formatting': 'format',
    'missing_information': 'completeness',
    'unsupported_additions': 'grounding',
    'wrong_register': 'arabic_quality',
    'verbosity': 'style',
    'weak_organization': 'style',
    'partial_factual_errors': 'factual_consistency',
    'less_faithful_reconstruction': 'grounding',
}

# The types with no deterministic check anywhere. Judged twice, positions swapped.
NO_AUTOMATED_COVERAGE = frozenset(
    t for t, d in cd.DETECTABILITY.items() if d == 'none')


SYSTEM_PROMPT = """\
You are a meticulous evaluator of Arabic preference data built from two lexicographic
sources: a classical Arabic rhetoric dictionary and a Saudi regional dialect dictionary.

أنت مُقيّم دقيق لأزواج التفضيل العربية المبنية على معجمين: معجم بلاغة كلاسيكي ومعجم
لهجات سعودية.

You will be shown a SOURCE passage, a PROMPT, and two candidate answers, RESPONSE A and
RESPONSE B. Decide which is better on each of eight dimensions, then overall.

1. grounding            - claims supported by the SOURCE; no invented lemmas, poets or
                          dates, however plausible they sound
2. factual_consistency  - names, years and figures match the SOURCE exactly
3. arabic_quality       - grammatical, natural, and in an APPROPRIATE REGISTER. A
                          lexicographic answer written in casual or chatty Arabic is
                          worse than one in proper dictionary register even when the
                          content is identical. Dialect content should read as that
                          dialect; classical content as classical.
4. instruction_following- actually does what the PROMPT asked
5. format               - shape matches what the PROMPT implies
6. completeness         - covers what was asked; nothing important omitted or truncated
7. style                - tone, ORGANISATION and flow, and conciseness. A response whose
                          points are in a confusing order, that repeats itself, that
                          buries the answer, or that pads with filler is worse than one
                          that is ordered and direct - even when both contain exactly the
                          same information.
8. relevance            - on topic for the PROMPT and the SOURCE

Rules:
- Judge ONLY against the SOURCE shown.
- The two answers may be very similar. Differences of register or of ordering alone are
  REAL differences and should be scored, not called a tie.
- Use "tie" only when the two are genuinely equivalent on that dimension.
- Length is not quality. A longer answer is not better; a shorter one is not worse.
- Do not assume either position is the intended answer. A and B are in arbitrary order.

Answer with ONLY a JSON object, no commentary. Each dimension takes "A", "B" or "tie":

{"dimensions": {"grounding": "A|B|tie", "factual_consistency": "A|B|tie",
                "arabic_quality": "A|B|tie", "instruction_following": "A|B|tie",
                "format": "A|B|tie", "completeness": "A|B|tie",
                "style": "A|B|tie", "relevance": "A|B|tie"},
 "overall": "A|B|tie",
 "evidence": "<one short sentence naming the decisive difference>",
 "confidence": "high|medium|low"}
"""

USER_TEMPLATE = """\
SOURCE_REGION: {source_region}

SOURCE:
{chunk_text}

PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}
"""


def chosen_position(source_chunk_id, swap=False):
    """Which slot `chosen` occupies. Deterministic, and uncorrelated with the pair.

    Derived from the chunk id so a rerun places the same pair the same way - a randomised
    position would make two runs of the suite incomparable - while nothing about the id
    tracks which response is the chosen one.
    """
    h = hashlib.sha256((source_chunk_id or '').encode('utf-8')).digest()
    first = WIN_A if (h[0] % 2 == 0) else WIN_B
    if swap:
        return WIN_B if first == WIN_A else WIN_A
    return first


def build_prompt(pair, chunk_text, chosen_pos):
    """(system, user) for one pair, with `chosen` placed in slot `chosen_pos`."""
    chosen, rejected = pair.get('chosen', ''), pair.get('rejected', '')
    a, b = (chosen, rejected) if chosen_pos == WIN_A else (rejected, chosen)
    return SYSTEM_PROMPT, USER_TEMPLATE.format(
        source_region=pair.get('source_region', '(unspecified)'),
        chunk_text=chunk_text or '(source unavailable)',
        prompt=pair.get('prompt', ''),
        response_a=a, response_b=b)


def _to_chosen_terms(slot_winner, chosen_pos):
    """Translate an A/B answer into chosen/rejected/tie."""
    if slot_winner == WIN_TIE:
        return 'tie'
    return 'chosen' if slot_winner == chosen_pos else 'rejected'


def _validate_payload(obj):
    dims = obj.get('dimensions')
    if not isinstance(dims, dict):
        raise ValueError('"dimensions" is not an object')
    out = {}
    for d in DIMENSIONS:
        v = dims.get(d)
        if v not in WIN_VALUES:
            raise ValueError('dimension %r has value %r' % (d, v))
        out[d] = v
    overall = obj.get('overall')
    if overall not in WIN_VALUES:
        raise ValueError('overall %r' % overall)
    conf = obj.get('confidence')
    if conf not in CONFIDENCE_VALUES:
        raise ValueError('confidence %r' % conf)
    return out, overall, conf


def _abstention(reason, detail, usage=None):
    return {'judge_verdict': JUDGE_ABSTAIN, 'abstain_reason': reason, 'reason': detail,
            'dimensions': None, 'overall': None, 'confidence': None,
            'corroborated': None, 'position_biased': None,
            'prompt_version': PROMPT_VERSION, 'usage': usage or {}}


def _one_pass(pair, chunk_text, backend, chosen_pos, ledger, probe_id, max_tokens):
    """A single judged pass. Returns (payload_dict, None) or (None, abstention)."""
    system, user = build_prompt(pair, chunk_text, chosen_pos)
    try:
        resp = backend.generate(system, user, max_tokens=max_tokens, temperature=0.0,
                                probe_id=probe_id)
    except JudgeUnavailable as exc:
        if ledger is not None:
            ledger.record(None, abstained=True)
        return None, _abstention(ABSTAIN_TRANSPORT, str(exc))

    obj, parse_reason = jc.parse_judge_json(resp.text, required_keys=REQUIRED_KEYS)
    if obj is None:
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return None, _abstention(parse_reason, 'could not read a verdict', resp.usage())
    if str(obj.get('refusal', '')).strip():
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return None, _abstention(ABSTAIN_REFUSED, str(obj['refusal']), resp.usage())
    try:
        dims, overall, conf = _validate_payload(obj)
    except ValueError as exc:
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return None, _abstention(jc.ABSTAIN_INCOMPLETE, str(exc), resp.usage())

    if ledger is not None:
        ledger.record(resp, abstained=False)
    return {'dims': dims, 'overall': overall, 'confidence': conf,
            'evidence': obj.get('evidence', ''), 'chosen_pos': chosen_pos,
            'usage': resp.usage()}, None


def judge_dpo_pair(pair, chunk_text, backend, ledger=None, probe_id=None,
                   max_tokens=1024, force_swap_check=None):
    """Judge one DPO pair. Never raises; failures become abstentions.

    Runs a second, position-swapped pass for rejection types with no automated coverage
    (or whenever `force_swap_check=True`). Disagreement between the two passes means the
    judge was tracking position rather than content, and the result is JUDGE_REVIEW.
    """
    rtype = pair.get('rejection_type')
    swap_check = (force_swap_check if force_swap_check is not None
                  else rtype in NO_AUTOMATED_COVERAGE)

    pos1 = chosen_position(pair.get('source_chunk_id'))
    p1, ab = _one_pass(pair, chunk_text, backend, pos1, ledger,
                       probe_id, max_tokens)
    if ab is not None:
        return ab

    overall_1 = _to_chosen_terms(p1['overall'], pos1)
    dims_1 = {d: _to_chosen_terms(v, pos1) for d, v in p1['dims'].items()}

    position_biased = None
    passes = [p1]
    if swap_check:
        pos2 = chosen_position(pair.get('source_chunk_id'), swap=True)
        p2, ab2 = _one_pass(pair, chunk_text, backend, pos2, ledger,
                            (probe_id + '_swap') if probe_id else None, max_tokens)
        if ab2 is not None:
            # One good pass and one failed pass is not two passes. Do not silently
            # fall back to the single answer for exactly the types that need two.
            ab2['reason'] = ('swapped-position pass failed (%s); a single pass is not '
                             'sufficient for rejection_type %r'
                             % (ab2.get('reason'), rtype))
            return ab2
        passes.append(p2)
        overall_2 = _to_chosen_terms(p2['overall'], pos2)
        position_biased = (overall_1 != overall_2)
        if position_biased:
            return {'judge_verdict': JUDGE_REVIEW,
                    'reason': ('the two position-swapped passes disagree (%s vs %s) - '
                               'the judge tracked slot order, not content'
                               % (overall_1, overall_2)),
                    'dimensions': dims_1, 'overall': 'inconsistent',
                    'confidence': p1['confidence'], 'corroborated': None,
                    'position_biased': True, 'passes': len(passes),
                    'prompt_version': PROMPT_VERSION, 'usage': p1['usage']}

    # Corroboration: did the dimension implied by the declared type favour `chosen`?
    corr_dim = CORROBORATING_DIMENSION.get(rtype)
    corroborated = None
    if corr_dim is not None:
        corroborated = (dims_1.get(corr_dim) == 'chosen')

    if overall_1 == 'rejected':
        verdict = JUDGE_FAIL
        reason = 'the judge prefers the REJECTED response - the pair may be inverted'
    elif overall_1 == 'tie':
        verdict = JUDGE_REVIEW
        reason = 'the judge cannot separate the two responses'
    elif p1['confidence'] == 'low':
        verdict = JUDGE_REVIEW
        reason = 'chosen preferred, but the judge reported low confidence'
    else:
        verdict = JUDGE_PASS
        reason = 'chosen preferred%s' % (
            '' if corr_dim is None else
            ', and %s on %s' % ('corroborated' if corroborated else
                                'NOT corroborated', corr_dim))
        if corroborated is False:
            verdict = JUDGE_REVIEW
            reason = ('chosen preferred overall, but the declared rejection_type %r '
                      'implies %s should separate them and it did not'
                      % (rtype, corr_dim))

    return {'judge_verdict': verdict, 'reason': reason, 'dimensions': dims_1,
            'overall': overall_1, 'confidence': p1['confidence'],
            'evidence': p1['evidence'], 'corroborated': corroborated,
            'corroborating_dimension': corr_dim, 'position_biased': position_biased,
            'passes': len(passes), 'prompt_version': PROMPT_VERSION,
            'usage': p1['usage']}


# --------------------------------------------------- combination with check_dpo's triage

CONFIRMED = 'CONFIRMED'
NEEDS_HUMAN = 'NEEDS_HUMAN'
FLAGGED = 'FLAGGED'


def combine_with_dpo_rules(triage, judge_result):
    """Merge check_dpo.py's triage with a judge verdict.

    Differs from the SFT table in one deliberate place. `NEEDS_JUDGE` is check_dpo's
    explicit statement that the deterministic layer has finished and the decision belongs
    to the judge, so a judge PASS there CONFIRMS. In the SFT contract a rule-based REVIEW
    keeps a record in review whatever the judge says, because there the rule layer is
    uncertain rather than deferring. Uncertainty and delegation are different states and
    are resolved differently.

    Unchanged from the SFT table: a rule-based FLAG_SUSPICIOUS is never overturned, and
    a judge FAIL never auto-rejects - it routes to a human.
    """
    if triage == cd.FLAG_SUSPICIOUS:
        return FLAGGED, 'the deterministic layer flagged this pair; a judge cannot clear it'
    if judge_result is None:
        return NEEDS_HUMAN, 'no judge result supplied'
    jv = judge_result.get('judge_verdict')
    if jv in (JUDGE_REVIEW, JUDGE_ABSTAIN):
        return NEEDS_HUMAN, 'the judge did not reach a verdict (%s)' % jv
    if jv == JUDGE_FAIL:
        return NEEDS_HUMAN, 'the judge prefers the rejected response - needs a human'
    if jv != JUDGE_PASS:
        raise ValueError('unknown judge verdict %r' % jv)
    if triage == cd.AUTO_CONFIRM:
        return CONFIRMED, 'both layers agree'
    if triage == cd.NEEDS_JUDGE:
        return CONFIRMED, 'the deterministic layer deferred and the judge confirms'
    raise ValueError('unknown triage %r' % triage)


# ------------------------------------------------------------------------------ self-test

def _payload(overall='A', confidence='high', **dim_overrides):
    dims = {d: 'tie' for d in DIMENSIONS}
    dims.update(dim_overrides)
    return json.dumps({'dimensions': dims, 'overall': overall,
                       'evidence': 'e', 'confidence': confidence}, ensure_ascii=False)


_PAIR = {'prompt': 'اشرح', 'chosen': 'جيد', 'rejected': 'رديء',
         'source_chunk_id': 'asas_albalagha_c0001', 'source_region': 'classical',
         'rejection_type': 'partial_factual_errors', 'model_version': 't'}


def run_self_test():
    ok = True

    # --- the coverage claim in the docstring must match check_dpo, not be asserted ------
    if NO_AUTOMATED_COVERAGE != frozenset(['wrong_register', 'weak_organization']):
        print('  [FAIL] zero-coverage set drifted from check_dpo: %s'
              % sorted(NO_AUTOMATED_COVERAGE))
        ok = False

    # Every charter rejection type must map to a dimension, or a type silently loses
    # its corroboration when someone adds one.
    for t in cd.REJECTION_TYPES:
        if t not in CORROBORATING_DIMENSION:
            print('  [FAIL] rejection_type %r has no corroborating dimension' % t)
            ok = False
    for t, d in CORROBORATING_DIMENSION.items():
        if d not in DIMENSIONS:
            print('  [FAIL] %r maps to unknown dimension %r' % (t, d))
            ok = False
        if t not in cd.REJECTION_TYPES:
            print('  [FAIL] %r is not a charter rejection type' % t)
            ok = False

    # --- position handling --------------------------------------------------------------
    pos = chosen_position('asas_albalagha_c0001')
    if pos not in (WIN_A, WIN_B):
        print('  [FAIL] chosen_position returned %r' % pos)
        ok = False
    if chosen_position('asas_albalagha_c0001') != pos:
        print('  [FAIL] chosen_position is not deterministic')
        ok = False
    if chosen_position('asas_albalagha_c0001', swap=True) == pos:
        print('  [FAIL] swap did not move the chosen response')
        ok = False
    # Position must actually vary across ids, or the swap check is vacuous.
    seen = set(chosen_position('asas_albalagha_c%04d' % i) for i in range(40))
    if seen != {WIN_A, WIN_B}:
        print('  [FAIL] chosen_position never varies: %s' % seen)
        ok = False

    # The chosen response must really occupy the slot the caller asked for.
    _, ua = build_prompt(_PAIR, 'src', WIN_A)
    if ua.index('جيد') > ua.index('رديء'):
        print('  [FAIL] chosen was not placed in slot A')
        ok = False
    _, ub = build_prompt(_PAIR, 'src', WIN_B)
    if ub.index('جيد') < ub.index('رديء'):
        print('  [FAIL] chosen was not placed in slot B')
        ok = False

    # --- slot -> chosen/rejected translation -------------------------------------------
    for slot, cpos, want in [(WIN_A, WIN_A, 'chosen'), (WIN_B, WIN_A, 'rejected'),
                             (WIN_A, WIN_B, 'rejected'), (WIN_B, WIN_B, 'chosen'),
                             (WIN_TIE, WIN_A, 'tie'), (WIN_TIE, WIN_B, 'tie')]:
        if _to_chosen_terms(slot, cpos) != want:
            print('  [FAIL] slot %s with chosen at %s -> %s, expected %s'
                  % (slot, cpos, _to_chosen_terms(slot, cpos), want))
            ok = False

    # --- end to end, scripted -----------------------------------------------------------
    cpos = chosen_position(_PAIR['source_chunk_id'])
    other = WIN_B if cpos == WIN_A else WIN_A
    led = jc.UsageLedger()

    be = jc.ScriptedBackend(by_key={
        'good': _payload(overall=cpos, **{'factual_consistency': cpos}),
        'inverted': _payload(overall=other, **{'factual_consistency': other}),
        'tie': _payload(overall=WIN_TIE),
        'lowconf': _payload(overall=cpos, confidence='low',
                            **{'factual_consistency': cpos}),
        'uncorrob': _payload(overall=cpos),          # overall win, but the dim is a tie
        'garbage': 'no json here',
        'baddim': _payload(overall='C'),
        'boom': JudgeUnavailable('simulated failure'),
    })
    cases = [
        ('good', JUDGE_PASS), ('inverted', JUDGE_FAIL), ('tie', JUDGE_REVIEW),
        ('lowconf', JUDGE_REVIEW), ('uncorrob', JUDGE_REVIEW),
        ('garbage', JUDGE_ABSTAIN), ('baddim', JUDGE_ABSTAIN), ('boom', JUDGE_ABSTAIN),
    ]
    for pid, want in cases:
        r = judge_dpo_pair(_PAIR, 'src', be, ledger=led, probe_id=pid,
                           force_swap_check=False)
        if r['judge_verdict'] != want:
            print('  [FAIL] pair probe %-9s -> %s (%s), expected %s'
                  % (pid, r['judge_verdict'], r.get('reason'), want))
            ok = False

    # A declared type whose dimension did not separate the pair must not read as a pass.
    r = judge_dpo_pair(_PAIR, 'src', be, probe_id='uncorrob', force_swap_check=False)
    if r['corroborated'] is not False:
        print('  [FAIL] uncorroborated pair reported corroborated=%s' % r['corroborated'])
        ok = False

    # --- the swap check: agreement, disagreement, and a failed second pass --------------
    reg = dict(_PAIR, rejection_type='wrong_register')
    rpos = chosen_position(reg['source_chunk_id'])
    rswap = chosen_position(reg['source_chunk_id'], swap=True)

    # Consistent across both orders -> a real preference.
    be_ok = jc.ScriptedBackend(by_key={
        'r': _payload(overall=rpos, **{'arabic_quality': rpos}),
        'r_swap': _payload(overall=rswap, **{'arabic_quality': rswap}),
    })
    r = judge_dpo_pair(reg, 'src', be_ok, probe_id='r')
    if r['judge_verdict'] != JUDGE_PASS or r['position_biased'] is not False:
        print('  [FAIL] consistent swap check -> %s (biased=%s), expected PASS/False'
              % (r['judge_verdict'], r.get('position_biased')))
        ok = False
    if r.get('passes') != 2:
        print('  [FAIL] zero-coverage type did not run two passes')
        ok = False

    # The judge picks slot A both times -> it tracked position, not content.
    be_biased = jc.ScriptedBackend(by_key={
        'r': _payload(overall=WIN_A, **{'arabic_quality': WIN_A}),
        'r_swap': _payload(overall=WIN_A, **{'arabic_quality': WIN_A}),
    })
    r = judge_dpo_pair(reg, 'src', be_biased, probe_id='r')
    if r['judge_verdict'] != JUDGE_REVIEW or not r['position_biased']:
        print('  [FAIL] position-biased judge -> %s (biased=%s), expected REVIEW/True'
              % (r['judge_verdict'], r.get('position_biased')))
        ok = False

    # A failed second pass must abstain, NOT fall back to the single good pass.
    be_half = jc.ScriptedBackend(by_key={
        'r': _payload(overall=rpos, **{'arabic_quality': rpos}),
        'r_swap': JudgeUnavailable('second pass timed out'),
    })
    r = judge_dpo_pair(reg, 'src', be_half, probe_id='r')
    if r['judge_verdict'] != JUDGE_ABSTAIN:
        print('  [FAIL] a failed swap pass resolved to %s instead of abstaining'
              % r['judge_verdict'])
        ok = False

    # Both zero-coverage types must trigger the swap check by default.
    for t in NO_AUTOMATED_COVERAGE:
        p = dict(_PAIR, rejection_type=t)
        cp = chosen_position(p['source_chunk_id'])
        sp = chosen_position(p['source_chunk_id'], swap=True)
        b = jc.ScriptedBackend(by_key={'z': _payload(overall=cp,
                                                     **{CORROBORATING_DIMENSION[t]: cp}),
                                       'z_swap': _payload(overall=sp,
                                                          **{CORROBORATING_DIMENSION[t]: sp})})
        res = judge_dpo_pair(p, 'src', b, probe_id='z')
        if res.get('passes') != 2:
            print('  [FAIL] %r ran %s pass(es), expected 2' % (t, res.get('passes')))
            ok = False

    # A covered type must NOT pay for two calls.
    b = jc.ScriptedBackend(by_key={'c': _payload(overall=cpos,
                                                 **{'factual_consistency': cpos})})
    if judge_dpo_pair(_PAIR, 'src', b, probe_id='c').get('passes') != 1:
        print('  [FAIL] a covered rejection_type ran the swap check')
        ok = False

    # --- triage combination -------------------------------------------------------------
    combos = [
        (cd.FLAG_SUSPICIOUS, JUDGE_PASS, FLAGGED),
        (cd.FLAG_SUSPICIOUS, JUDGE_FAIL, FLAGGED),
        (cd.FLAG_SUSPICIOUS, JUDGE_ABSTAIN, FLAGGED),
        (cd.AUTO_CONFIRM, JUDGE_PASS, CONFIRMED),
        (cd.AUTO_CONFIRM, JUDGE_FAIL, NEEDS_HUMAN),
        (cd.AUTO_CONFIRM, JUDGE_REVIEW, NEEDS_HUMAN),
        (cd.AUTO_CONFIRM, JUDGE_ABSTAIN, NEEDS_HUMAN),
        (cd.NEEDS_JUDGE, JUDGE_PASS, CONFIRMED),
        (cd.NEEDS_JUDGE, JUDGE_FAIL, NEEDS_HUMAN),
        (cd.NEEDS_JUDGE, JUDGE_REVIEW, NEEDS_HUMAN),
        (cd.NEEDS_JUDGE, JUDGE_ABSTAIN, NEEDS_HUMAN),
    ]
    for tri, jv, want in combos:
        got, _ = combine_with_dpo_rules(tri, {'judge_verdict': jv})
        if got != want:
            print('  [FAIL] triage=%s judge=%s -> %s, expected %s' % (tri, jv, got, want))
            ok = False
    if combine_with_dpo_rules(cd.NEEDS_JUDGE, None)[0] != NEEDS_HUMAN:
        print('  [FAIL] a missing judge result was confirmed')
        ok = False
    # A judge FAIL must never auto-reject a pair outright.
    for tri in (cd.AUTO_CONFIRM, cd.NEEDS_JUDGE):
        if combine_with_dpo_rules(tri, {'judge_verdict': JUDGE_FAIL})[0] == FLAGGED:
            print('  [FAIL] a judge FAIL auto-flagged a pair')
            ok = False

    # --- the declared type must never reach the prompt ----------------------------------
    _, u = build_prompt(_PAIR, 'src', WIN_A)
    for t in cd.REJECTION_TYPES:
        if t in u:
            print('  [FAIL] rejection_type %r leaked into the judge prompt' % t)
            ok = False
    for d in DIMENSIONS:
        if d not in SYSTEM_PROMPT:
            print('  [FAIL] dimension %r is scored but never defined in the rubric' % d)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM judge for DPO pairs.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--show-prompt', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    if a.show_prompt:
        s, u = build_prompt(_PAIR, '<SOURCE CHUNK>', WIN_A)
        sys.stdout.write(s + '\n' + '-' * 70 + '\n' + u)
        sys.exit(0)
    ap.print_help()
