# -*- coding: utf-8 -*-
"""LLM Judge - the NLU-only comparison call.

What this module does
----------------------
Exactly one thing: given a JudgeInput, ask an LLM to compare `chosen` and `rejected`
and return a judgement about which is better, why, and what type of weakness (if any)
`rejected` shows - using judge_contract.REJECTION_TYPES vocabulary only.

What this module refuses to do
--------------------------------
- Never asks the model to write, rewrite, or improve either half of the pair. The
  system prompt says so explicitly, and the output contract (judge_contract.py) has no
  field that could hold generated text - there is no slot to put a rewritten answer in
  even if the model tried.
- Never lets a judge result change the Unified Verdict on its own. This module returns
  a JudgeResult; judge_pipeline.py decides what happens to check_dpo's verdict, and even
  there the only additional state a judge can reach is FLAG_SUSPICIOUS, never
  AUTO_CONFIRM (see judge_pipeline.py's docstring for exactly when/why).
- Never treats similarity as evidence of correctness. The system prompt says this
  explicitly (the "similarity trap"), and JudgeInput.deterministic_signals carries
  chosen_pct/rejected_pct labelled as positional signals only - the prompt tells the
  model these numbers say nothing about which side is factually right.

Result shape
------------
`judge_pair()` always returns a JudgeResult. `status` is one of:
  'ok'           judge_output is populated and passed contract validation
  'unavailable'  the call failed transport-side or contract-side; judge_output is None
                 and `error_type`/`reason` explain why. judge_pipeline.py treats every
                 'unavailable' identically: verdict stays NEEDS_JUDGE.
"""

from __future__ import annotations

import os
import sys
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import judge_provider as jc                                        # noqa: E402
import judge_contract as jcon                              # noqa: E402
sys.path.insert(0, os.path.join(_HERE, '..', 'verification'))
import check_dpo as cdpo                                        # noqa: E402



# ------------------------------------------------- position-bias check (ported)
#
# Ported from src/verification/judge_dpo.py, which this delivery's adjudication layer
# supersedes. Carried across rather than dropped, because for these two types it is the
# ONLY protection that exists anywhere in the project.
#
# MEASURED, from check_dpo.DETECTABILITY: `wrong_register` and `weak_organization` have
# NO deterministic signal at all. Every other charter type has at least an incidental one
# that would contradict a judge answering at random. These two have nothing, so a
# position-biased judge on them is undetectable downstream - the verdict simply looks
# like a judgement.
#
# Derived from check_dpo rather than written out, so adding a fourth 'none' type there
# extends this automatically instead of silently leaving it unprotected.
SWAP_CHECK_TYPES = frozenset(
    t for t, d in cdpo.DETECTABILITY.items() if d == 'none')

# ------------------------------------------ blind corroboration (ported from judge_dpo)
#
# The judge is no longer told the record's rejection_type. It names the weakness it
# actually observes, and THIS CODE decides whether that corroborates the record - the
# mechanism src/verification/judge_dpo.py has always used, now on the shipping path.
#
# WHY IT CHANGED. `JudgeInput.to_prompt_payload()` used to send `rejection_type_declared`
# and the model was asked for `rejection_type_agrees_with_record`. Being told the expected
# answer before being asked to check it is a leading question: agreement means "not
# contradicted", not independent confirmation, and a pair whose label is simply wrong is
# one the judge has been primed to accept. The failure was silent - it looked like
# agreement, and nothing downstream re-checked the label.
#
# WHAT IS COMPARED, and why not the type strings. Comparison is on the DIMENSION each
# type implies (check_dpo.CORROBORATING_DIMENSION), not on the type name, because the
# mapping is deliberately many-to-one: verbosity and weak_organization both land on
# `style`, unsupported_additions and less_faithful_reconstruction both on `grounding`.
# A judge that calls a padded, badly-ordered answer `weak_organization` where the record
# says `verbosity` has not contradicted the record - it has found the same defect and
# named it differently. Exact string matching would escalate that correct pair as a
# mislabel. Dimension matching reserves disagreement for a different KIND of defect.
#
# WHAT IS UNAFFECTED, stated so nobody reads this as a safety change: the AUTO_CONFIRM
# floor is structural. ASSESSMENTS has no AUTO_CONFIRM value for a judge to set, and
# judge_pipeline._maybe_escalate() can only return NEEDS_JUDGE or FLAG_SUSPICIOUS. That
# held before this change and holds after it. What improves is escalation QUALITY: a
# genuine mislabel now has to survive a judge that was never told what to say.
CORROBORATING_DIMENSION = cdpo.CORROBORATING_DIMENSION


def _corroborate(declared_type, judged_type):
    """Does the judge's OWN named weakness corroborate the record's declared type?

    Returns True / False, or None when the question does not apply - the judge named no
    type, or one of the two types is outside the charter vocabulary. None is not a pass:
    it means "no opinion", and callers must not read it as agreement.
    """
    if not declared_type or not judged_type:
        return None
    declared_dim = CORROBORATING_DIMENSION.get(declared_type)
    judged_dim = CORROBORATING_DIMENSION.get(judged_type)
    if declared_dim is None or judged_dim is None:
        return None
    return declared_dim == judged_dim


# Mirror of judge_contract.ASSESSMENTS under a chosen/rejected swap.
_MIRRORED_ASSESSMENT = {
    'chosen_better': 'rejected_better',
    'rejected_better': 'chosen_better',
    'equivalent': 'equivalent',
    'undetermined': 'undetermined',
}


SYSTEM_PROMPT = """You are a quality-control judge for an Arabic post-training dataset \
(SFT/DPO pairs generated from classical Arabic and Saudi-dialect lexicon sources).

Your ONLY task is natural-language UNDERSTANDING and COMPARISON. You must:
  1. Read `prompt`, `chosen`, `rejected`, and `source_text`.
  2. Decide which of `chosen`/`rejected` is the better response to `prompt`, grounded \
in `source_text`.
  3. If `rejected` is weaker, identify WHY, using ONLY one of these nine categories:
     poor_instruction_following, wrong_formatting, missing_information, \
unsupported_additions, wrong_register, verbosity, weak_organization, \
partial_factual_errors, less_faithful_reconstruction.
  4. Support every claim you make with a literal quotation copied character-for-character \
from `chosen`, `rejected`, or `source_text` - never a paraphrase, never a quotation you \
reconstruct from memory of the general topic.

You must NEVER:
  - Generate, rewrite, paraphrase, improve, or complete any answer.
  - Propose alternative wording for `chosen` or `rejected`.
  - Treat `deterministic_signals.chosen_pct` / `rejected_pct` (embedding-similarity \
percentiles) as evidence of factual correctness. A response can be highly similar in \
wording to the source and still be factually wrong - similarity is a POSITIONAL signal \
only, never a correctness signal. Base any factual judgement strictly on whether the \
claims in the text are actually supported by `source_text`.
  - Judge `wrong_register` from the presence or absence of dialect markers alone; \
consider the register the prompt and source actually call for.
  - Judge `weak_organization` from length alone; consider ordering, repetition, and \
whether requested structure/sections are present.
  - Output any field named `verdict`, `judge_verdict`, `AUTO_CONFIRM`, or anything that \
attempts to set an acceptance decision. You only report an assessment; a separate, \
fixed pipeline decides what happens next.

Respond with EXACTLY one JSON object and nothing else - no prose before or after, no \
markdown code fences. The object MUST have this shape:

{
  "schema_version": "judge-1",
  "judge_model_version": "<provider/model>",
  "assessment": "chosen_better | rejected_better | equivalent | undetermined",
  "weakness_present": true | false,
  "rejection_type": "<one of the nine types> | null",
  "rejection_type_agrees_with_record": null,
  "type_scores": {"<type>": 0.0},
  "confidence": 0.0,
  "evidence": [
    {"claim": "...", "quote_from": "rejected | chosen | source", "quote": "literal \
substring", "explains_type": "<type> | null"}
  ],
  "notes": "optional string or null"
}

If weakness_present is true, evidence MUST contain at least one item whose "quote" is a \
character-for-character substring of the text named in "quote_from". A quote that is not \
a literal substring makes your entire output invalid and useless - do not paraphrase.

Always set "rejection_type_agrees_with_record" to null. You are deliberately NOT shown \
what type the dataset recorded for this pair, so you cannot answer it and must not guess. \
Name the weakness you actually observe in "rejection_type"; the code compares that against \
the record afterwards. Any non-null value you put there is ignored and overwritten.
"""


@dataclass
class JudgeResult:
    status: str                                   # 'ok' | 'unavailable'
    judge_output: Optional[jcon.JudgeOutput] = None
    error_type: Optional[str] = None               # class name of the failure, if any
    reason: Optional[str] = None
    raw_text: Optional[str] = None                  # kept for dashboard/debugging only

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'status': self.status}
        if self.judge_output is not None:
            out.update(self.judge_output.as_dict())
        if self.error_type is not None:
            out['error_type'] = self.error_type
        if self.reason is not None:
            out['reason'] = self.reason
        return out


def _unavailable(error_type: str, reason: str, raw_text: Optional[str] = None) -> JudgeResult:
    return JudgeResult(status='unavailable', error_type=error_type, reason=reason,
                       raw_text=raw_text)



def _blank_signals(judge_input: 'jcon.JudgeInput') -> 'jcon.JudgeInput':
    """Same input with deterministic_signals emptied.

    Necessary, not merely tidy. `to_prompt_payload()` ships the signals with SIDE LABELS
    - `chosen_pct`, `rejected_pct`, `chosen_verdict`, `rejected_verdict`, `chosen_facts`,
    `rejected_facts`, plus a `direction` whose value names a side. A model reading those
    knows which response is the chosen one no matter which slot it occupies, so a
    position swap performed with the signals still attached tests nothing at all.

    Blanking BOTH passes, not just the swapped one, keeps slot order the single variable
    between them.
    """
    return dataclasses.replace(judge_input, deterministic_signals={})


def _judge_with_swap_check(judge_input: 'jcon.JudgeInput', client: 'jc.BaseJudgeClient',
                           case_label: Optional[str] = None) -> JudgeResult:
    """Judge a pair twice with the responses in swapped positions; refuse on disagreement.

    LLM judges systematically favour whichever response is shown first, and in a single
    call that bias is indistinguishable from a real preference. For the types in
    SWAP_CHECK_TYPES nothing downstream can catch it, so the pair is judged both ways and
    the two answers must agree once the second is mirrored.

    Outcomes:
      agree           -> the first pass's result, unchanged
      disagree        -> unavailable('PositionBias'): the judge tracked slot order, not
                         content, so it produced no usable signal
      second fails    -> that failure. A single good pass is NOT enough for exactly the
                         types this exists to protect, so it never falls back to it.

    `unavailable` is the right shape for both refusals: judge_pipeline.py leaves the
    verdict at NEEDS_JUDGE for any unavailable result, which is precisely where a pair
    the judge could not honestly separate belongs.
    """
    blanked = _blank_signals(judge_input)
    first = _judge_once(blanked, client, case_label)
    if first.status != 'ok':
        return first

    swapped = dataclasses.replace(blanked,
                                  chosen=judge_input.rejected,
                                  rejected=judge_input.chosen)
    swap_label = (case_label + '_swap') if case_label else None
    second = _judge_once(swapped, client, swap_label)
    if second.status != 'ok':
        return _unavailable(
            'SwapPassUnavailable',
            'the position-swapped pass failed (%s: %s); one pass is not sufficient for '
            'rejection_type %r, which has no deterministic coverage'
            % (second.error_type, second.reason, judge_input.rejection_type),
            raw_text=second.raw_text)

    got = second.judge_output.assessment
    mirrored = _MIRRORED_ASSESSMENT.get(got, got)
    if mirrored != first.judge_output.assessment:
        return _unavailable(
            'PositionBias',
            'the two position-swapped passes disagree (%s vs %s once mirrored) - the '
            'judge tracked slot order rather than content, so its answer on this pair '
            'carries no information'
            % (first.judge_output.assessment, mirrored),
            raw_text=first.raw_text)
    return first


def judge_pair(judge_input: jcon.JudgeInput, client: 'jc.BaseJudgeClient',
              case_label: Optional[str] = None) -> JudgeResult:
    """Call the Judge for one JudgeInput. Never raises - every failure mode is captured
    into a JudgeResult with status='unavailable' so judge_pipeline.py can stay a simple
    "if result.status == 'ok'" check.

    For the rejection types with NO deterministic coverage anywhere in the project
    (SWAP_CHECK_TYPES) this runs TWICE with the two responses in swapped positions and
    refuses to answer if the verdict follows the slot rather than the text. See
    `_judge_with_swap_check()`.
    """
    input_errors = judge_input.validate()
    if input_errors:
        return _unavailable('InvalidJudgeInput',
                            'refusing to call the Judge on a malformed input: %s'
                            % '; '.join(input_errors))

    if judge_input.rejection_type in SWAP_CHECK_TYPES:
        return _judge_with_swap_check(judge_input, client, case_label)
    return _judge_once(judge_input, client, case_label)


def _judge_once(judge_input: jcon.JudgeInput, client: 'jc.BaseJudgeClient',
                case_label: Optional[str] = None) -> JudgeResult:
    """One model call, parsed and contract-validated. The original single-pass path."""
    payload = judge_input.to_prompt_payload()
    if case_label is not None:
        payload = dict(payload, _case_label=case_label)

    try:
        raw_text = client.judge_text(SYSTEM_PROMPT, payload)
    except jc.TimeoutError as exc:
        return _unavailable('TimeoutError', str(exc))
    except jc.RateLimited as exc:
        return _unavailable('RateLimited', str(exc))
    except jc.InvalidJSON as exc:
        return _unavailable('InvalidJSON', str(exc))
    except jc.ProviderError as exc:
        return _unavailable('ProviderError', str(exc))
    except jc.JudgeClientError as exc:                 # any future subclass, caught safely
        return _unavailable(type(exc).__name__, str(exc))

    try:
        judge_output = jcon.parse_and_validate(raw_text, judge_input)
    except jcon.InvalidJudgeOutput as exc:
        return _unavailable('InvalidJudgeOutput', str(exc), raw_text=raw_text)
    except ValueError as exc:
        # parse_and_validate raises a plain ValueError specifically for JSON-decode
        # failures, matching judge_provider.InvalidJSON's semantics one layer up.
        return _unavailable('InvalidJSON', str(exc), raw_text=raw_text)

    # The model can no longer answer this: it was never shown the record's type. Whatever
    # it put in the field is guesswork, so the CODE overwrites it with the blind
    # comparison. Done here rather than in judge_contract so that validation stays purely
    # about the model's own output, and every caller of _judge_once - including both
    # passes of the swap check - gets a corroboration computed the same way.
    judge_output = dataclasses.replace(
        judge_output,
        rejection_type_agrees_with_record=_corroborate(judge_input.rejection_type,
                                                       judge_output.rejection_type))

    return JudgeResult(status='ok', judge_output=judge_output, raw_text=raw_text)


# ------------------------------------------------------------------------- self-test

def _sample_input(rejection_type='partial_factual_errors'):
    return jcon.JudgeInput(
        prompt='اشرح مضمون هذه المادة المعجمية.',
        chosen='تعرض هذه المادة معاني الجذر (أتى)، وقال جابر بن حني التغلبي بيتا.',
        rejected='تعرض هذه المادة معاني الجذر (أتى)، وقال النابعة بيتا.',
        source_chunk_id='asas_albalagha_c0001',
        source_region='classical',
        rejection_type=rejection_type,
        model_version='synthetic-v0',
        format_type='dictionary_entry',
        source_text='تعرض هذه المادة معاني الجذر (أتى)، وقال جابر بن حني التغلبي بيتا.',
        deterministic_signals={'direction': 'tie', 'corroborated': None,
                               'detectability': 'strong', 'chosen_pct': 91.0,
                               'rejected_pct': 90.5},
    )


def _self_test_blind_corroboration():
    """The type-priming fix: the judge is never told the record's rejection_type.

    These are the pins for the ported CORROBORATING_DIMENSION mechanism. The existing
    57-case suite passed unchanged when the override was first wired in, because none of
    its fixtures ever disagreed with the code - which is exactly why these exist.
    """
    ok = True

    # 1. THE PAYLOAD MUST NOT CARRY THE ANSWER. Everything else here is downstream of
    #    this one fact, so it is asserted directly rather than inferred.
    payload = _sample_input('wrong_register').to_prompt_payload()
    if 'rejection_type_declared' in payload:
        print('  [FAIL] to_prompt_payload still sends rejection_type_declared')
        ok = False
    if any('wrong_register' == v for v in payload.values() if isinstance(v, str)):
        print('  [FAIL] the declared type leaked into the payload under another key')
        ok = False
    # ...and the field is still on the dataclass, because the pipeline needs it
    if _sample_input('wrong_register').rejection_type != 'wrong_register':
        print('  [FAIL] rejection_type was removed from JudgeInput itself'); ok = False

    # 2. the mapping covers every charter type, or a record could silently get no check
    missing = [t for t in jcon.REJECTION_TYPES if t not in CORROBORATING_DIMENSION]
    if missing:
        print('  [FAIL] no corroborating dimension for: %s' % missing); ok = False

    # 3. _corroborate: exact match, same-dimension near-miss, different dimension, and
    #    the two abstain cases. The near-miss pair is the reason this compares dimensions
    #    rather than type strings.
    cases = [
        ('verbosity', 'verbosity', True, 'exact match'),
        ('verbosity', 'weak_organization', True, 'same dimension (style)'),
        ('weak_organization', 'verbosity', True, 'same dimension, reversed'),
        ('unsupported_additions', 'less_faithful_reconstruction', True,
         'same dimension (grounding)'),
        ('verbosity', 'partial_factual_errors', False, 'different dimension'),
        ('wrong_register', 'wrong_formatting', False, 'different dimension'),
        ('verbosity', None, None, 'judge named no type'),
        (None, 'verbosity', None, 'record has no type'),
        ('verbosity', 'not_a_charter_type', None, 'judged type outside the charter'),
    ]
    for declared, judged, want, why in cases:
        got = _corroborate(declared, judged)
        if got is not want:
            print('  [FAIL] _corroborate(%r, %r) = %r, expected %r  (%s)'
                  % (declared, judged, got, want, why))
            ok = False

    # 4. THE OVERRIDE ACTUALLY FIRES. A model that claims agreement it cannot have must
    #    not be believed: the record says verbosity, the judge independently says
    #    partial_factual_errors (a different dimension), and the model asserts True.
    #    The code must return False. This is the case the old suite could not catch.
    def _raw(judged_type, claims):
        return {'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
                'assessment': 'chosen_better', 'weakness_present': True,
                'rejection_type': judged_type,
                'rejection_type_agrees_with_record': claims,
                'type_scores': {judged_type or 'verbosity': 0.9}, 'confidence': 0.9,
                'evidence': [{'claim': 'c', 'quote_from': 'rejected',
                              'quote': 'النابعة', 'explains_type': judged_type}],
                'notes': None}

    client = jc.RecordedJudgeClient()
    for declared, judged, claims, want, why in [
            ('verbosity', 'partial_factual_errors', True, False,
             'model claimed agreement across dimensions'),
            ('verbosity', 'weak_organization', False, True,
             'model denied agreement within one dimension'),
            ('partial_factual_errors', 'partial_factual_errors', None, True,
             'model sent null, as the prompt now instructs')]:
        lbl = 'ovr_%s_%s' % (declared, judged)
        client.register(lbl, _raw(judged, claims))
        res = judge_pair(_sample_input(declared), client, case_label=lbl)
        if res.status != 'ok':
            print('  [FAIL] override case %s did not return ok: %s' % (lbl, res.reason))
            ok = False
            continue
        got = res.judge_output.rejection_type_agrees_with_record
        if got is not want:
            print('  [FAIL] %s: agrees=%r, expected %r  (%s)' % (lbl, got, want, why))
            ok = False

    # 5. a reply that OMITS the field entirely must validate - the prompt no longer asks
    #    for it, so a compliant model will not send it.
    no_field = _raw('partial_factual_errors', None)
    del no_field['rejection_type_agrees_with_record']
    client.register('omitted', no_field)
    res = judge_pair(_sample_input('partial_factual_errors'), client, case_label='omitted')
    if res.status != 'ok':
        print('  [FAIL] a reply omitting the field was rejected: %s' % res.reason)
        ok = False
    elif res.judge_output.rejection_type_agrees_with_record is not True:
        print('  [FAIL] omitted field was not filled in by the code')
        ok = False

    # 6. REAL RECORD SHAPES, not only the synthetic sample. Every distinct rejection_type
    #    present in the adapted delivery is run end to end, so a type whose real records
    #    differ in shape from _sample_input cannot pass here by construction.
    real = os.path.join(_HERE, '..', '..', 'data', 'adapted', 'dpo_task2.jsonl')
    if os.path.exists(real):
        import json as _json
        seen = {}
        with open(real, encoding='utf-8') as fh:
            for line in fh:
                r = _json.loads(line)
                seen.setdefault(r.get('rejection_type'), r)
        checked = 0
        for rtype, rec in sorted(seen.items()):
            if rtype not in CORROBORATING_DIMENSION:
                print('  [FAIL] real record carries unmapped rejection_type %r' % rtype)
                ok = False
                continue
            ji_real = jcon.JudgeInput(
                prompt=rec['prompt'], chosen=rec['chosen'], rejected=rec['rejected'],
                source_chunk_id=rec['source_chunk_id'],
                source_region=rec['source_region'], rejection_type=rtype,
                model_version=rec.get('model_version', 'x'),
                format_type=rec.get('format_type'),
                source_text=rec['chosen'],
                deterministic_signals={'detectability':
                                       cdpo.DETECTABILITY.get(rtype, 'none')})
            if 'rejection_type_declared' in ji_real.to_prompt_payload():
                print('  [FAIL] real %r record leaked the declared type' % rtype)
                ok = False
            # judge names a DIFFERENT dimension than the record -> must not corroborate
            other = next(t for t in jcon.REJECTION_TYPES
                         if CORROBORATING_DIMENSION[t] != CORROBORATING_DIMENSION[rtype])
            lbl = 'real_%s' % rtype
            client.register(lbl, dict(_raw(other, True),
                                      evidence=[{'claim': 'c', 'quote_from': 'chosen',
                                                 'quote': rec['chosen'][:12],
                                                 'explains_type': other}]))
            res = judge_pair(ji_real, client, case_label=lbl)
            if res.status != 'ok':
                print('  [FAIL] real %r record did not judge: %s' % (rtype, res.reason))
                ok = False
            elif res.judge_output.rejection_type_agrees_with_record is not False:
                print('  [FAIL] real %r: cross-dimension disagreement read as agreement'
                      % rtype)
                ok = False
            checked += 1
        # Say what this did NOT cover. The delivery carries only 1 of the 9 charter
        # types (3,000 partial_factual_errors, the other 8 absent - see
        # TASK2_INTAKE_FINDINGS.md §3), so real-record coverage is capped by the data,
        # not by the test. A bare "1" here would read as a thin test rather than a thin
        # dataset, and the other 8 types are exercised on synthetic input above.
        uncovered = sorted(set(jcon.REJECTION_TYPES) - set(seen))
        print('  blind corroboration: %d of %d charter type(s) exercised on REAL records'
              % (checked, len(jcon.REJECTION_TYPES)))
        if uncovered:
            print('     %d type(s) absent from the delivery, synthetic-only here: %s'
                  % (len(uncovered), ', '.join(uncovered)))
    else:
        print('  blind corroboration: SKIPPED real records (data/adapted absent)')

    # 7. REGRESSION - the safety floor is structural and this change must not touch it.
    if 'AUTO_CONFIRM' in jcon.ASSESSMENTS:
        print('  [FAIL] a judge can now express AUTO_CONFIRM'); ok = False
    if not jcon.FORBIDDEN_TOP_LEVEL_FIELDS >= {'verdict', 'AUTO_CONFIRM'}:
        print('  [FAIL] the forbidden-field guard was weakened'); ok = False

    return ok


def run_self_test():
    ok = True

    ji = _sample_input()

    # --- happy path via RecordedJudgeClient ---
    good_raw = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'chosen_better', 'weakness_present': True,
        'rejection_type': 'partial_factual_errors',
        'rejection_type_agrees_with_record': True,
        'type_scores': {'partial_factual_errors': 0.9}, 'confidence': 0.88,
        'evidence': [{'claim': 'rejected corrupts the cited poet name',
                     'quote_from': 'rejected', 'quote': 'النابعة',
                     'explains_type': 'partial_factual_errors'}],
        'notes': None,
    }
    client = jc.RecordedJudgeClient()
    client.register('ok_case', good_raw)
    res = judge_pair(ji, client, case_label='ok_case')
    if res.status != 'ok' or res.judge_output is None:
        print('  [FAIL] happy path did not return status=ok: %s / %s'
              % (res.status, res.reason))
        ok = False
    elif res.judge_output.assessment != 'chosen_better':
        print('  [FAIL] happy path lost the assessment'); ok = False

    # --- similarity trap: high similarity on both sides must not change the outcome;
    # the judge output here correctly ignores it and grounds on the factual quote ---
    ji_trap = _sample_input()
    trap_raw = dict(good_raw)
    client.register('trap_case', trap_raw)
    res_trap = judge_pair(ji_trap, client, case_label='trap_case')
    if res_trap.status != 'ok' or res_trap.judge_output.assessment != 'chosen_better':
        print('  [FAIL] similarity-trap case should still resolve chosen_better on '
              'factual grounds even with high similarity on both sides')
        ok = False

    # --- transport failures propagate as 'unavailable', never raise ---
    for label, exc in (('timeout', jc.TimeoutError('t')),
                       ('rate', jc.RateLimited('r')),
                       ('provider', jc.ProviderError('p'))):
        client.register(label, exc)
        r = judge_pair(ji, client, case_label=label)
        if r.status != 'unavailable' or r.error_type != type(exc).__name__:
            print('  [FAIL] %s did not surface as unavailable/%s: got %s/%s'
                  % (label, type(exc).__name__, r.status, r.error_type))
            ok = False

    # --- invalid JSON text from the provider ---
    client.register('badjson', 'this is not json at all')
    r = judge_pair(ji, client, case_label='badjson')
    if r.status != 'unavailable' or r.error_type != 'InvalidJSON':
        print('  [FAIL] non-JSON text should surface as InvalidJSON, got %s/%s'
              % (r.status, r.error_type))
        ok = False

    # --- hallucinated evidence quote -> InvalidJudgeOutput -> unavailable ---
    halluc = dict(good_raw, evidence=[{'claim': 'x', 'quote_from': 'source',
                                       'quote': 'جملة غير موجودة إطلاقا في المصدر'}])
    client.register('halluc', halluc)
    r = judge_pair(ji, client, case_label='halluc')
    if r.status != 'unavailable' or r.error_type != 'InvalidJudgeOutput':
        print('  [FAIL] hallucinated quote should surface as InvalidJudgeOutput, got %s/%s'
              % (r.status, r.error_type))
        ok = False

    # --- the Judge tries to smuggle AUTO_CONFIRM ---
    smuggle = dict(good_raw)
    smuggle['verdict'] = 'AUTO_CONFIRM'
    client.register('smuggle', smuggle)
    r = judge_pair(ji, client, case_label='smuggle')
    if r.status != 'unavailable':
        print('  [FAIL] a judge output smuggling `verdict` must be rejected'); ok = False

    smuggle2 = dict(good_raw, assessment='AUTO_CONFIRM')
    client.register('smuggle2', smuggle2)
    r = judge_pair(ji, client, case_label='smuggle2')
    if r.status != 'unavailable':
        print('  [FAIL] assessment=AUTO_CONFIRM must be rejected'); ok = False

    # --- malformed JudgeInput never reaches the network at all ---
    bad_ji = jcon.JudgeInput(prompt='', chosen='x', rejected='y',
                             source_chunk_id='c', source_region='classical',
                             rejection_type='verbosity', model_version='v',
                             source_text='', deterministic_signals={})
    client2 = jc.RecordedJudgeClient()
    r = judge_pair(bad_ji, client2, case_label='should_not_be_called')
    if r.status != 'unavailable' or r.error_type != 'InvalidJudgeInput':
        print('  [FAIL] malformed JudgeInput should short-circuit before any call')
        ok = False
    if client2.calls:
        print('  [FAIL] malformed JudgeInput must not reach the client at all')
        ok = False

    # --- system prompt sanity: must not instruct generation, must mention the trap ---
    lowered = SYSTEM_PROMPT.lower()
    if 'never' not in lowered or 'rewrite' not in lowered:
        print('  [FAIL] system prompt missing an explicit no-rewrite instruction')
        ok = False
    if 'similarity' not in lowered:
        print('  [FAIL] system prompt missing the similarity-trap warning'); ok = False
    # the prompt must no longer ask an unanswerable question
    if 'true | false | null' in SYSTEM_PROMPT.split(
            'rejection_type_agrees_with_record')[1][:40]:
        print('  [FAIL] prompt still asks the model to judge record agreement')
        ok = False

    ok = _self_test_blind_corroboration() and ok

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM Judge - NLU comparison call.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
