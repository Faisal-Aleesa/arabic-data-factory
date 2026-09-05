# -*- coding: utf-8 -*-
"""Task 3, Group B - LLM judge for SFT records.

Scores what the rule-based layer structurally cannot reach: tone, completeness,
naturalness, and register. `check_facts.py` compares propositions against a fact table,
`check_similarity.py` measures embedding proximity, `check_format.py` matches structural
patterns. None of them can say whether an answer reads like Arabic a competent
lexicographer would write, or whether it actually served the instruction it was given.

حَكَم لغوي للسجلات: يقيس ما لا تصل إليه الفحوص القاعدية - الأسلوب والاكتمال والسجل.

Four dimensions, from the project charter
-----------------------------------------
  grounding           does it stick to the source chunk
  completeness        does it fully answer, without truncation or a dangling clause
  format_correctness  does it match its declared format_type
  arabic_quality      natural, grammatical, correct register and dialect

Two of these overlap with rule-based checks on purpose. `grounding` and
`format_correctness` are scored independently here so that agreement between the layers
is evidence rather than assumption - if the judge and `check_facts.py` disagree about
grounding on a record, that disagreement is the useful signal, and it can only exist if
both measured it.

THE KEY DECISION: the judge never sees the rule-based verdict
-------------------------------------------------------------
`judge_sft_record()` takes the record and the source chunk. It does NOT take, and must
never be given, the output of `verify_sft.py`. A judge told "the rule layer passed this"
will anchor on it, and the disagreement policy in
`src/deployment/ACCEPTANCE_CONTRACT.md` becomes worthless the moment the two layers stop
being independent - a judge that agrees because it was told the answer corroborates
nothing. The whole value of the second layer is that it is wrong in different places.

The same reasoning forbids passing the declared `rejection_type` to the DPO judge; see
`judge_dpo.py`, where it matters even more.

Verdict shape, and why it is these four values
----------------------------------------------
`JUDGE_PASS` / `JUDGE_FAIL` / `JUDGE_REVIEW` / `JUDGE_ABSTAIN`, defined in
`judge_client.py`. The contract's disagreement table needs exactly these: it routes a
judge FAIL over a rule PASS to review, and it routes "any judge REVIEW / abstain" to
review as well. ABSTAIN is a separate value from REVIEW because they mean different
things - REVIEW is the judge saying "I looked and I am unsure", ABSTAIN is the judge
never having answered (backend down, body unparseable, required key missing). Collapsing
them would hide a broken pipeline as model uncertainty.

`combine_with_rules()` implements the contract's table and is self-tested on every cell.
It is NOT wired into `verify_sft.acceptance_decision()`, which still knows nothing about
a judge. That wiring waits for real judge output, exactly as the contract requires: every
threshold in this project that was set before meeting real data had to be corrected
afterwards.

Not calibrated
--------------
No real model has scored a single record through this module - the judging backend is a
Qwen instance whose endpoint is still pending. The rubric and the parsing are exercised
against a scripted backend and 13 hand-built probes. That validates the plumbing and the
prompt's SHAPE. It does not validate the model's judgement, which cannot be assessed
until a real model answers.
"""

from __future__ import unicode_literals

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge_client as jc              # noqa: E402
from judge_client import (             # noqa: E402
    JUDGE_PASS, JUDGE_FAIL, JUDGE_REVIEW, JUDGE_ABSTAIN,
    JudgeUnavailable, ABSTAIN_TRANSPORT, ABSTAIN_REFUSED,
)

PROMPT_VERSION = 'sft-v1'

DIMENSIONS = ('grounding', 'completeness', 'format_correctness', 'arabic_quality')

# Per-dimension answers the judge may give. Anything else is treated as unparseable.
DIM_PASS = 'pass'
DIM_FAIL = 'fail'
DIM_REVIEW = 'review'
DIM_VALUES = (DIM_PASS, DIM_FAIL, DIM_REVIEW)

CONFIDENCE_VALUES = ('high', 'medium', 'low')

REQUIRED_KEYS = ('dimensions', 'confidence')


# ------------------------------------------------------------------------------ the rubric
#
# Bilingual, matching src/data_engineering/'s convention. The Arabic is not decoration:
# the judge is assessing Arabic register, and a rubric that describes register only in
# English invites the model to reason about the task in English and answer about a
# language it was not asked to think in.

SYSTEM_PROMPT = """\
You are a meticulous evaluator of Arabic instruction-tuning data derived from two
lexicographic sources: a classical Arabic rhetoric dictionary and a Saudi regional
dialect dictionary.

أنت مُقيّم دقيق لبيانات التدريب العربية المشتقة من معجمين: معجم بلاغة كلاسيكي ومعجم
لهجات سعودية.

You will be shown a SOURCE passage, an INSTRUCTION, and a RESPONSE. Judge the RESPONSE
only. Score exactly four dimensions:

1. grounding - Is every substantive claim in the RESPONSE supported by the SOURCE?
   Invented lemmas, invented poets, invented dates, or content drawn from general
   knowledge rather than the SOURCE are failures, however plausible they read.
   A confident tone is not evidence. Fluency is not grounding.

2. completeness - Does the RESPONSE fully answer the INSTRUCTION? Truncation, a
   dangling clause, an abrupt stop, or an answer covering only part of what was asked
   are failures. A short but complete answer is NOT a failure.

3. format_correctness - Does the RESPONSE match the declared FORMAT_TYPE?
   dictionary_entry = headword followed by glosses/usages, dictionary register.
   prose / narrative_paragraph = flowing connected sentences.
   verse = metrical lines. list = enumerated items. footnote_block = short notes.

4. arabic_quality - Is the Arabic grammatical, natural, and in an appropriate register?
   A lexicographic answer written in casual chat Arabic is a register failure even when
   every fact is right. Dialect content should read as that dialect; classical content
   should read as classical. Disfluency, agreement errors, and calques are failures.

Rules:
- Judge ONLY against the SOURCE shown. If the SOURCE does not contain something the
  RESPONSE asserts, that is a grounding failure even if you believe it is true.
- If the RESPONSE is fluent and confident but does not actually address the SOURCE,
  that is a grounding failure, not a pass.
- Use "review" for a dimension you genuinely cannot decide. Do not use it to avoid
  committing on a clear case.
- Set confidence to "low" if the SOURCE is too fragmentary to judge against.

Answer with ONLY a JSON object, no commentary:

{"dimensions": {"grounding": {"verdict": "pass|fail|review", "evidence": "<short>"},
                "completeness": {"verdict": "...", "evidence": "..."},
                "format_correctness": {"verdict": "...", "evidence": "..."},
                "arabic_quality": {"verdict": "...", "evidence": "..."}},
 "confidence": "high|medium|low"}
"""

USER_TEMPLATE = """\
FORMAT_TYPE: {format_type}
SOURCE_REGION: {source_region}

SOURCE:
{chunk_text}

INSTRUCTION:
{instruction}

RESPONSE:
{response}
"""


def build_prompt(record, chunk_text):
    """The (system, user) pair for one SFT record. Pure - no I/O, no model."""
    return SYSTEM_PROMPT, USER_TEMPLATE.format(
        format_type=record.get('format_type', '(unspecified)'),
        source_region=record.get('source_region', '(unspecified)'),
        chunk_text=chunk_text or '(source unavailable)',
        instruction=record.get('instruction', ''),
        response=record.get('response', ''))


# ----------------------------------------------------------------------- verdict assembly

def combine_dimensions(dims, confidence):
    """Fold four per-dimension answers into one judge verdict.

    Any dimension failing is a FAIL: the dimensions are not commensurable and do not
    average. A response that is perfectly formatted, complete and fluent while being
    ungrounded is the single most dangerous record this stack exists to catch, and three
    passes must not outvote that one failure. This mirrors `verify_sft.combine()`, where
    a precise contradiction fails at every similarity band.

    Low confidence downgrades a clean sheet to REVIEW but does NOT rescue a failure -
    a judge unsure of itself is not thereby evidence of correctness.
    """
    verdicts = [dims[d]['verdict'] for d in DIMENSIONS]
    if DIM_FAIL in verdicts:
        failed = [d for d in DIMENSIONS if dims[d]['verdict'] == DIM_FAIL]
        return JUDGE_FAIL, 'failed on %s' % ', '.join(failed)
    if DIM_REVIEW in verdicts:
        unsure = [d for d in DIMENSIONS if dims[d]['verdict'] == DIM_REVIEW]
        return JUDGE_REVIEW, 'undecided on %s' % ', '.join(unsure)
    if confidence == 'low':
        return JUDGE_REVIEW, 'all dimensions pass but the judge reported low confidence'
    return JUDGE_PASS, 'all four dimensions pass'


def _validate_payload(obj):
    """Return (dims, confidence) or raise ValueError describing what is malformed."""
    dims = obj.get('dimensions')
    if not isinstance(dims, dict):
        raise ValueError('"dimensions" is not an object')
    out = {}
    for d in DIMENSIONS:
        entry = dims.get(d)
        if not isinstance(entry, dict):
            raise ValueError('dimension %r missing' % d)
        v = entry.get('verdict')
        if v not in DIM_VALUES:
            raise ValueError('dimension %r has verdict %r' % (d, v))
        out[d] = {'verdict': v, 'evidence': entry.get('evidence', '')}
    conf = obj.get('confidence')
    if conf not in CONFIDENCE_VALUES:
        raise ValueError('confidence %r' % conf)
    return out, conf


def _abstention(reason, detail, usage=None):
    return {'judge_verdict': JUDGE_ABSTAIN,
            'abstain_reason': reason,
            'reason': detail,
            'dimensions': None,
            'confidence': None,
            'prompt_version': PROMPT_VERSION,
            'usage': usage or {}}


def judge_sft_record(record, chunk_text, backend, ledger=None, probe_id=None,
                     max_tokens=1024):
    """Judge one SFT record. Never raises; every failure becomes an abstention.

    Deliberately does not accept a verify_sft result - see the module docstring.
    """
    system, user = build_prompt(record, chunk_text)
    resp = None
    try:
        resp = backend.generate(system, user, max_tokens=max_tokens, temperature=0.0,
                                probe_id=probe_id)
    except JudgeUnavailable as exc:
        if ledger is not None:
            ledger.record(None, abstained=True)
        return _abstention(ABSTAIN_TRANSPORT, str(exc))
    except TypeError:
        # A backend whose generate() has no probe_id kwarg is still valid.
        try:
            resp = backend.generate(system, user, max_tokens=max_tokens,
                                    temperature=0.0)
        except JudgeUnavailable as exc:
            if ledger is not None:
                ledger.record(None, abstained=True)
            return _abstention(ABSTAIN_TRANSPORT, str(exc))

    obj, parse_reason = jc.parse_judge_json(resp.text, required_keys=REQUIRED_KEYS)
    if obj is None:
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return _abstention(parse_reason, 'could not read a verdict from the response',
                           resp.usage())

    # An explicit refusal is an abstention, not a failure of the record.
    if str(obj.get('refusal', '')).strip():
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return _abstention(ABSTAIN_REFUSED, str(obj.get('refusal')), resp.usage())

    try:
        dims, conf = _validate_payload(obj)
    except ValueError as exc:
        if ledger is not None:
            ledger.record(resp, abstained=True)
        return _abstention(jc.ABSTAIN_INCOMPLETE, str(exc), resp.usage())

    verdict, reason = combine_dimensions(dims, conf)
    if ledger is not None:
        ledger.record(resp, abstained=False)
    return {'judge_verdict': verdict,
            'reason': reason,
            'dimensions': dims,
            'confidence': conf,
            'prompt_version': PROMPT_VERSION,
            'usage': resp.usage()}


# ------------------------------------------------- combination with the rule-based layer
#
# Implements the table in src/deployment/ACCEPTANCE_CONTRACT.md, "Judge vs rule-based
# disagreement". NOT wired into verify_sft.acceptance_decision() - see the docstring.

ACCEPT = 'ACCEPT'
REJECT = 'REJECT'
HOLD_FOR_REVIEW = 'HOLD_FOR_REVIEW'


def combine_with_rules(rule_decision, judge_result):
    """Merge an acceptance_decision() outcome with a judge verdict.

    Takes the DECISION from `verify_sft.acceptance_decision()`, not the raw verdict,
    because format gating lives in that function: a grounded-but-malformed record is
    already a REJECT there, and a merge that read only `result['verdict']` would lose it.

    The asymmetry is the contract's, and it is deliberate: rules override the judge, the
    judge never overrides the rules. Not because rules are better, but because a
    rule-based failure is a checkable claim about the source while a judge failure is an
    assessment - a checkable claim can be confirmed by a human in seconds.
    """
    if rule_decision == REJECT:
        return REJECT, 'the rule-based layer rejected it; a judge cannot overturn that'
    if judge_result is None:
        return HOLD_FOR_REVIEW, 'no judge result supplied'
    jv = judge_result.get('judge_verdict')
    if rule_decision == HOLD_FOR_REVIEW:
        return (HOLD_FOR_REVIEW,
                'the rule-based layer is uncertain; a judge %s does not resolve that' % jv)
    if rule_decision != ACCEPT:
        raise ValueError('unknown rule decision %r' % rule_decision)
    if jv == JUDGE_PASS:
        return ACCEPT, 'both layers agree'
    if jv == JUDGE_FAIL:
        return (HOLD_FOR_REVIEW,
                'the judge failed a record the rules passed - probably a real defect in '
                'a region the rules cannot see, but a judge can be mistaken')
    if jv in (JUDGE_REVIEW, JUDGE_ABSTAIN):
        return HOLD_FOR_REVIEW, 'the judge did not reach a verdict (%s)' % jv
    raise ValueError('unknown judge verdict %r' % jv)


# ------------------------------------------------------------------------------ self-test

def _payload(g='pass', c='pass', f='pass', a='pass', confidence='high'):
    return json.dumps({
        'dimensions': {
            'grounding': {'verdict': g, 'evidence': 'e'},
            'completeness': {'verdict': c, 'evidence': 'e'},
            'format_correctness': {'verdict': f, 'evidence': 'e'},
            'arabic_quality': {'verdict': a, 'evidence': 'e'},
        },
        'confidence': confidence}, ensure_ascii=False)


_REC = {'instruction': 'اشرح معنى الجذر', 'response': 'شرح', 'source_chunk_id': 'x_c0001',
        'source_region': 'classical', 'format_type': 'dictionary_entry',
        'model_version': 'test'}


def run_self_test():
    ok = True

    # --- dimension folding --------------------------------------------------------------
    fold_cases = [
        (('pass', 'pass', 'pass', 'pass', 'high'), JUDGE_PASS),
        (('fail', 'pass', 'pass', 'pass', 'high'), JUDGE_FAIL),
        (('pass', 'pass', 'pass', 'fail', 'high'), JUDGE_FAIL),
        (('pass', 'review', 'pass', 'pass', 'high'), JUDGE_REVIEW),
        (('pass', 'pass', 'pass', 'pass', 'low'), JUDGE_REVIEW),
        # a failure is NOT downgraded to review by low confidence
        (('fail', 'pass', 'pass', 'pass', 'low'), JUDGE_FAIL),
        # a failure outranks a review
        (('fail', 'review', 'pass', 'pass', 'high'), JUDGE_FAIL),
    ]
    for (g, c, f, a, conf), want in fold_cases:
        dims = {d: {'verdict': v, 'evidence': ''}
                for d, v in zip(DIMENSIONS, (g, c, f, a))}
        got, _ = combine_dimensions(dims, conf)
        if got != want:
            print('  [FAIL] fold %s/%s/%s/%s conf=%s -> %s, expected %s'
                  % (g, c, f, a, conf, got, want))
            ok = False

    # A single ungrounded dimension must never be outvoted by three passes.
    dims = {d: {'verdict': 'pass', 'evidence': ''} for d in DIMENSIONS}
    dims['grounding']['verdict'] = 'fail'
    if combine_dimensions(dims, 'high')[0] != JUDGE_FAIL:
        print('  [FAIL] ungrounded record outvoted by the other three dimensions')
        ok = False

    # --- end-to-end against a scripted backend, no network ------------------------------
    be = jc.ScriptedBackend(by_key={
        'clean': _payload(),
        'ungrounded': _payload(g='fail'),
        'lowconf': _payload(confidence='low'),
        'prose': 'I think the response is fine, honestly.',      # unparseable
        'partial': '{"confidence": "high"}',                     # missing dimensions
        'badverdict': _payload(g='excellent'),                   # outside the vocabulary
        'badconf': _payload(confidence='very sure'),
        'refuses': '{"dimensions": {}, "confidence": "high", "refusal": "cannot assist"}',
        'boom': JudgeUnavailable('simulated transport failure'),
    })
    e2e = [
        ('clean', JUDGE_PASS, None),
        ('ungrounded', JUDGE_FAIL, None),
        ('lowconf', JUDGE_REVIEW, None),
        ('prose', JUDGE_ABSTAIN, jc.ABSTAIN_UNPARSEABLE),
        ('partial', JUDGE_ABSTAIN, jc.ABSTAIN_INCOMPLETE),
        ('badverdict', JUDGE_ABSTAIN, jc.ABSTAIN_INCOMPLETE),
        ('badconf', JUDGE_ABSTAIN, jc.ABSTAIN_INCOMPLETE),
        ('refuses', JUDGE_ABSTAIN, ABSTAIN_REFUSED),
        ('boom', JUDGE_ABSTAIN, ABSTAIN_TRANSPORT),
    ]
    led = jc.UsageLedger()
    for pid, want_v, want_reason in e2e:
        res = judge_sft_record(_REC, 'نص المصدر', be, ledger=led, probe_id=pid)
        if res['judge_verdict'] != want_v:
            print('  [FAIL] probe %-11s -> %s, expected %s'
                  % (pid, res['judge_verdict'], want_v))
            ok = False
        if want_reason and res.get('abstain_reason') != want_reason:
            print('  [FAIL] probe %-11s abstain_reason %s, expected %s'
                  % (pid, res.get('abstain_reason'), want_reason))
            ok = False

    # Every failure mode must abstain, and NONE of them may produce a pass.
    if led.abstentions != 6:
        print('  [FAIL] ledger counted %d abstentions, expected 6' % led.abstentions)
        ok = False
    if led.summary()['estimated_cost'] is not None:
        print('  [FAIL] cost reported without pricing')
        ok = False

    # --- the contract table, every cell -------------------------------------------------
    table = [
        (REJECT, JUDGE_PASS, REJECT),
        (REJECT, JUDGE_FAIL, REJECT),
        (REJECT, JUDGE_REVIEW, REJECT),
        (REJECT, JUDGE_ABSTAIN, REJECT),
        (ACCEPT, JUDGE_PASS, ACCEPT),
        (ACCEPT, JUDGE_FAIL, HOLD_FOR_REVIEW),
        (ACCEPT, JUDGE_REVIEW, HOLD_FOR_REVIEW),
        (ACCEPT, JUDGE_ABSTAIN, HOLD_FOR_REVIEW),
        (HOLD_FOR_REVIEW, JUDGE_PASS, HOLD_FOR_REVIEW),
        (HOLD_FOR_REVIEW, JUDGE_FAIL, HOLD_FOR_REVIEW),
        (HOLD_FOR_REVIEW, JUDGE_REVIEW, HOLD_FOR_REVIEW),
        (HOLD_FOR_REVIEW, JUDGE_ABSTAIN, HOLD_FOR_REVIEW),
    ]
    for rule, jv, want in table:
        got, _ = combine_with_rules(rule, {'judge_verdict': jv})
        if got != want:
            print('  [FAIL] contract cell rule=%s judge=%s -> %s, expected %s'
                  % (rule, jv, got, want))
            ok = False

    # The two invariants the contract turns on.
    for jv in (JUDGE_PASS, JUDGE_FAIL, JUDGE_REVIEW, JUDGE_ABSTAIN):
        if combine_with_rules(REJECT, {'judge_verdict': jv})[0] != REJECT:
            print('  [FAIL] a judge %s overturned a rule-based REJECT' % jv)
            ok = False
        if combine_with_rules(ACCEPT, {'judge_verdict': jv})[0] == REJECT:
            print('  [FAIL] a judge %s produced an outright REJECT' % jv)
            ok = False

    # A missing judge result must hold, never accept.
    if combine_with_rules(ACCEPT, None)[0] != HOLD_FOR_REVIEW:
        print('  [FAIL] a missing judge result was accepted')
        ok = False

    # --- the judge must not be reachable through a verify_sft result --------------------
    # Guards the independence property in the docstring: if someone later adds a
    # rule-verdict argument to the prompt builder, this fails.
    system, user = build_prompt(_REC, 'نص')
    for leak in ('verdict', 'FAIL_CONTRADICTED', 'percentile', 'similarity'):
        if leak in user:
            print('  [FAIL] rule-layer signal %r leaked into the judge prompt' % leak)
            ok = False

    # --- the prompt carries every dimension it asks the judge to score ------------------
    for d in DIMENSIONS:
        if d not in SYSTEM_PROMPT:
            print('  [FAIL] dimension %r is scored but never defined in the rubric' % d)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM judge for SFT records.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--show-prompt', action='store_true',
                    help='print the rubric and an example user message')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    if a.show_prompt:
        s, u = build_prompt(_REC, '<SOURCE CHUNK>')
        sys.stdout.write(s + '\n' + '-' * 70 + '\n' + u)
        sys.exit(0)
    ap.print_help()
