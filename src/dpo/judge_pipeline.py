# -*- coding: utf-8 -*-
"""LLM Judge - integration with check_dpo.py's Unified Verdict.

This is the ONLY module that decides what a judge call is allowed to do to a DPO
record's verdict. The rule, restated exactly as specified:

  AUTO_CONFIRM      -> check_dpo says the pair is fine on its own. Return as-is.
                       NO LLM CALL.
  FLAG_SUSPICIOUS    -> check_dpo says something is wrong with the PAIR itself
                       (mislabeled / rejected not actually worse / degenerate).
                       Return as-is. NO LLM CALL.
  NEEDS_JUDGE        -> the only state that spends an LLM call. On success, the judge
                       output is attached under a single top-level `judge` key.
                       The verdict can move from NEEDS_JUDGE to FLAG_SUSPICIOUS if the
                       Judge disagrees with high confidence AND nothing deterministic
                       already contradicts that (see `_maybe_escalate` below) - it can
                       NEVER move to AUTO_CONFIRM. On any judge failure (timeout,
                       rate limit, provider error, invalid JSON, contract violation),
                       the verdict stays NEEDS_JUDGE and the failure is recorded under
                       `judge`.

check_dpo.check_pair() and combine_dpo() are used exactly as they are - imported, never
copied, never reimplemented, never edited by this module.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'verification'))

import judge_provider as jc                                        # noqa: E402
import judge_contract as jcon                                    # noqa: E402
import llm_judge                                                 # noqa: E402
import check_dpo as cdpo                                         # noqa: E402


# Keep the two rejection-type vocabularies from silently drifting apart. This is a
# runtime check, not an import-time crash, so a partial/older checkout can still import
# this module for inspection; run_self_test() turns it into a hard failure.
def _contract_matches_check_dpo() -> bool:
    return set(jcon.REJECTION_TYPES) == set(cdpo.REJECTION_TYPES)


# The exact keys check_dpo.check_pair() returns that make up `deterministic_signals`.
# Copied from the spec, and every one of them is a real key check_pair() emits - see
# check_dpo.check_pair()'s return dict. Nothing here is invented.
DETERMINISTIC_SIGNAL_KEYS = (
    'direction', 'corroborated', 'detectability', 'chosen_verdict', 'rejected_verdict',
    'chosen_facts', 'rejected_facts', 'chosen_pct', 'rejected_pct', 'distinctness',
    'length_signal', 'format_check',
)

# How confidently the Judge must disagree with a NEEDS_JUDGE pair before the pipeline
# will act on that disagreement at all. Deliberately conservative: this value only ever
# moves a pair FROM NEEDS_JUDGE TO FLAG_SUSPICIOUS (i.e. "look at this one by hand"),
# never accepts anything automatically, so erring toward more human review rather than
# less is the safe direction for this threshold.
ESCALATION_CONFIDENCE = 0.75

# direction values that already show chosen ahead on the deterministic axis. Escalating
# to FLAG_SUSPICIOUS when the Judge disagrees with a signal this strong would let one
# LLM call overrule the pipeline's own measurement without any adjudication - the
# spec's "لا يوجد تعارض deterministic واضح" condition.
_CLEAR_CHOSEN_LEAD = frozenset(['chosen_better_verdict'])


def build_judge_input(record: Dict[str, Any], ctx: Any,
                      base_result: Dict[str, Any]) -> jcon.JudgeInput:
    """Build the read-only JudgeInput from a DPO record + check_pair()'s own result.

    `ctx` is the same verify_sft.VerificationContext check_dpo.check_pair() itself
    takes - source_text comes from the SAME chunk store the deterministic checks already
    used, so the Judge and the rule-based layer never see two different versions of the
    source.
    """
    chunk = ctx.chunks.get(record.get('source_chunk_id'))
    source_text = (chunk or {}).get('chunk_text', '') or ''
    signals = {k: base_result.get(k) for k in DETERMINISTIC_SIGNAL_KEYS}
    return jcon.JudgeInput(
        prompt=record.get('prompt', ''),
        chosen=record.get('chosen', ''),
        rejected=record.get('rejected', ''),
        source_chunk_id=record.get('source_chunk_id', ''),
        source_region=record.get('source_region', ''),
        rejection_type=record.get('rejection_type', ''),
        model_version=record.get('model_version', ''),
        format_type=record.get('format_type'),
        source_text=source_text,
        deterministic_signals=signals,
    )


def _maybe_escalate(base_result: Dict[str, Any],
                    judge_output: jcon.JudgeOutput) -> (str, Optional[str]):
    """Decide whether a successful judge call moves NEEDS_JUDGE -> FLAG_SUSPICIOUS.

    Returns (verdict, escalation_reason_or_None). Can only return NEEDS_JUDGE (no
    change) or FLAG_SUSPICIOUS - AUTO_CONFIRM is not a reachable value from this
    function, structurally, since it is not in the return set below.
    """
    if judge_output.assessment in ('rejected_better', 'equivalent') \
            and judge_output.confidence >= ESCALATION_CONFIDENCE \
            and base_result.get('direction') not in _CLEAR_CHOSEN_LEAD:
        reason = ('judge assessment=%r confidence=%.2f contradicts the pair as labeled, '
                  'and no deterministic signal already shows chosen clearly ahead - '
                  'escalated for human review'
                  % (judge_output.assessment, judge_output.confidence))
        return cdpo.FLAG_SUSPICIOUS, reason
    return cdpo.NEEDS_JUDGE, None


def adjudicate(record: Dict[str, Any], ctx: Any,
              client: Optional['jc.BaseJudgeClient'] = None,
              case_label: Optional[str] = None) -> Dict[str, Any]:
    """The single integration entry point requested by the spec.

    1. Runs check_dpo.check_pair() - untouched, unmodified.
    2. AUTO_CONFIRM / FLAG_SUSPICIOUS -> returned exactly as check_pair() produced them.
       No `judge` key is added, because no judge call was made.
    3. NEEDS_JUDGE -> calls the Judge (if a client was supplied) and attaches `judge`.
    4. Never sets verdict to AUTO_CONFIRM. Never edits check_dpo.combine_dpo().
    """
    base_result = cdpo.check_pair(record, ctx)

    if base_result.get('verdict') in (cdpo.AUTO_CONFIRM, cdpo.FLAG_SUSPICIOUS):
        return dict(base_result)

    result = dict(base_result)

    if base_result.get('verdict') != cdpo.NEEDS_JUDGE:
        # Defensive: check_pair() only ever returns one of the three verdicts, but if
        # that ever changes this must not silently start calling the LLM on an unknown
        # state.
        return result

    if client is None:
        result['judge'] = {
            'status': 'unavailable', 'error_type': 'NoClient',
            'reason': 'adjudicate() was called without a JudgeClient; verdict remains '
                      'NEEDS_JUDGE',
        }
        return result

    judge_input = build_judge_input(record, ctx, base_result)
    jres = llm_judge.judge_pair(judge_input, client, case_label=case_label)
    result['judge'] = jres.as_dict()

    if jres.status == 'ok':
        new_verdict, escalation_reason = _maybe_escalate(base_result, jres.judge_output)
        result['verdict'] = new_verdict
        if escalation_reason:
            result['reason'] = '%s | %s' % (result.get('reason', ''), escalation_reason)
    # else: jres.status == 'unavailable' -> result['verdict'] is already NEEDS_JUDGE,
    # inherited unchanged from base_result. Nothing else to do.

    return result


def adjudicate_batch(records, ctx, client=None):
    """Convenience for the CLI/dashboard: adjudicate() over a list of records."""
    return [adjudicate(r, ctx, client=client) for r in records]


# ------------------------------------------------------------------------- self-test

class _FakeCtx(object):
    """Minimal stand-in that carries only what build_judge_input() reads
    (ctx.chunks). Used ONLY to test build_judge_input() in isolation; the full
    integration self-tests below build a real verify_sft.VerificationContext against
    tiny synthetic fixtures (see tests/fixtures under this module's --self-test run,
    invoked from src/dpo/test_llm_judge.py which owns the shared fixtures).
    """
    def __init__(self, chunks):
        self.chunks = chunks


def run_self_test():
    ok = True

    if not _contract_matches_check_dpo():
        print('  [FAIL] judge_contract.REJECTION_TYPES has drifted from '
              'check_dpo.REJECTION_TYPES: %s vs %s'
              % (sorted(jcon.REJECTION_TYPES), sorted(cdpo.REJECTION_TYPES)))
        ok = False

    # --- build_judge_input pulls exactly the documented signal keys, nothing invented
    record = {'prompt': 'p', 'chosen': 'c', 'rejected': 'r',
             'source_chunk_id': 'x1', 'source_region': 'classical',
             'rejection_type': 'verbosity', 'model_version': 'v0',
             'format_type': 'dictionary_entry'}
    base_result = {'verdict': cdpo.NEEDS_JUDGE, 'direction': 'tie', 'corroborated': True,
                   'detectability': 'strong', 'chosen_verdict': 'PASS',
                   'rejected_verdict': 'PASS', 'chosen_facts': 'SUPPORTED',
                   'rejected_facts': 'SUPPORTED', 'chosen_pct': 80.0, 'rejected_pct': 78.0,
                   'distinctness': {'lexical': 0.5, 'degenerate': False},
                   'length_signal': 'rejected_longer',
                   'format_check': {'relation': 'both_match'},
                   'reason': 'tie but corroborated'}
    ctx = _FakeCtx({'x1': {'chunk_text': 'نص المصدر هنا.'}})
    ji = build_judge_input(record, ctx, base_result)
    if ji.source_text != 'نص المصدر هنا.':
        print('  [FAIL] build_judge_input did not read source_text from ctx.chunks')
        ok = False
    if set(ji.deterministic_signals) != set(DETERMINISTIC_SIGNAL_KEYS):
        print('  [FAIL] deterministic_signals keys drifted from the documented set: %s'
              % sorted(ji.deterministic_signals))
        ok = False

    # --- unknown source_chunk_id -> empty source_text, never a crash
    ji2 = build_judge_input(record, _FakeCtx({}), base_result)
    if ji2.source_text != '':
        print('  [FAIL] missing chunk should yield empty source_text, not a crash')
        ok = False

    # --- _maybe_escalate: rejected_better + high confidence + no clear chosen lead
    v, reason = _maybe_escalate({'direction': 'tie'},
                                jcon.JudgeOutput(
                                    schema_version='judge-1',
                                    judge_model_version='m', assessment='rejected_better',
                                    weakness_present=True, rejection_type=None,
                                    rejection_type_agrees_with_record=None,
                                    type_scores={}, confidence=0.9, evidence=[]))
    if v != cdpo.FLAG_SUSPICIOUS or not reason:
        print('  [FAIL] high-confidence disagreement on a tie should escalate, got %s'
              % v)
        ok = False

    # --- _maybe_escalate: same disagreement, but deterministic already favors chosen
    # clearly -> must NOT escalate (spec's "no clear deterministic conflict" clause)
    v2, r2 = _maybe_escalate({'direction': 'chosen_better_verdict'},
                             jcon.JudgeOutput(
                                 schema_version='judge-1', judge_model_version='m',
                                 assessment='rejected_better', weakness_present=True,
                                 rejection_type=None,
                                 rejection_type_agrees_with_record=None,
                                 type_scores={}, confidence=0.99, evidence=[]))
    if v2 != cdpo.NEEDS_JUDGE:
        print('  [FAIL] disagreement against a CLEAR deterministic chosen-lead must '
              'not escalate, got %s' % v2)
        ok = False

    # --- _maybe_escalate: low confidence must not escalate
    v3, _ = _maybe_escalate({'direction': 'tie'},
                            jcon.JudgeOutput(
                                schema_version='judge-1', judge_model_version='m',
                                assessment='rejected_better', weakness_present=True,
                                rejection_type=None,
                                rejection_type_agrees_with_record=None,
                                type_scores={}, confidence=0.4, evidence=[]))
    if v3 != cdpo.NEEDS_JUDGE:
        print('  [FAIL] low-confidence disagreement must not escalate, got %s' % v3)
        ok = False

    # --- _maybe_escalate: chosen_better assessment must never escalate regardless of
    # confidence (it AGREES with the pair, nothing suspicious to flag)
    v4, _ = _maybe_escalate({'direction': 'tie'},
                            jcon.JudgeOutput(
                                schema_version='judge-1', judge_model_version='m',
                                assessment='chosen_better', weakness_present=True,
                                rejection_type='verbosity',
                                rejection_type_agrees_with_record=True,
                                type_scores={}, confidence=0.99, evidence=[]))
    if v4 != cdpo.NEEDS_JUDGE:
        print('  [FAIL] chosen_better agreement must never escalate, got %s' % v4)
        ok = False

    # --- structural guarantee: _maybe_escalate cannot return AUTO_CONFIRM under any
    # input, by construction (grep the source, but also assert on a few edge cases)
    for assessment in ('chosen_better', 'rejected_better', 'equivalent', 'undetermined'):
        for conf in (0.0, 0.5, 0.75, 1.0):
            for direction in ('tie', 'chosen_better_verdict', 'chosen_better_similarity',
                              None):
                v5, _ = _maybe_escalate(
                    {'direction': direction},
                    jcon.JudgeOutput(schema_version='judge-1', judge_model_version='m',
                                     assessment=assessment, weakness_present=False,
                                     rejection_type=None,
                                     rejection_type_agrees_with_record=None,
                                     type_scores={}, confidence=conf, evidence=[]))
                if v5 == cdpo.AUTO_CONFIRM:
                    print('  [FAIL] _maybe_escalate produced AUTO_CONFIRM - must be '
                          'structurally impossible'); ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM Judge <-> check_dpo integration.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
