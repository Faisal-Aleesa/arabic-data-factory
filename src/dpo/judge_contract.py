# -*- coding: utf-8 -*-
"""LLM Judge - input/output contracts.

This module defines the data shapes the LLM Judge consumes and produces, and the
validation that keeps a judge call's output trustworthy before it ever reaches
judge_pipeline.py. It contains NO model call and NO IO - pure dataclasses and pure
validation functions, exactly like check_dpo.combine_dpo() is pure.

Why this exists as its own module
----------------------------------
check_dpo.py already defines the deterministic Unified Verdict (AUTO_CONFIRM /
NEEDS_JUDGE / FLAG_SUSPICIOUS) and check_pair() that produces it. This module does NOT
redefine or touch that. It defines a *second*, independent contract: what goes IN to an
LLM call (JudgeInput) and what is an acceptable answer coming OUT of it (JudgeOutput).
The two contracts are deliberately disjoint in vocabulary - JudgeOutput has no `verdict`
field and cannot express AUTO_CONFIRM - so a judge call can never impersonate the
Unified Verdict.

Field provenance (do not invent fields)
----------------------------------------
JudgeInput.deterministic_signals is filled from check_dpo.check_pair()'s own return
dict - see judge_pipeline.build_judge_input(). The keys used
(direction, corroborated, detectability, chosen_verdict, rejected_verdict, chosen_facts,
rejected_facts, chosen_pct, rejected_pct, distinctness, length_signal, format_check) are
exactly the keys check_pair() already returns; nothing here invents a field that is not
already produced by the existing deterministic layer. `dataset_stated_rejection_type` is
NOT used anywhere - the field is `rejection_type`, matching check_dpo.DPO_FIELDS.

KNOWN LIMITATION: `rejection_type_declared` is a leading question
-----------------------------------------------------------------
`JudgeInput.to_prompt_payload()` sends `rejection_type_declared`, and JudgeOutput asks
the model for `rejection_type_agrees_with_record`. So the Judge is told the answer the
record expects before it is asked whether that answer is right.

**Agreement therefore corroborates less than it appears to.** A model that agrees may be
confirming the label rather than reading the pair - the two are indistinguishable from
the output. `rejection_type_agrees_with_record: true` should be read as "not contradicted"
rather than as independent confirmation, and it must not be treated as a second opinion
in any future weighting.

قيد معروف: إبلاغ الحَكَم بالنوع المُعلن سؤال موجِّه، فالموافقة تُقرأ على أنها "لم
يُخالَف"، لا على أنها تأكيد مستقل.

What this does NOT threaten
    The AUTO_CONFIRM safety floor is unaffected, and not because of anything here:
    ASSESSMENTS has no AUTO_CONFIRM value for a judge to set, and
    judge_pipeline._maybe_escalate() can only return NEEDS_JUDGE or FLAG_SUSPICIOUS.
    That floor is structural and holds however biased the judge is.

What it DOES cost
    Escalation may UNDER-detect genuine mislabels. A pair whose `rejection_type` is
    simply wrong is one the judge has been primed to accept, so the disagreement that
    would have escalated it to FLAG_SUSPICIOUS is less likely to occur. The failure is
    silent: it looks like agreement, and nothing downstream re-checks the label.

FOLLOW-UP, not a blocker for this merge
    Switch to independent detection: withhold `rejection_type_declared` from the payload,
    have the model name the weakness it actually observes, and let the CODE compare that
    against the record. That is the approach `src/verification/judge_dpo.py` took - it
    never passes the declared type and maps type -> dimension in
    `CORROBORATING_DIMENSION` instead, so corroboration is a fact about the judge's own
    independent scoring rather than about the hint it was given. The same discipline is
    already applied here for position: `llm_judge._blank_signals()` withholds the
    side-labelled deterministic signals precisely so the swap check cannot be gamed by a
    model reading which side is which.

Why literal-quote verification matters
---------------------------------------
The Judge is NLU-only: it must not paraphrase evidence into existence. A "quote" that
is not a literal substring of the text it claims to come from is worse than no evidence
at all, because it looks like grounding and is not - the same failure mode
check_facts.py's CONTRADICTED verdict exists to catch on the deterministic side. So
evidence validation here is byte-for-byte substring containment, not fuzzy matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 'judge-1'

# The nine charter rejection types - the ONLY values judge_pipeline / check_dpo accept.
# Single source of truth would ideally be check_dpo.REJECTION_TYPES; duplicated here as a
# plain tuple (not imported) so this module has zero dependency on check_dpo and can be
# unit-tested in total isolation. judge_pipeline.py asserts the two stay in sync.
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

ASSESSMENTS = ('chosen_better', 'rejected_better', 'equivalent', 'undetermined')
QUOTE_SOURCES = ('rejected', 'chosen', 'source')

# Field names the Judge is never allowed to emit. Any of these appearing anywhere in the
# top-level output is an automatic InvalidJudgeOutput - they are reserved for the
# Unified Verdict (check_dpo.py) and letting a judge write them would let an LLM call
# impersonate the deterministic layer.
FORBIDDEN_TOP_LEVEL_FIELDS = frozenset([
    'verdict', 'judge_verdict', 'AUTO_CONFIRM', 'auto_confirm',
    'NEEDS_JUDGE', 'needs_judge', 'FLAG_SUSPICIOUS', 'flag_suspicious',
])

REQUIRED_TOP_LEVEL_FIELDS = (
    'schema_version', 'judge_model_version', 'assessment', 'weakness_present',
    'rejection_type', 'rejection_type_agrees_with_record', 'type_scores',
    'confidence', 'evidence',
)


# --------------------------------------------------------------------------- JudgeInput

@dataclass(frozen=True)
class JudgeInput:
    """Everything sent to the LLM. Read-only once built.

    deterministic_signals is a plain dict copied verbatim from check_dpo.check_pair(),
    never hand-authored - see judge_pipeline.build_judge_input(). It is documentation-only
    context for the model; the Judge cannot write back into it and judge_pipeline never
    trusts anything the Judge says about it over the dict itself.
    """
    prompt: str
    chosen: str
    rejected: str
    source_chunk_id: str
    source_region: str
    rejection_type: str
    model_version: str
    source_text: str
    deterministic_signals: Dict[str, Any]
    format_type: Optional[str] = None

    def validate(self) -> List[str]:
        """Structural sanity check on the input itself. Returns a list of problems."""
        errors = []
        for f in ('prompt', 'chosen', 'rejected', 'source_chunk_id', 'source_region',
                  'model_version'):
            if not getattr(self, f):
                errors.append('JudgeInput.%s is empty' % f)
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            errors.append('JudgeInput.source_text is empty - the Judge cannot ground '
                          'anything without the actual source passage')
        if self.rejection_type not in REJECTION_TYPES:
            errors.append('JudgeInput.rejection_type %r is not one of the nine charter '
                          'types' % self.rejection_type)
        if not isinstance(self.deterministic_signals, dict):
            errors.append('JudgeInput.deterministic_signals must be a dict')
        return errors

    def to_prompt_payload(self) -> Dict[str, Any]:
        """The exact, minimal dict handed to the model. No extra fields, no answer key."""
        return {
            'prompt': self.prompt,
            'chosen': self.chosen,
            'rejected': self.rejected,
            'source_chunk_id': self.source_chunk_id,
            'source_region': self.source_region,
            'rejection_type_declared': self.rejection_type,
            'format_type': self.format_type,
            'source_text': self.source_text,
            'deterministic_signals': self.deterministic_signals,
        }


# -------------------------------------------------------------------------- JudgeOutput

@dataclass(frozen=True)
class EvidenceItem:
    claim: str
    quote_from: str          # 'rejected' | 'chosen' | 'source'
    quote: str
    explains_type: Optional[str] = None


@dataclass(frozen=True)
class JudgeOutput:
    schema_version: str
    judge_model_version: str
    assessment: str
    weakness_present: bool
    rejection_type: Optional[str]
    rejection_type_agrees_with_record: Optional[bool]
    type_scores: Dict[str, float]
    confidence: float
    evidence: List[EvidenceItem] = field(default_factory=list)
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class InvalidJudgeOutput(ValueError):
    """The raw dict from the LLM failed contract validation.

    Distinct from judge_provider's transport-level errors (TimeoutError, RateLimited,
    ProviderError, InvalidJSON) - this is a semantic failure of an otherwise
    well-formed JSON reply, e.g. a hallucinated quote or a disallowed rejection_type.
    judge_pipeline treats this exactly like every other judge failure: verdict stays
    NEEDS_JUDGE, nothing is promoted.
    """


def _text_for_source(judge_input: JudgeInput, quote_from: str) -> Optional[str]:
    if quote_from == 'chosen':
        return judge_input.chosen
    if quote_from == 'rejected':
        return judge_input.rejected
    if quote_from == 'source':
        return judge_input.source_text
    return None


def validate_judge_output(raw: Any, judge_input: JudgeInput) -> JudgeOutput:
    """Validate a raw (already-JSON-parsed) dict against the contract.

    Raises InvalidJudgeOutput with a precise reason on any violation. Returns a frozen
    JudgeOutput only when every rule passes. This is the single gate judge_pipeline.py
    calls before a judge result is allowed to touch anything.
    """
    if not isinstance(raw, dict):
        raise InvalidJudgeOutput('judge output is not a JSON object: %r' % type(raw))

    present_keys = set(raw.keys())
    forbidden_hit = present_keys & FORBIDDEN_TOP_LEVEL_FIELDS
    if forbidden_hit:
        raise InvalidJudgeOutput(
            'judge output contains reserved field(s) %s - the Judge cannot write the '
            'Unified Verdict' % sorted(forbidden_hit))

    missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in raw]
    if missing:
        raise InvalidJudgeOutput('judge output missing required field(s): %s'
                                 % ', '.join(missing))

    if raw['schema_version'] != SCHEMA_VERSION:
        raise InvalidJudgeOutput('unexpected schema_version %r (expected %r)'
                                 % (raw['schema_version'], SCHEMA_VERSION))

    if raw['assessment'] not in ASSESSMENTS:
        raise InvalidJudgeOutput('assessment %r is not one of %s - this is also the '
                                 'gate that blocks an LLM-invented verdict string like '
                                 'AUTO_CONFIRM from ever validating'
                                 % (raw['assessment'], ASSESSMENTS))

    if not isinstance(raw['weakness_present'], bool):
        raise InvalidJudgeOutput('weakness_present must be a bool')

    rtype = raw['rejection_type']
    if rtype is not None and rtype not in REJECTION_TYPES:
        raise InvalidJudgeOutput('rejection_type %r is not one of the nine charter '
                                 'types or null' % rtype)

    agrees = raw['rejection_type_agrees_with_record']
    if agrees is not None and not isinstance(agrees, bool):
        raise InvalidJudgeOutput('rejection_type_agrees_with_record must be a bool or '
                                 'null')

    type_scores = raw['type_scores']
    if not isinstance(type_scores, dict):
        raise InvalidJudgeOutput('type_scores must be an object')
    for k, v in type_scores.items():
        if k not in REJECTION_TYPES:
            raise InvalidJudgeOutput('type_scores has an unknown type key %r' % k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
            raise InvalidJudgeOutput('type_scores[%r] = %r is not a number in [0, 1]'
                                     % (k, v))

    confidence = raw['confidence']
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or not (0.0 <= confidence <= 1.0):
        raise InvalidJudgeOutput('confidence must be a number in [0, 1], got %r'
                                 % confidence)

    evidence_raw = raw['evidence']
    if not isinstance(evidence_raw, list):
        raise InvalidJudgeOutput('evidence must be a list')

    evidence: List[EvidenceItem] = []
    for i, e in enumerate(evidence_raw):
        if not isinstance(e, dict):
            raise InvalidJudgeOutput('evidence[%d] is not an object' % i)
        for f in ('claim', 'quote_from', 'quote'):
            if not e.get(f):
                raise InvalidJudgeOutput('evidence[%d] missing/empty field %r' % (i, f))
        if e['quote_from'] not in QUOTE_SOURCES:
            raise InvalidJudgeOutput('evidence[%d].quote_from %r not in %s'
                                     % (i, e['quote_from'], QUOTE_SOURCES))
        source_text = _text_for_source(judge_input, e['quote_from'])
        if source_text is None or e['quote'] not in source_text:
            raise InvalidJudgeOutput(
                'evidence[%d].quote is not a literal substring of %s - hallucinated or '
                'paraphrased evidence is rejected outright (quote=%r)'
                % (i, e['quote_from'], e['quote'][:80]))
        explains = e.get('explains_type')
        if explains is not None and explains not in REJECTION_TYPES:
            raise InvalidJudgeOutput('evidence[%d].explains_type %r is not one of the '
                                     'nine charter types' % (i, explains))
        evidence.append(EvidenceItem(claim=e['claim'], quote_from=e['quote_from'],
                                     quote=e['quote'], explains_type=explains))

    if raw['weakness_present'] and not evidence:
        raise InvalidJudgeOutput('weakness_present=true but evidence is empty - a '
                                 'claimed weakness must be backed by at least one '
                                 'literal quote')

    notes = raw.get('notes')
    if notes is not None and not isinstance(notes, str):
        raise InvalidJudgeOutput('notes must be a string or null')

    return JudgeOutput(
        schema_version=raw['schema_version'],
        judge_model_version=raw['judge_model_version'],
        assessment=raw['assessment'],
        weakness_present=raw['weakness_present'],
        rejection_type=rtype,
        rejection_type_agrees_with_record=agrees,
        type_scores=dict(type_scores),
        confidence=float(confidence),
        evidence=evidence,
        notes=notes,
    )


def parse_and_validate(raw_text: str, judge_input: JudgeInput) -> JudgeOutput:
    """Convenience: JSON-decode then validate. Raises InvalidJSON-compatible ValueError
    on decode failure (caller in llm_judge.py maps this to judge_provider.InvalidJSON),
    or InvalidJudgeOutput on a semantically bad-but-parseable reply.
    """
    try:
        raw = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('judge reply is not valid JSON: %s' % exc) from exc
    return validate_judge_output(raw, judge_input)


# ------------------------------------------------------------------------- self-test

def _R(**kw):
    """Minimal valid raw dict builder for tests."""
    base = {
        'schema_version': SCHEMA_VERSION,
        'judge_model_version': 'test-provider/test-model',
        'assessment': 'chosen_better',
        'weakness_present': False,
        'rejection_type': None,
        'rejection_type_agrees_with_record': None,
        'type_scores': {},
        'confidence': 0.5,
        'evidence': [],
        'notes': None,
    }
    base.update(kw)
    return base


def run_self_test():
    ok = True

    ji = JudgeInput(
        prompt='اشرح مضمون هذه المادة.',
        chosen='تعرض هذه المادة معاني الجذر (أتى).',
        rejected='قال النابعة في ذلك بيتا.',
        source_chunk_id='asas_albalagha_c0001',
        source_region='classical',
        rejection_type='partial_factual_errors',
        model_version='synthetic-v0',
        format_type='dictionary_entry',
        source_text='تعرض هذه المادة معاني الجذر (أتى)، وقال جابر بن حني التغلبي بيتا.',
        deterministic_signals={'direction': 'tie', 'corroborated': None},
    )
    if ji.validate():
        print('  [FAIL] a well-formed JudgeInput reported errors: %s' % ji.validate())
        ok = False

    bad_ji = JudgeInput(prompt='', chosen='x', rejected='y', source_chunk_id='c',
                        source_region='classical', rejection_type='not_a_real_type',
                        model_version='v', source_text='', deterministic_signals={})
    errs = bad_ji.validate()
    if len(errs) < 3:
        print('  [FAIL] a broken JudgeInput should report multiple errors, got %s' % errs)
        ok = False

    # --- valid minimal output ---
    try:
        out = validate_judge_output(_R(), ji)
        if out.assessment != 'chosen_better':
            print('  [FAIL] valid output round-trip lost assessment'); ok = False
    except InvalidJudgeOutput as e:
        print('  [FAIL] a valid minimal output was rejected: %s' % e); ok = False

    # --- valid output WITH evidence ---
    good_evidence = _R(weakness_present=True, rejection_type='partial_factual_errors',
                       rejection_type_agrees_with_record=True,
                       type_scores={'partial_factual_errors': 0.9},
                       confidence=0.85,
                       evidence=[{'claim': 'rejected corrupts the cited poet name',
                                 'quote_from': 'rejected', 'quote': 'النابعة',
                                 'explains_type': 'partial_factual_errors'}])
    try:
        out = validate_judge_output(good_evidence, ji)
        if len(out.evidence) != 1:
            print('  [FAIL] evidence not round-tripped'); ok = False
    except InvalidJudgeOutput as e:
        print('  [FAIL] valid evidence-bearing output rejected: %s' % e); ok = False

    # --- rejection cases ---
    cases = [
        ('unknown assessment', _R(assessment='AUTO_CONFIRM')),
        ('bad rejection_type', _R(rejection_type='hallucination')),
        ('confidence out of range', _R(confidence=1.5)),
        ('confidence not numeric', _R(confidence='high')),
        ('weakness without evidence', _R(weakness_present=True,
                                         rejection_type='verbosity')),
        ('forbidden field verdict', {**_R(), 'verdict': 'AUTO_CONFIRM'}),
        ('forbidden field judge_verdict', {**_R(), 'judge_verdict': 'chosen_better'}),
        ('missing schema_version', {k: v for k, v in _R().items()
                                    if k != 'schema_version'}),
        ('type_scores bad key', _R(type_scores={'made_up_type': 0.5})),
        ('type_scores out of range', _R(type_scores={'verbosity': 1.4})),
        ('not a dict', 'chosen_better'),
        ('evidence not a list', _R(evidence='trust me')),
        ('evidence quote_from invalid', _R(
            weakness_present=True, rejection_type='verbosity',
            evidence=[{'claim': 'x', 'quote_from': 'prompt', 'quote': 'y'}])),
    ]
    for label, raw in cases:
        try:
            validate_judge_output(raw, ji)
            print('  [FAIL] case %r should have raised InvalidJudgeOutput' % label)
            ok = False
        except InvalidJudgeOutput:
            pass

    # --- hallucinated quote: not a literal substring of the claimed source ---
    halluc = _R(weakness_present=True, rejection_type='less_faithful_reconstruction',
               evidence=[{'claim': 'rejected invents a detail', 'quote_from': 'source',
                         'quote': 'جملة لا وجود لها إطلاقا في هذا المصدر'}])
    try:
        validate_judge_output(halluc, ji)
        print('  [FAIL] hallucinated quote should have been rejected'); ok = False
    except InvalidJudgeOutput:
        pass

    # --- quote IS literal but from the wrong side ---
    wrong_side = _R(weakness_present=True, rejection_type='partial_factual_errors',
                    evidence=[{'claim': 'x', 'quote_from': 'chosen',
                              # this string is only in `rejected`, not `chosen`
                              'quote': 'النابعة'}])
    try:
        validate_judge_output(wrong_side, ji)
        print('  [FAIL] a quote literal in rejected but claimed from chosen should fail')
        ok = False
    except InvalidJudgeOutput:
        pass

    # --- parse_and_validate: invalid JSON surfaces as ValueError, not InvalidJudgeOutput
    try:
        parse_and_validate('{not json', ji)
        print('  [FAIL] malformed JSON should raise'); ok = False
    except InvalidJudgeOutput:
        print('  [FAIL] malformed JSON should be a JSON error, not a contract error')
        ok = False
    except ValueError:
        pass

    try:
        out2 = parse_and_validate(json.dumps(_R()), ji)
        if not isinstance(out2, JudgeOutput):
            print('  [FAIL] parse_and_validate did not return a JudgeOutput'); ok = False
    except Exception as e:
        print('  [FAIL] valid JSON text should validate cleanly: %s' % e); ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import sys
    import argparse
    ap = argparse.ArgumentParser(description='LLM Judge input/output contracts.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
