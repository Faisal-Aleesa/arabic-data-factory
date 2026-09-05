# -*- coding: utf-8 -*-
"""LLM Judge - test suite (project's --self-test convention, not pytest).

Two layers, mirroring how check_similarity.py documents its own two-layer self-test:

  A. Contract/judge layer (no corpus needed): every one of the nine rejection_type
     values is exercised directly against llm_judge.judge_pair() with a
     judge_provider.RecordedJudgeClient - no network, no API key, ever. This is where
     every required failure mode lives: hallucinated quote, invalid JSON, timeout,
     rate limiting, provider error, a Judge trying to smuggle AUTO_CONFIRM, and the
     similarity trap.

  B. Pipeline/integration layer: judge_pipeline.adjudicate() is run over REAL
     check_dpo.check_pair() results, computed from real (not stubbed) deterministic
     checks against a tiny synthetic classical_lexicon fixture built the same way the
     real corpus is built (see tests/fixtures/*.jsonl, generated with
     extract_facts.extract_classical_lexicon() - not hand-authored fact tables). This
     is where AUTO_CONFIRM-skips-the-call and FLAG_SUSPICIOUS-skips-the-call are
     verified against the actual Unified Verdict logic, not a mock of it.

Honest limitation, stated once here rather than buried in a comment somewhere: this
sandbox has no sentence-transformers model available (see check_similarity.py's own
self-test, which reports "[skip] sentence-transformers not installed"), so every
VerificationContext built in this file uses embedder=None. That makes the similarity
axis unavailable throughout layer B, which is a real, pre-existing constraint of the
verification stack in this environment - not something judge_pipeline.py works around.
Concretely this means `less_faithful_reconstruction`, whose primary corroboration axis
is similarity, resolves to FLAG_SUSPICIOUS rather than NEEDS_JUDGE in the layer-B
fixture (its facts stay SUPPORTED on both sides with no embedder to separate them) -
that specific type's NEEDS_JUDGE path is therefore proven at layer A (contract/judge)
instead. The report below states this plainly against the affected case rather than
hiding it.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'verification'))

import judge_provider as jc                                        # noqa: E402
import judge_contract as jcon                                    # noqa: E402
import llm_judge                                                 # noqa: E402
import judge_pipeline as jp                                      # noqa: E402
import check_dpo as cdpo                                         # noqa: E402
import verify_sft as vs                                          # noqa: E402

FIXTURES_DIR = os.path.join(_HERE, '..', '..', 'tests', 'fixtures')
CHUNKS_PATH = os.path.join(FIXTURES_DIR, 'chunks_classical_lexicon.jsonl')
FACTS_PATH = os.path.join(FIXTURES_DIR, 'facts_classical_lexicon.jsonl')

REPORT = []          # (case, expected, actual, passed) - printed as a table at the end


def _record_case(case, expected, actual, passed):
    REPORT.append((case, expected, actual, passed))
    return passed


# ============================================================== fixtures (layer B)

CHUNK_ID = 'atc_c0001'
REGION = 'classical'
CHOSEN = ('أتى: أتى إليه إحسانا أي فعله، وأتى عليهم الدهر بمعنى أفناهم، '
         'وتذكر الإتاوة وهي الخراج والجباية، وقال جابر بن حني التغلبي في ذلك بيتا.')
CITE = 'وقال جابر بن حني التغلبي في ذلك بيتا.'

# rejected candidates, one per charter rejection_type, built to exercise the REAL
# deterministic layer (not synthetic verdict dicts) - see the module docstring for why
# each one lands where it does.
REJECTED_BY_TYPE = {
    'poor_instruction_following': (
        'المعاجم البلاغية القديمة تمثل ذخيرة لغوية مهمة، وقد اعتنى بها العلماء قديما '
        'وحديثا، وهي مادة خصبة للدارسين في أقسام اللغة العربية. ' + CITE),
    'wrong_formatting': (
        'يذكر الكاتب هنا شرحا لهذا الجذر ومعانيه المختلفة كإفناء الدهر وأخذ الجباية '
        'باسم الإتاوة، ' + CITE),
    'missing_information': 'أتى: أتى إليه إحسانا أي فعله. ' + CITE,
    'unsupported_additions': (
        CHOSEN + ' وذكر أيضا (زقزق) بمعنى الصياح، وقال سعدان البصري فيها بيتا.'),
    'wrong_register': (
        'يعني هالجذر أتى، لما نقول أتى عليهم الدهر يعني خلصهم كلهم. والإتاوة هي '
        'الفلوس اللي تتاخذ من الناس. ' + CITE),
    'verbosity': (
        CHOSEN + ' وهذا من المعاني المشهورة المتداولة المعروفة عند أهل اللغة قاطبة، '
        'وقد تكلم فيه العلماء كلاما كثيرا مستفيضا مطولا، وهو باب واسع كبير عظيم '
        'يحتاج إلى تأمل ونظر.'),
    'weak_organization': (
        'وقال جابر بن حني التغلبي في ذلك بيتا. وهي الخراج والجباية، وتذكر الإتاوة. '
        'بمعنى أفناهم وأتى عليهم الدهر، أي فعله فمنها أتى إليه إحسانا. أتى:'),
    'partial_factual_errors': (
        'أتى: أتى إليه إحسانا أي فعله، وأتى عليهم الدهر بمعنى أفناهم، وتذكر الإتاوة '
        'وهي الخراج والجباية، وقال جابر بن حني التغلبيي في ذلك بيتا.'),
    'less_faithful_reconstruction': (
        'تبين هذه المادة ثراء اللغة العربية وقدرتها على التعبير عن المعاني الدقيقة '
        'في أقسام اللغة العربية. ' + CITE),
}

# The verdict each candidate REALLY reaches through check_dpo.check_pair() in this
# fixture, measured (see module docstring on the similarity-axis limitation).
EXPECTED_VERDICT = {
    'poor_instruction_following': cdpo.NEEDS_JUDGE,
    'wrong_formatting': cdpo.AUTO_CONFIRM,
    'missing_information': cdpo.NEEDS_JUDGE,
    'unsupported_additions': cdpo.NEEDS_JUDGE,
    'wrong_register': cdpo.NEEDS_JUDGE,
    'verbosity': cdpo.NEEDS_JUDGE,
    'weak_organization': cdpo.NEEDS_JUDGE,
    'partial_factual_errors': cdpo.AUTO_CONFIRM,
    'less_faithful_reconstruction': cdpo.FLAG_SUSPICIOUS,
}

UNRELATED_TO_SOURCE = (
    'تشرح هذه المادة أحكام العروض والقافية وأنواع البحور الشعرية، مع أمثلة على '
    'الزحاف والعلة في بحر الطويل.')


def _make_record(rejection_type, rejected=None):
    return {
        'prompt': 'اشرح مضمون هذه المادة المعجمية.',
        'chosen': CHOSEN,
        'rejected': rejected if rejected is not None else REJECTED_BY_TYPE[rejection_type],
        'source_chunk_id': CHUNK_ID,
        'source_region': REGION,
        'rejection_type': rejection_type,
        'model_version': 'synthetic-v0',
        'format_type': 'dictionary_entry',
    }


def _make_ctx():
    return vs.VerificationContext(
        corpus='classical_lexicon',
        facts_path=FACTS_PATH,
        chunks_path=CHUNKS_PATH,
        embedder=None,          # no sentence-transformers in this sandbox - see docstring
        baseline_n=3,
        seed=0,
    )


def _judge_input_for(rejection_type):
    """A layer-A JudgeInput independent of check_pair() - used for pure contract/judge
    coverage of every type, including the two whose pipeline verdict in this fixture
    isn't NEEDS_JUDGE (see docstring)."""
    return jcon.JudgeInput(
        prompt='اشرح مضمون هذه المادة المعجمية.',
        chosen=CHOSEN,
        rejected=REJECTED_BY_TYPE[rejection_type],
        source_chunk_id=CHUNK_ID,
        source_region=REGION,
        rejection_type=rejection_type,
        model_version='synthetic-v0',
        format_type='dictionary_entry',
        source_text=CHOSEN,
        deterministic_signals={'direction': 'tie', 'corroborated': None,
                               'detectability': cdpo.DETECTABILITY.get(rejection_type),
                               'chosen_pct': 88.0, 'rejected_pct': 86.5},
    )


# --------------------------------------------------- position-bias swap check helpers
#
# llm_judge.judge_pair() judges the SWAP_CHECK_TYPES twice, with the two responses in
# swapped positions, and refuses if the verdict follows the slot. A recorded client
# therefore needs an answer for both passes. These helpers keep every scenario below
# reading the way it did before the check was ported in.

def _mirror(output):
    """The same recorded output as it would read with chosen/rejected swapped."""
    flip = {'chosen_better': 'rejected_better', 'rejected_better': 'chosen_better'}
    out = dict(output)
    out['assessment'] = flip.get(output.get('assessment'), output.get('assessment'))
    return out


def _register_with_swap(client, label, output):
    """Register a recording plus its mirror, so a swap-checked type gets a consistent
    (non-position-biased) answer on both passes. Harmless for the other seven types -
    the `_swap` key is simply never looked up."""
    client.register(label, output)
    client.register(label + '_swap', _mirror(output))


def _expected_calls(rejection_type):
    return 2 if rejection_type in llm_judge.SWAP_CHECK_TYPES else 1


# ==================================================================== layer A tests

def test_contract_all_nine_types():
    ok = True
    client = jc.RecordedJudgeClient()
    for rtype in jcon.REJECTION_TYPES:
        ji = _judge_input_for(rtype)
        good = {
            'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
            'assessment': 'chosen_better', 'weakness_present': True,
            'rejection_type': rtype, 'rejection_type_agrees_with_record': True,
            'type_scores': {rtype: 0.8}, 'confidence': 0.8,
            'evidence': [{'claim': 'grounded in the source',
                         'quote_from': 'source', 'quote': 'جابر بن حني التغلبي',
                         'explains_type': rtype}],
            'notes': None,
        }
        _register_with_swap(client, rtype, good)
        res = llm_judge.judge_pair(ji, client, case_label=rtype)
        passed = (res.status == 'ok' and res.judge_output is not None
                  and res.judge_output.rejection_type == rtype)
        ok &= _record_case('layer_A: contract round-trip for type %r' % rtype,
                           'status=ok, rejection_type=%r' % rtype,
                           'status=%s, rejection_type=%r'
                           % (res.status, res.judge_output.rejection_type
                              if res.judge_output else None),
                           passed)
    return ok


def test_similarity_trap():
    """chosen is factually correct; rejected is near-identical in wording but has a
    small factual corruption. deterministic_signals report a HIGH percentile for BOTH
    sides (simulating an embedder that would call them equally similar to the source).
    A correct judge output must still prefer chosen on factual grounds - and the
    pipeline/contract layer must not second-guess that using the percentages.
    """
    ji = jcon.JudgeInput(
        prompt='اشرح مضمون هذه المادة المعجمية.',
        chosen=CHOSEN,
        rejected=REJECTED_BY_TYPE['partial_factual_errors'],
        source_chunk_id=CHUNK_ID, source_region=REGION,
        rejection_type='partial_factual_errors', model_version='synthetic-v0',
        format_type='dictionary_entry', source_text=CHOSEN,
        deterministic_signals={'direction': 'tie', 'corroborated': None,
                               'detectability': 'strong',
                               'chosen_pct': 96.5, 'rejected_pct': 96.1},   # both HIGH
    )
    grounded_output = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'chosen_better', 'weakness_present': True,
        'rejection_type': 'partial_factual_errors',
        'rejection_type_agrees_with_record': True,
        'type_scores': {'partial_factual_errors': 0.92}, 'confidence': 0.9,
        'evidence': [{'claim': 'rejected corrupts the cited authority\'s name',
                     'quote_from': 'rejected', 'quote': 'التغلبيي',
                     'explains_type': 'partial_factual_errors'}],
        'notes': 'high wording similarity does not make the corrupted name correct',
    }
    client = jc.RecordedJudgeClient()
    client.register('trap', grounded_output)
    res = llm_judge.judge_pair(ji, client, case_label='trap')
    passed = (res.status == 'ok' and res.judge_output.assessment == 'chosen_better'
             and res.judge_output.weakness_present)
    return _record_case('layer_A: similarity trap (high pct both sides)',
                        'assessment=chosen_better despite high similarity on rejected',
                        'status=%s assessment=%s' % (res.status,
                            res.judge_output.assessment if res.judge_output else None),
                        passed)


def test_hallucinated_quote_rejected():
    ji = _judge_input_for('less_faithful_reconstruction')
    bad = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'chosen_better', 'weakness_present': True,
        'rejection_type': 'less_faithful_reconstruction',
        'rejection_type_agrees_with_record': True, 'type_scores': {}, 'confidence': 0.8,
        'evidence': [{'claim': 'x', 'quote_from': 'source',
                     'quote': 'جملة لا وجود لها إطلاقا في هذا المصدر'}],
    }
    client = jc.RecordedJudgeClient()
    client.register('halluc', bad)
    res = llm_judge.judge_pair(ji, client, case_label='halluc')
    passed = res.status == 'unavailable' and res.error_type == 'InvalidJudgeOutput'
    return _record_case('layer_A: hallucinated evidence quote',
                        'status=unavailable, error_type=InvalidJudgeOutput',
                        'status=%s, error_type=%s' % (res.status, res.error_type),
                        passed)


def test_invalid_json():
    ji = _judge_input_for('wrong_register')
    client = jc.RecordedJudgeClient()
    client.register('badjson', 'not json at all { { {')
    res = llm_judge.judge_pair(ji, client, case_label='badjson')
    passed = res.status == 'unavailable' and res.error_type == 'InvalidJSON'
    return _record_case('layer_A: provider returns non-JSON text',
                        'status=unavailable, error_type=InvalidJSON',
                        'status=%s, error_type=%s' % (res.status, res.error_type),
                        passed)


def test_transport_failures():
    ok = True
    ji = _judge_input_for('verbosity')
    for label, exc, exc_name in (
            ('timeout', jc.TimeoutError('t'), 'TimeoutError'),
            ('rate_limited', jc.RateLimited('r'), 'RateLimited'),
            ('provider_error', jc.ProviderError('p'), 'ProviderError')):
        client = jc.RecordedJudgeClient()
        client.register(label, exc)
        res = llm_judge.judge_pair(ji, client, case_label=label)
        passed = res.status == 'unavailable' and res.error_type == exc_name
        ok &= _record_case('layer_A: transport failure (%s)' % label,
                           'status=unavailable, error_type=%s' % exc_name,
                           'status=%s, error_type=%s' % (res.status, res.error_type),
                           passed)
    return ok


def test_judge_tries_auto_confirm():
    ok = True
    ji = _judge_input_for('wrong_register')

    smuggle_field = {
        'schema_version': 'judge-1', 'judge_model_version': 'm',
        'assessment': 'chosen_better', 'weakness_present': False,
        'rejection_type': None, 'rejection_type_agrees_with_record': None,
        'type_scores': {}, 'confidence': 0.5, 'evidence': [],
        'verdict': 'AUTO_CONFIRM',
    }
    client = jc.RecordedJudgeClient()
    client.register('smuggle_field', smuggle_field)
    res = llm_judge.judge_pair(ji, client, case_label='smuggle_field')
    ok &= _record_case('layer_A: judge output smuggles verdict=AUTO_CONFIRM field',
                       'status=unavailable (rejected outright)',
                       'status=%s' % res.status, res.status == 'unavailable')

    smuggle_assessment = {
        'schema_version': 'judge-1', 'judge_model_version': 'm',
        'assessment': 'AUTO_CONFIRM', 'weakness_present': False,
        'rejection_type': None, 'rejection_type_agrees_with_record': None,
        'type_scores': {}, 'confidence': 0.99, 'evidence': [],
    }
    client.register('smuggle_assessment', smuggle_assessment)
    res2 = llm_judge.judge_pair(ji, client, case_label='smuggle_assessment')
    ok &= _record_case('layer_A: judge sets assessment="AUTO_CONFIRM"',
                       'status=unavailable (not a valid assessment value)',
                       'status=%s' % res2.status, res2.status == 'unavailable')
    return ok


# ==================================================================== layer B tests

def test_pipeline_all_nine_types_real_check_pair():
    ok = True
    ctx = _make_ctx()
    good_llm_output = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'chosen_better', 'weakness_present': True,
        'rejection_type': None, 'rejection_type_agrees_with_record': None,
        'type_scores': {}, 'confidence': 0.8,
        'evidence': [{'claim': 'chosen matches the source citation exactly',
                     'quote_from': 'source', 'quote': 'جابر بن حني التغلبي'}],
    }
    for rtype in jcon.REJECTION_TYPES:
        record = _make_record(rtype)
        client = jc.RecordedJudgeClient()
        _register_with_swap(client, rtype, good_llm_output)
        result = jp.adjudicate(record, ctx, client=client, case_label=rtype)
        expected = EXPECTED_VERDICT[rtype]
        actual = result.get('verdict')
        passed = actual == expected
        ok &= _record_case('layer_B: check_pair() verdict for %r (real, not stubbed)'
                           % rtype, expected, actual, passed)

        # calling discipline: AUTO_CONFIRM/FLAG_SUSPICIOUS must NEVER call the client
        if expected in (cdpo.AUTO_CONFIRM, cdpo.FLAG_SUSPICIOUS):
            no_call = len(client.calls) == 0
            no_judge_key = 'judge' not in result
            ok &= _record_case('layer_B: %r (%s) makes NO LLM call' % (rtype, expected),
                               '0 calls, no judge key',
                               '%d calls, judge key present=%s'
                               % (len(client.calls), 'judge' in result),
                               no_call and no_judge_key)
        else:
            want_calls = _expected_calls(rtype)
            called = len(client.calls) == want_calls
            has_judge_key = 'judge' in result and result['judge'].get('status') == 'ok'
            ok &= _record_case('layer_B: %r (NEEDS_JUDGE) DOES call the Judge' % rtype,
                               '%d call(s), judge.status=ok' % want_calls,
                               '%d calls, judge=%s' % (len(client.calls),
                                                       result.get('judge', {}).get('status')),
                               called and has_judge_key)
    return ok


def test_identical_chosen_rejected_no_call():
    ctx = _make_ctx()
    record = _make_record('partial_factual_errors', rejected=CHOSEN)      # identical
    client = jc.RecordedJudgeClient()
    result = jp.adjudicate(record, ctx, client=client, case_label='identical')
    passed = (result.get('verdict') == cdpo.FLAG_SUSPICIOUS and len(client.calls) == 0)
    return _record_case('layer_B: identical chosen/rejected',
                        'FLAG_SUSPICIOUS (degenerate pair), 0 LLM calls',
                        '%s, %d calls' % (result.get('verdict'), len(client.calls)),
                        passed)


def test_rejected_unrelated_to_source_no_call():
    ctx = _make_ctx()
    record = _make_record('less_faithful_reconstruction', rejected=UNRELATED_TO_SOURCE)
    client = jc.RecordedJudgeClient()
    result = jp.adjudicate(record, ctx, client=client, case_label='unrelated')
    passed = (result.get('verdict') == cdpo.FLAG_SUSPICIOUS and len(client.calls) == 0)
    return _record_case('layer_B: rejected completely unrelated to source',
                        'FLAG_SUSPICIOUS, 0 LLM calls',
                        '%s, %d calls' % (result.get('verdict'), len(client.calls)),
                        passed)


def test_pipeline_judge_failure_keeps_needs_judge():
    ok = True
    ctx = _make_ctx()
    for label, exc in (('timeout', jc.TimeoutError('t')),
                       ('rate_limited', jc.RateLimited('r')),
                       ('provider_error', jc.ProviderError('p')),
                       ('invalid_json', 'not json {{{')):
        record = _make_record('verbosity')          # a real NEEDS_JUDGE case
        client = jc.RecordedJudgeClient()
        client.register('f', exc)
        result = jp.adjudicate(record, ctx, client=client, case_label='f')
        passed = (result.get('verdict') == cdpo.NEEDS_JUDGE
                 and result.get('judge', {}).get('status') == 'unavailable')
        ok &= _record_case('layer_B: judge failure (%s) leaves verdict untouched' % label,
                           'verdict=NEEDS_JUDGE, judge.status=unavailable',
                           'verdict=%s, judge.status=%s'
                           % (result.get('verdict'), result.get('judge', {}).get('status')),
                           passed)
    return ok


def test_pipeline_no_client_is_safe():
    """adjudicate() with client=None on a NEEDS_JUDGE record must not crash and must
    not silently invent a verdict."""
    ctx = _make_ctx()
    record = _make_record('verbosity')
    result = jp.adjudicate(record, ctx, client=None)
    passed = (result.get('verdict') == cdpo.NEEDS_JUDGE
             and result.get('judge', {}).get('status') == 'unavailable')
    return _record_case('layer_B: adjudicate() with no client at all',
                        'verdict=NEEDS_JUDGE, judge.status=unavailable',
                        'verdict=%s, judge.status=%s'
                        % (result.get('verdict'), result.get('judge', {}).get('status')),
                        passed)


def test_escalation_to_flag_suspicious():
    """A NEEDS_JUDGE pair where the Judge disagrees with high confidence, and nothing
    deterministic already shows chosen clearly ahead, escalates to FLAG_SUSPICIOUS -
    never to AUTO_CONFIRM."""
    ctx = _make_ctx()
    record = _make_record('weak_organization')            # real tie -> NEEDS_JUDGE
    disagree = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'rejected_better', 'weakness_present': False,
        'rejection_type': None, 'rejection_type_agrees_with_record': False,
        'type_scores': {}, 'confidence': 0.9, 'evidence': [],
    }
    client = jc.RecordedJudgeClient()
    _register_with_swap(client, 'weak_organization', disagree)
    result = jp.adjudicate(record, ctx, client=client, case_label='weak_organization')
    passed = result.get('verdict') == cdpo.FLAG_SUSPICIOUS
    ok = _record_case('layer_B: high-confidence judge disagreement escalates',
                      'FLAG_SUSPICIOUS', result.get('verdict'), passed)

    never_auto = result.get('verdict') != cdpo.AUTO_CONFIRM
    ok &= _record_case('layer_B: escalation never produces AUTO_CONFIRM',
                       'verdict != AUTO_CONFIRM', result.get('verdict'), never_auto)
    return ok


def test_low_confidence_disagreement_does_not_escalate():
    ctx = _make_ctx()
    record = _make_record('weak_organization')
    disagree_low_conf = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'rejected_better', 'weakness_present': False,
        'rejection_type': None, 'rejection_type_agrees_with_record': False,
        'type_scores': {}, 'confidence': 0.4, 'evidence': [],
    }
    client = jc.RecordedJudgeClient()
    _register_with_swap(client, 'weak_organization', disagree_low_conf)
    result = jp.adjudicate(record, ctx, client=client, case_label='weak_organization')
    passed = result.get('verdict') == cdpo.NEEDS_JUDGE
    return _record_case('layer_B: LOW-confidence disagreement must not escalate',
                        'NEEDS_JUDGE (unchanged)', result.get('verdict'), passed)


def test_position_bias_detected():
    """A judge that picks the SAME SLOT both times is tracking position, not content.

    This is the ported check's whole purpose. wrong_register and weak_organization have
    no deterministic coverage anywhere (check_dpo.DETECTABILITY == 'none'), so a
    position-biased answer on them would otherwise be indistinguishable from a real one.
    """
    ok = True
    for rtype in sorted(llm_judge.SWAP_CHECK_TYPES):
        ji = _judge_input_for(rtype)
        biased = {
            'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
            'assessment': 'chosen_better', 'weakness_present': False,
            'rejection_type': None, 'rejection_type_agrees_with_record': None,
            'type_scores': {}, 'confidence': 0.9, 'evidence': [],
        }
        client = jc.RecordedJudgeClient()
        # the SAME answer both times, i.e. whichever text sits in the first slot wins
        client.register(rtype, biased)
        client.register(rtype + '_swap', biased)
        res = llm_judge.judge_pair(ji, client, case_label=rtype)
        passed = (res.status == 'unavailable' and res.error_type == 'PositionBias')
        ok &= _record_case('swap: position-biased judge refused for %r' % rtype,
                           "unavailable/PositionBias",
                           '%s/%s' % (res.status, res.error_type), passed)
    return ok


def test_swap_pass_failure_does_not_fall_back():
    """If the second pass fails, one good pass is NOT enough for these types."""
    rtype = sorted(llm_judge.SWAP_CHECK_TYPES)[0]
    ji = _judge_input_for(rtype)
    good = {
        'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
        'assessment': 'chosen_better', 'weakness_present': False,
        'rejection_type': None, 'rejection_type_agrees_with_record': None,
        'type_scores': {}, 'confidence': 0.9, 'evidence': [],
    }
    client = jc.RecordedJudgeClient()
    client.register(rtype, good)          # first pass fine, no `_swap` recording at all
    res = llm_judge.judge_pair(ji, client, case_label=rtype)
    passed = (res.status == 'unavailable' and res.error_type == 'SwapPassUnavailable')
    return _record_case('swap: failed second pass does not fall back to one pass',
                        'unavailable/SwapPassUnavailable',
                        '%s/%s' % (res.status, res.error_type), passed)


def test_swap_check_scoped_to_zero_coverage_types():
    """The other seven types must NOT pay for a second call."""
    ok = True
    for rtype in jcon.REJECTION_TYPES:
        if rtype in llm_judge.SWAP_CHECK_TYPES:
            continue
        ji = _judge_input_for(rtype)
        good = {
            'schema_version': 'judge-1', 'judge_model_version': 'recorded/offline',
            'assessment': 'chosen_better', 'weakness_present': False,
            'rejection_type': None, 'rejection_type_agrees_with_record': None,
            'type_scores': {}, 'confidence': 0.8, 'evidence': [],
        }
        client = jc.RecordedJudgeClient()
        client.register(rtype, good)      # deliberately no `_swap`
        res = llm_judge.judge_pair(ji, client, case_label=rtype)
        passed = (res.status == 'ok' and len(client.calls) == 1)
        ok &= _record_case('swap: %r judged once, no swap pass' % rtype,
                           'status=ok, 1 call',
                           'status=%s, %d call(s)' % (res.status, len(client.calls)),
                           passed)
    return ok


def test_swap_types_track_check_dpo():
    """The swap set is derived from check_dpo, not written out - so a new 'none' type
    there is protected automatically instead of silently going uncovered."""
    expected = frozenset(t for t, d in cdpo.DETECTABILITY.items() if d == 'none')
    passed = llm_judge.SWAP_CHECK_TYPES == expected
    return _record_case('swap: SWAP_CHECK_TYPES tracks check_dpo.DETECTABILITY',
                        sorted(expected), sorted(llm_judge.SWAP_CHECK_TYPES), passed)


def test_combine_dpo_and_check_pair_not_modified():
    """Sanity: check_dpo's own self-test still passes unchanged - i.e. nothing in this
    package altered check_dpo.py's behaviour. Imported and run in-process rather than
    shelling out, so it participates in this same report.
    """
    passed = cdpo.run_self_test()
    return _record_case('sanity: check_dpo.py self-test still passes untouched',
                        True, passed, passed)


# =========================================================================== runner

def run_self_test():
    results = [
        test_contract_all_nine_types(),
        test_similarity_trap(),
        test_hallucinated_quote_rejected(),
        test_invalid_json(),
        test_transport_failures(),
        test_judge_tries_auto_confirm(),
        test_pipeline_all_nine_types_real_check_pair(),
        test_identical_chosen_rejected_no_call(),
        test_rejected_unrelated_to_source_no_call(),
        test_pipeline_judge_failure_keeps_needs_judge(),
        test_pipeline_no_client_is_safe(),
        test_escalation_to_flag_suspicious(),
        test_low_confidence_disagreement_does_not_escalate(),
        test_position_bias_detected(),
        test_swap_pass_failure_does_not_fall_back(),
        test_swap_check_scoped_to_zero_coverage_types(),
        test_swap_types_track_check_dpo(),
        test_combine_dpo_and_check_pair_not_modified(),
    ]
    ok = all(results)

    width_case = max(len(r[0]) for r in REPORT)
    print('%-*s | %-40s | %-40s | %s' % (width_case, 'case', 'expected', 'actual', 'result'))
    print('-' * (width_case + 100))
    for case, expected, actual, passed in REPORT:
        print('%-*s | %-40s | %-40s | %s'
              % (width_case, case, str(expected)[:40], str(actual)[:40],
                 'PASS' if passed else 'FAIL'))
    n_pass = sum(1 for *_, p in REPORT if p)
    print('-' * (width_case + 100))
    print('%d/%d cases passed' % (n_pass, len(REPORT)))
    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM Judge test suite.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
