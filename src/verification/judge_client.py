# -*- coding: utf-8 -*-
"""Task 3 - LLM judge transport layer: the backend seam.

This module is the ONLY place in the judge stack that knows how a model is reached.
`judge_sft.py` and `judge_dpo.py` build prompts and interpret answers; neither imports a
provider SDK, and neither may. Swapping the model swaps a backend object here and nothing
else.

طبقة النقل للحَكَم: هي الموضع الوحيد الذي يعرف كيف يُستدعى النموذج.

Why a seam instead of an SDK call
---------------------------------
The judging model is a Qwen instance chosen by the team, distinct from the model used for
reconstruction. The endpoint and credential are not available yet. Everything in this
stack that does NOT depend on which model answers - the rubric, the verdict vocabulary,
the combination with the rule-based layer, the synthetic suite, the self-tests - is built
and tested now against a scripted backend; the provider adapter is added later without
touching any of it.

So this module deliberately contains:
  * no `import anthropic`, no `import openai`, no `requests`/`httpx` call
  * no provider base URL, no model id, no pricing constant
A backend is anything exposing `.generate(system, user, **kw) -> JudgeResponse`. That is
the whole contract.

Pricing is a PARAMETER, not a constant
--------------------------------------
`estimate_cost()` takes per-MTok rates from the caller. Nothing here knows what a token
costs, because the model is not chosen yet and hardcoding one vendor's rates would
produce confidently wrong numbers for another. With no rates supplied, usage is reported
in tokens and cost reads `None` - an unknown cost must never render as a number.

Fail-closed, the same rule as the rest of the stack
---------------------------------------------------
Every failure path here - transport error, timeout, refusal, truncated body, unparseable
JSON, a body that parses but omits required keys - resolves to ABSTAIN, never to a pass.
`src/deployment/ACCEPTANCE_CONTRACT.md` routes a judge abstention to HOLD_FOR_REVIEW, so
a broken or unreachable judge degrades to human review instead of silently approving
records. This is the same principle as `audit_staged_arabic.py` refusing rather than
returning 0, and as `NO_FACT_COVERAGE` being neither a pass nor a fail: the absence of a
check must never read as a clean result.
"""

from __future__ import unicode_literals

import hashlib
import io
import json
import os
import re
import sys
import time


# --------------------------------------------------------------------- verdict vocabulary
#
# Distinct string values, not bare PASS/FAIL, so a verdict is self-identifying in a log or
# a merged record. `verify_sft.PASS` is also the string 'PASS'; if both layers used it, a
# combined record would carry two different 'PASS' values meaning different things.
JUDGE_PASS = 'JUDGE_PASS'
JUDGE_FAIL = 'JUDGE_FAIL'
JUDGE_REVIEW = 'JUDGE_REVIEW'
JUDGE_ABSTAIN = 'JUDGE_ABSTAIN'

JUDGE_VERDICTS = (JUDGE_PASS, JUDGE_FAIL, JUDGE_REVIEW, JUDGE_ABSTAIN)

# Why a judge declined to answer. Recorded so an abstention rate can be read by cause -
# a stack abstaining 40% of the time because of parse failures is a prompt bug, while one
# abstaining because the backend is down is an ops problem, and they need different fixes.
ABSTAIN_TRANSPORT = 'transport_error'
ABSTAIN_UNPARSEABLE = 'unparseable_response'
ABSTAIN_INCOMPLETE = 'incomplete_response'
ABSTAIN_REFUSED = 'model_refused'
ABSTAIN_NOT_CONFIGURED = 'backend_not_configured'


class JudgeUnavailable(Exception):
    """The backend could not produce any answer. Callers turn this into an abstention."""


# ------------------------------------------------------------------------------- response

class JudgeResponse(object):
    """One raw backend answer, before any judging semantics are read into it."""

    def __init__(self, text, model_id=None, input_tokens=None, output_tokens=None,
                 latency_s=None, raw=None):
        self.text = text
        self.model_id = model_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_s = latency_s
        self.raw = raw

    def usage(self):
        return {'model_id': self.model_id,
                'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'latency_s': self.latency_s}


def estimate_cost(input_tokens, output_tokens,
                  price_in_per_mtok=None, price_out_per_mtok=None):
    """Cost in currency units, or None when rates were not supplied.

    Rates are arguments because the judging model is not chosen yet. Returning None rather
    than 0.0 for "unknown" is deliberate: a zero would sum into a total and read as free.
    """
    if price_in_per_mtok is None or price_out_per_mtok is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1e6) * price_in_per_mtok + \
           (output_tokens / 1e6) * price_out_per_mtok


class UsageLedger(object):
    """Running token / latency / cost totals across a batch of judge calls.

    This is the first module in the project that spends money per record, so the totals
    are part of the output rather than something to reconstruct from logs afterwards.
    """

    def __init__(self, price_in_per_mtok=None, price_out_per_mtok=None):
        self.calls = 0
        self.abstentions = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_s = 0.0
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok

    def record(self, resp, abstained=False):
        self.calls += 1
        if abstained:
            self.abstentions += 1
        if resp is None:
            return
        self.input_tokens += resp.input_tokens or 0
        self.output_tokens += resp.output_tokens or 0
        self.latency_s += resp.latency_s or 0.0

    def summary(self):
        cost = estimate_cost(self.input_tokens, self.output_tokens,
                             self.price_in, self.price_out)
        return {'calls': self.calls,
                'abstentions': self.abstentions,
                'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'total_latency_s': round(self.latency_s, 3),
                'mean_latency_s': round(self.latency_s / self.calls, 3) if self.calls else None,
                'estimated_cost': cost,
                'cost_basis': ('per-MTok in=%s out=%s' % (self.price_in, self.price_out)
                               if cost is not None else
                               'UNKNOWN - no pricing supplied for this backend')}


# ------------------------------------------------------------------------------- backends

class JudgeBackend(object):
    """Interface. A backend maps (system, user) to a JudgeResponse.

    Implementations must raise JudgeUnavailable on any failure rather than returning a
    degraded answer, so the caller can abstain instead of judging on noise.
    """

    name = 'abstract'

    def generate(self, system, user, max_tokens=1024, temperature=0.0):
        raise NotImplementedError('JudgeBackend is an interface; use a concrete backend')


class ScriptedBackend(JudgeBackend):
    """Deterministic canned answers. Used by every self-test in this stack.

    Two modes, both offline:
      by_key   dict {probe_id: response_text}; the caller passes probe_id through
               `generate(..., probe_id=...)`
      by_order list of response texts returned in sequence

    This is what lets the judge's own logic - rubric parsing, verdict combination, the
    contract table - be tested without a credential and without spending anything.
    """

    name = 'scripted'

    def __init__(self, by_key=None, by_order=None, model_id='scripted-mock',
                 latency_s=0.0):
        self._by_key = by_key or {}
        self._by_order = list(by_order or [])
        self._i = 0
        self.model_id = model_id
        self.latency_s = latency_s
        self.seen = []

    def generate(self, system, user, max_tokens=1024, temperature=0.0, probe_id=None):
        self.seen.append({'probe_id': probe_id, 'system': system, 'user': user})
        if probe_id is not None and probe_id in self._by_key:
            text = self._by_key[probe_id]
        elif self._by_order:
            if self._i >= len(self._by_order):
                raise JudgeUnavailable('scripted backend exhausted after %d calls'
                                       % self._i)
            text = self._by_order[self._i]
            self._i += 1
        else:
            raise JudgeUnavailable('scripted backend has no answer for probe_id=%r'
                                   % (probe_id,))
        if isinstance(text, Exception):
            raise text
        return JudgeResponse(text,
                             model_id=self.model_id,
                             input_tokens=_approx_tokens(system) + _approx_tokens(user),
                             output_tokens=_approx_tokens(text),
                             latency_s=self.latency_s)


class CachingBackend(JudgeBackend):
    """Wraps another backend and records every answer to a JSONL transcript.

    Two purposes. It makes a paid run replayable offline - the same suite re-runs from the
    transcript at zero cost - and it keeps the evidence for any reported result, so a
    verdict table in a commit message can be checked against what the model actually
    returned rather than trusted.

    `replay_only=True` turns it into an offline backend over a previous run; a cache miss
    then raises rather than silently reaching for the network.
    """

    name = 'caching'

    def __init__(self, inner, path, replay_only=False):
        self.inner = inner
        self.path = path
        self.replay_only = replay_only
        self._cache = {}
        if os.path.exists(path):
            with io.open(path, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._cache[rec['key']] = rec

    @staticmethod
    def key_for(system, user, model_id):
        h = hashlib.sha256()
        for part in (model_id or '', system or '', user or ''):
            h.update(part.encode('utf-8'))
            h.update(b'\x00')
        return h.hexdigest()[:32]

    def generate(self, system, user, max_tokens=1024, temperature=0.0, probe_id=None):
        model_id = getattr(self.inner, 'model_id', None) or self.inner.name
        key = self.key_for(system, user, model_id)
        hit = self._cache.get(key)
        if hit is not None:
            return JudgeResponse(hit['text'], model_id=hit.get('model_id'),
                                 input_tokens=hit.get('input_tokens'),
                                 output_tokens=hit.get('output_tokens'),
                                 latency_s=hit.get('latency_s'))
        if self.replay_only:
            raise JudgeUnavailable(
                'replay-only cache miss (probe_id=%r). The transcript at %s does not '
                'cover this prompt; re-run against a live backend to extend it.'
                % (probe_id, self.path))
        resp = self.inner.generate(system, user, max_tokens=max_tokens,
                                   temperature=temperature)
        rec = {'key': key, 'probe_id': probe_id, 'model_id': resp.model_id,
               'text': resp.text, 'input_tokens': resp.input_tokens,
               'output_tokens': resp.output_tokens, 'latency_s': resp.latency_s,
               'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        self._cache[key] = rec
        with io.open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return resp


class NotConfiguredBackend(JudgeBackend):
    """Placeholder for a provider not yet wired. Always raises, never returns.

    The judging model is a Qwen instance; its endpoint and credential are pending from the
    team. Until that adapter is written this stands in its place, so a caller that reaches
    for a live judge gets an explicit, named failure that routes to ABSTAIN - rather than
    an import error, or worse, a silent default to some other provider that happens to be
    installed.
    """

    name = 'not_configured'

    def __init__(self, provider='qwen', detail=None):
        self.provider = provider
        self.detail = detail or (
            'no endpoint or credential has been supplied for %r yet' % provider)

    def generate(self, system, user, max_tokens=1024, temperature=0.0, probe_id=None):
        raise JudgeUnavailable('backend %r is not configured: %s' % (self.provider,
                                                                    self.detail))


def make_backend(name, **kw):
    """Factory. Keeps provider selection in one place and out of the judge modules.

    Adding the Qwen adapter means adding one branch here and one class; no prompt, no
    rubric and no self-test changes.
    """
    name = (name or '').lower()
    if name in ('scripted', 'mock'):
        return ScriptedBackend(**kw)
    if name in ('http', 'endpoint', 'qwen', 'openai'):
        # Imported lazily so this module stays a pure interface: judge_client must not
        # grow a transport, or the seam it exists to provide stops being a seam.
        import judge_backend_http as hb
        try:
            return hb.HttpBackend(hb.from_env(), **kw)
        except (JudgeUnavailable, hb.JudgeUnavailable) as exc:
            # Both spellings on purpose. Running this file as __main__ makes
            # __main__.JudgeUnavailable a DIFFERENT class object from the
            # judge_client.JudgeUnavailable that judge_backend_http imported, so a
            # single-name except silently misses and the "not configured" path
            # crashes instead of abstaining. The self-tests run as __main__, which
            # is how this surfaced.
            # Not configured is not an error here - it is a backend that abstains, so a
            # caller gets HOLD_FOR_REVIEW with a message naming what is missing rather
            # than a crash or a silent fallback to some other installed provider.
            return NotConfiguredBackend(provider=name, detail=str(exc))
    raise ValueError('unknown backend %r (known: scripted, http)' % name)


# --------------------------------------------------------------------------- JSON parsing

_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


def parse_judge_json(text, required_keys=()):
    """Extract the judge's JSON object, tolerantly.

    Models wrap JSON in prose, in ``` fences, or emit it with trailing commentary. This
    tries, in order: the whole string, any fenced block, then the outermost {...} span.

    Returns (obj, None) or (None, abstain_reason). A body that parses but lacks a required
    key is INCOMPLETE, not a pass - the judge did not answer the question asked.
    """
    if text is None or not text.strip():
        return None, ABSTAIN_UNPARSEABLE

    candidates = [text]
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    i, j = text.find('{'), text.rfind('}')
    if i >= 0 and j > i:
        candidates.append(text[i:j + 1])

    obj = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return None, ABSTAIN_UNPARSEABLE

    missing = [k for k in required_keys if k not in obj]
    if missing:
        return None, ABSTAIN_INCOMPLETE
    return obj, None


def _approx_tokens(text):
    """Rough token count for accounting when a backend reports none.

    MEASURED on this corpus: the two chunk files run 6.11 and 5.52 characters per
    whitespace token. Sub-word tokenizers split Arabic far finer than that; the e5
    tokenizer measured ~2.3 chars/token earlier in this project. 3.0 is a deliberate
    middle estimate and is NOT a substitute for the backend's own usage numbers - it
    exists so the scripted backend produces plausible ledger totals in tests.
    """
    if not text:
        return 0
    return int(len(text) / 3.0)


# ------------------------------------------------------------------------------ self-test

_SCHEMA_OK = '{"overall": "pass", "confidence": "high"}'


def run_self_test():
    ok = True

    # --- parsing: the three shapes a model actually emits -------------------------------
    parse_cases = [
        ('bare object', _SCHEMA_OK, True),
        ('fenced', '```json\n' + _SCHEMA_OK + '\n```', True),
        ('prose wrapped', 'Here is my assessment:\n' + _SCHEMA_OK + '\nHope that helps.',
         True),
        ('prose only', 'The response looks good to me.', False),
        ('empty', '', False),
        ('none', None, False),
    ]
    for label, text, want_ok in parse_cases:
        obj, reason = parse_judge_json(text, required_keys=('overall',))
        if (obj is not None) != want_ok:
            print('  [FAIL] parse %-14s -> obj=%s reason=%s, expected ok=%s'
                  % (label, obj, reason, want_ok))
            ok = False

    # A body that parses but omits a required key must NOT read as an answer.
    obj, reason = parse_judge_json('{"confidence": "high"}', required_keys=('overall',))
    if obj is not None or reason != ABSTAIN_INCOMPLETE:
        print('  [FAIL] missing required key resolved to %s/%s, expected INCOMPLETE'
              % (obj, reason))
        ok = False

    # A JSON scalar/array is not a verdict object.
    for text in ('"pass"', '[1,2,3]', '42'):
        obj, reason = parse_judge_json(text, required_keys=('overall',))
        if obj is not None:
            print('  [FAIL] non-object JSON %r accepted as a verdict' % text)
            ok = False

    # --- cost: unknown rates must stay unknown -----------------------------------------
    if estimate_cost(1000, 500) is not None:
        print('  [FAIL] cost with no rates returned a number instead of None')
        ok = False
    if estimate_cost(1000, 500, 1.0, None) is not None:
        print('  [FAIL] cost with a half-supplied rate pair returned a number')
        ok = False
    c = estimate_cost(1000000, 1000000, 2.0, 10.0)
    if abs(c - 12.0) > 1e-9:
        print('  [FAIL] cost arithmetic: got %s, expected 12.0' % c)
        ok = False

    led = UsageLedger()
    led.record(JudgeResponse('x', input_tokens=10, output_tokens=5, latency_s=0.5))
    s = led.summary()
    if s['estimated_cost'] is not None or 'UNKNOWN' not in s['cost_basis']:
        print('  [FAIL] ledger reported a cost without pricing')
        ok = False

    # --- the not-configured backend must raise, never return ---------------------------
    nb = make_backend('qwen')
    try:
        nb.generate('sys', 'user')
        print('  [FAIL] NotConfiguredBackend returned instead of raising')
        ok = False
    except JudgeUnavailable:
        pass

    # --- scripted backend: keyed, ordered, exhausted, and raising -----------------------
    sb = ScriptedBackend(by_key={'p1': _SCHEMA_OK})
    if sb.generate('s', 'u', probe_id='p1').text != _SCHEMA_OK:
        print('  [FAIL] scripted keyed lookup')
        ok = False
    try:
        sb.generate('s', 'u', probe_id='missing')
        print('  [FAIL] scripted backend answered an unknown probe_id')
        ok = False
    except JudgeUnavailable:
        pass

    sb2 = ScriptedBackend(by_order=[_SCHEMA_OK])
    sb2.generate('s', 'u')
    try:
        sb2.generate('s', 'u')
        print('  [FAIL] exhausted scripted backend kept answering')
        ok = False
    except JudgeUnavailable:
        pass

    # A scripted Exception is raised, so transport failures are testable offline.
    sb3 = ScriptedBackend(by_key={'boom': JudgeUnavailable('simulated timeout')})
    try:
        sb3.generate('s', 'u', probe_id='boom')
        print('  [FAIL] scripted exception was not raised')
        ok = False
    except JudgeUnavailable:
        pass

    # --- caching backend round-trips and then replays offline ---------------------------
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), 'transcript.jsonl')
    inner = ScriptedBackend(by_order=[_SCHEMA_OK])
    cb = CachingBackend(inner, tmp)
    r1 = cb.generate('sys', 'user', probe_id='c1')
    cb2 = CachingBackend(ScriptedBackend(by_order=[]), tmp, replay_only=True)
    r2 = cb2.generate('sys', 'user', probe_id='c1')
    if r1.text != r2.text:
        print('  [FAIL] cached replay returned different text')
        ok = False
    try:
        cb2.generate('sys', 'DIFFERENT user', probe_id='c2')
        print('  [FAIL] replay-only cache miss did not raise')
        ok = False
    except JudgeUnavailable:
        pass

    # --- the module must not import a provider SDK --------------------------------------
    # The seam is the whole point; an SDK import here would defeat it silently.
    src = io.open(__file__.replace('.pyc', '.py'), encoding='utf-8').read()
    for banned in ('import anthropic', 'import openai', 'from anthropic',
                   'from openai', 'import requests', 'import httpx'):
        # match only real import statements, not prose in this docstring
        if re.search(r'^\s*%s' % re.escape(banned), src, re.M):
            print('  [FAIL] provider SDK import found in the seam: %r' % banned)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LLM judge transport layer.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
