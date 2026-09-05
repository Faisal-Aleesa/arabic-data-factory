# -*- coding: utf-8 -*-
"""LLM Judge - provider/model client abstraction.

Design goals (from spec)
-------------------------
- provider/model/endpoint/model_version/timeout/max_retries are configuration, not
  hardcoded.
- The API key is read ONLY from the JUDGE_API_KEY environment variable. It is never
  logged, never placed in an exception message, never serialized into any result dict.
- Errors are classified so judge_pipeline.py can react uniformly: TimeoutError,
  RateLimited, ProviderError, InvalidJSON. All four subclass JudgeClientError so a
  caller that only wants "did this fail" can catch one type.
- RecordedJudgeClient lets llm_judge.py and the test suite run completely offline, with
  no network and no API key, by replaying pre-scripted raw text responses keyed by a
  request fingerprint.

What this module deliberately does NOT do
-------------------------------------------
It does not parse or validate the JSON returned by the model - that is
judge_contract.parse_and_validate()'s job. This module's contract ends at "here is the
raw text the model returned" or "here is why I could not get one". Keeping the layers
separate means a transport failure (timeout) and a semantic failure (bad JSON schema)
can never be confused with each other, which judge_pipeline.py depends on to decide
what to report.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- errors

class JudgeClientError(Exception):
    """Base for every classified client failure. Never carries the API key."""


class TimeoutError(JudgeClientError):          # noqa: A001 - intentional, see module doc
    """The provider did not respond within `timeout` seconds, after retries."""


class RateLimited(JudgeClientError):
    """The provider returned a rate-limit response (HTTP 429 or provider-specific)."""


class ProviderError(JudgeClientError):
    """The provider returned an error response (HTTP 4xx/5xx other than rate-limit,
    a malformed HTTP envelope, or a connection failure)."""


class InvalidJSON(JudgeClientError):
    """The provider responded successfully at the transport level, but the completion
    text is not parseable JSON. (A structurally valid-but-wrong JSON is instead an
    judge_contract.InvalidJudgeOutput, raised one layer up in llm_judge.py.)"""


# ---------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class ClientConfig:
    provider: str
    model: str
    model_version: str
    endpoint: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 2

    @staticmethod
    def from_env(provider: str = 'unset',
                model: str = 'unset',
                model_version: Optional[str] = None,
                endpoint: Optional[str] = None,
                timeout: float = 30.0,
                max_retries: int = 2) -> 'ClientConfig':
        return ClientConfig(
            provider=provider,
            model=model,
            model_version=model_version or ('%s/%s' % (provider, model)),
            endpoint=endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )


# -------------------------------------------------------------------------- base class

class BaseJudgeClient(object):
    """provider/model abstraction. Concrete clients implement `_call()` only.

    `judge_text(system, user_payload)` is the public entry point every caller in
    llm_judge.py uses; it is identical for every subclass and handles retry counting,
    so a new provider only has to implement one transport method.
    """

    def __init__(self, config: ClientConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def model_version(self) -> str:
        return self.config.model_version

    def judge_text(self, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        """Return the raw completion TEXT (not parsed). Retries on Timeout/RateLimited
        only - a ProviderError or InvalidJSON is not retried, since neither is likely to
        change on an immediate retry and both need to surface promptly.
        """
        last_exc: Optional[JudgeClientError] = None
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            try:
                return self._call(system_prompt, user_payload)
            except (TimeoutError, RateLimited) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise
        # unreachable, but keeps type-checkers happy
        raise last_exc or ProviderError('exhausted retries with no exception recorded')

    def _call(self, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        raise NotImplementedError


# ----------------------------------------------------------------- real HTTP provider

class BackendJudgeClient(BaseJudgeClient):
    """Real provider access, over the project's own `judge_backend_http.HttpBackend`.

    Replaces the delivery's AnthropicJudgeClient. The judging model is a Qwen instance
    served over a hosted API, so a client hardwired to Anthropic's Messages API - a
    top-level `system` field, an `x-api-key` header, and a `content` block list to unpack
    - was the wrong wire format on all three counts. `judge_backend_http` already speaks
    the OpenAI-compatible shape most hosted Qwen endpoints expose, plus a `custom` dialect
    for anything else, and it is the transport this project has already proven end to end
    against a live HTTP server.

    عميل حقيقي فوق طبقة النقل الخاصة بالمشروع، لا فوق واجهة مزوّد بعينه.

    Nothing else in the delivery changes: `judge_text()` on the base class is untouched,
    and `llm_judge.judge_pair()` only ever sees a raw completion string, so it never knew
    which provider answered in the first place.

    Retry policy - deliberately NOT stacked
    ---------------------------------------
    `BaseJudgeClient.judge_text()` retries on TimeoutError / RateLimited only. HttpBackend
    ALREADY retries internally, with its own backoff and its own retryable-status set. So
    every failure here is mapped to ProviderError, which the base class does not retry.

    Mapping a transport failure to a retryable type instead would multiply the two layers
    - `max_retries=2` on each side is nine attempts, not three - and each one re-sends a
    full source chunk. One retry policy, owned by the transport that can actually see the
    HTTP status.
    """

    def __init__(self, config: ClientConfig, backend=None):
        super().__init__(config)
        if backend is None:
            import os as _os
            import sys as _sys
            _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                              '..', 'verification'))
            import judge_backend_http as _hb
            backend = _hb.HttpBackend(_hb.from_env())
        self._backend = backend

    @classmethod
    def from_env(cls, timeout: float = 60.0, max_retries: int = 2):
        """Build from JUDGE_API_* - see docs/JUDGE_ENDPOINT_SETUP.md.

        The endpoint, model and credential all come from the environment, so no provider
        name or model id is written down here.
        """
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          '..', 'verification'))
        import judge_backend_http as _hb
        cfg_http = _hb.from_env()
        cfg = ClientConfig(provider=cfg_http.dialect,
                           model=cfg_http.model,
                           model_version='%s/%s' % (cfg_http.dialect, cfg_http.model),
                           endpoint=cfg_http.url(),
                           timeout=timeout,
                           max_retries=max_retries)
        return cls(cfg, backend=_hb.HttpBackend(cfg_http))

    def _call(self, system_prompt, user_payload):
        """One completion, with `_case_label` lifted out of the payload.

        Two things happen to that key, and both are deliberate:

        1. It is REMOVED from the user message. `_case_label` is test-harness metadata
           (RecordedJudgeClient's fingerprint), not part of the judging task, and the
           delivery's own client serialised the whole payload - so every live call would
           have paid to send the model a field naming its own test case. Stripping it
           keeps the prompt identical whether a run is scripted or live.

        2. It is passed to the backend as `probe_id`, which is how this project's
           scripted backend keys its canned answers. Without that bridge, an offline run
           through this client cannot be scripted at all.
        """
        payload = dict(user_payload)
        probe_id = payload.pop('_case_label', None)
        user_text = json.dumps(payload, ensure_ascii=False)
        try:
            try:
                resp = self._backend.generate(system_prompt, user_text,
                                              max_tokens=1024, temperature=0.0,
                                              probe_id=probe_id)
            except TypeError:
                # a backend whose generate() takes no probe_id is still valid
                resp = self._backend.generate(system_prompt, user_text,
                                              max_tokens=1024, temperature=0.0)
        except Exception as exc:
            # judge_backend_http raises judge_client.JudgeUnavailable. Importing that
            # class here to catch it by name would re-create the module-identity trap
            # this package was renamed to escape, so match on the name instead and
            # re-raise anything unexpected untouched.
            if type(exc).__name__ != 'JudgeUnavailable':
                raise
            raise ProviderError('judge endpoint unavailable: %s' % exc) from None
        text = resp.text
        if not text or not text.strip():
            raise ProviderError('judge endpoint returned an empty completion')
        return text


class RecordedJudgeClient(BaseJudgeClient):
    """Replays pre-scripted raw completion text. No network, no API key.

    `recordings` maps a fingerprint (built from the request payload) to either:
      - a raw JSON text string (success), or
      - a JudgeClientError instance/subclass (to simulate that failure mode), or
      - a callable(payload) -> str | JudgeClientError, for cases where the response
        must depend on the input (e.g. echoing a literal quote back).

    Lookup key defaults to the case `label` if the caller includes one in the payload
    under `_case_label` (llm_judge.py's test harness sets this); otherwise falls back to
    a hash of (chosen, rejected, rejection_type_declared), which is stable enough for
    fixed test fixtures without needing exact byte-for-byte payload equality.
    """

    def __init__(self, recordings: Optional[Dict[str, Any]] = None,
                config: Optional[ClientConfig] = None):
        super().__init__(config or ClientConfig.from_env(
            provider='recorded', model='recorded-fixture', model_version='recorded/offline'))
        self.recordings: Dict[str, Any] = dict(recordings or {})
        self.calls: List[Dict[str, Any]] = []          # inspectable call log for tests

    @staticmethod
    def fingerprint(user_payload: Dict[str, Any]) -> str:
        if '_case_label' in user_payload:
            return user_payload['_case_label']
        return '%s|%s|%s' % (user_payload.get('rejection_type_declared'),
                             hash(user_payload.get('chosen')),
                             hash(user_payload.get('rejected')))

    def register(self, label: str, response: Any) -> None:
        self.recordings[label] = response

    def _call(self, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        self.calls.append(user_payload)
        key = self.fingerprint(user_payload)
        if key not in self.recordings:
            raise ProviderError('RecordedJudgeClient has no recording for key %r - add '
                                'one with .register() before calling' % key)
        entry = self.recordings[key]
        if callable(entry):
            entry = entry(user_payload)
        if isinstance(entry, JudgeClientError):
            raise entry
        if isinstance(entry, type) and issubclass(entry, JudgeClientError):
            raise entry('scripted failure for key %r' % key)
        if not isinstance(entry, str):
            entry = json.dumps(entry, ensure_ascii=False)
        return entry


# ------------------------------------------------------------------------- self-test

def run_self_test():
    ok = True

    # RecordedJudgeClient basic replay
    rc = RecordedJudgeClient()
    rc.register('c1', '{"assessment": "chosen_better"}')
    out = rc.judge_text('sys', {'_case_label': 'c1'})
    if out != '{"assessment": "chosen_better"}':
        print('  [FAIL] RecordedJudgeClient did not replay the registered text')
        ok = False
    if len(rc.calls) != 1:
        print('  [FAIL] RecordedJudgeClient did not log the call'); ok = False

    # dict payloads get JSON-encoded automatically
    rc.register('c2', {'assessment': 'equivalent'})
    out2 = rc.judge_text('sys', {'_case_label': 'c2'})
    if json.loads(out2)['assessment'] != 'equivalent':
        print('  [FAIL] dict recording was not encoded correctly'); ok = False

    # scripted transport failures
    rc.register('c3', TimeoutError)
    try:
        rc.judge_text('sys', {'_case_label': 'c3'})
        print('  [FAIL] c3 should have raised TimeoutError'); ok = False
    except TimeoutError:
        pass

    rc.register('c4', RateLimited('too many requests'))
    try:
        rc.judge_text('sys', {'_case_label': 'c4'})
        print('  [FAIL] c4 should have raised RateLimited'); ok = False
    except RateLimited:
        pass

    # retries: a client that fails twice then succeeds should still return the value,
    # given max_retries >= 2
    attempts_made = {'n': 0}

    def flaky(payload):
        attempts_made['n'] += 1
        if attempts_made['n'] < 3:
            return TimeoutError('flaky')
        return '{"ok": true}'

    cfg = ClientConfig.from_env(provider='recorded', model='flaky', max_retries=3,
                                timeout=1.0)
    rc2 = RecordedJudgeClient(config=cfg)
    rc2.register('c5', flaky)
    out5 = rc2.judge_text('sys', {'_case_label': 'c5'})
    if out5 != '{"ok": true}' or attempts_made['n'] != 3:
        print('  [FAIL] retry-until-success did not behave as expected: %s attempts=%d'
              % (out5, attempts_made['n']))
        ok = False

    # a ProviderError must NOT be retried
    attempts2 = {'n': 0}

    def always_provider_error(payload):
        attempts2['n'] += 1
        return ProviderError('nope')

    rc3 = RecordedJudgeClient(config=ClientConfig.from_env(max_retries=3))
    rc3.register('c6', always_provider_error)
    try:
        rc3.judge_text('sys', {'_case_label': 'c6'})
        print('  [FAIL] ProviderError should propagate'); ok = False
    except ProviderError:
        if attempts2['n'] != 1:
            print('  [FAIL] ProviderError should not be retried, got %d attempts'
                  % attempts2['n'])
            ok = False

    # unknown key -> ProviderError (never a silent None)
    rc4 = RecordedJudgeClient()
    try:
        rc4.judge_text('sys', {'_case_label': 'never_registered'})
        print('  [FAIL] unregistered key should raise'); ok = False
    except ProviderError:
        pass

    # BackendJudgeClient (which replaced AnthropicJudgeClient) must refuse cleanly when
    # the endpoint is not configured, and must never leak a credential in the message.
    saved = {k: os.environ.pop(k, None)
             for k in ('JUDGE_API_BASE_URL', 'JUDGE_API_MODEL', 'JUDGE_API_KEY_VAR')}
    try:
        try:
            BackendJudgeClient.from_env()
            print('  [FAIL] BackendJudgeClient.from_env() succeeded with nothing set')
            ok = False
        except Exception as e:
            # judge_backend_http raises JudgeUnavailable; it must name what is missing.
            if 'JUDGE_API_BASE_URL' not in str(e):
                print('  [FAIL] error should name the missing endpoint variable')
                ok = False
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # A transport failure must arrive as ProviderError - NOT a retryable type. The base
    # class retries TimeoutError/RateLimited, and HttpBackend already retries internally;
    # mapping to a retryable type would multiply the two layers.
    class _Boom(object):
        def generate(self, system, user, **kw):
            raise type('JudgeUnavailable', (Exception,), {})('simulated transport failure')

    bc = BackendJudgeClient(ClientConfig.from_env(provider='test', model='m'),
                            backend=_Boom())
    try:
        bc.judge_text('sys', {'x': 1})
        print('  [FAIL] BackendJudgeClient returned on a transport failure'); ok = False
    except ProviderError:
        pass
    except (TimeoutError, RateLimited):
        print('  [FAIL] transport failure mapped to a RETRYABLE type - this stacks '
              'HttpBackend\'s retries on top of judge_text\'s'); ok = False

    # An empty completion is a failure, not an answer.
    class _Empty(object):
        def generate(self, system, user, **kw):
            return type('R', (object,), {'text': '   '})()

    bc2 = BackendJudgeClient(ClientConfig.from_env(provider='test', model='m'),
                             backend=_Empty())
    try:
        bc2.judge_text('sys', {'x': 1})
        print('  [FAIL] empty completion accepted as an answer'); ok = False
    except ProviderError:
        pass

    # The payload must reach the backend as the user message, unchanged.
    seen = {}

    class _Echo(object):
        def generate(self, system, user, **kw):
            seen['system'] = system
            seen['user'] = user
            return type('R', (object,), {'text': '{"ok": 1}'})()

    bc3 = BackendJudgeClient(ClientConfig.from_env(provider='test', model='m'),
                             backend=_Echo())
    bc3.judge_text('SYSTEM', {'chosen': 'A', 'rejected': 'B'})
    if seen.get('system') != 'SYSTEM' or '"chosen"' not in seen.get('user', ''):
        print('  [FAIL] payload did not reach the backend intact: %r' % (seen,))
        ok = False

    # error hierarchy: all four are catchable as JudgeClientError
    for exc_cls in (TimeoutError, RateLimited, ProviderError, InvalidJSON):
        if not issubclass(exc_cls, JudgeClientError):
            print('  [FAIL] %s does not subclass JudgeClientError' % exc_cls.__name__)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import sys
    import argparse
    ap = argparse.ArgumentParser(description='LLM Judge provider/model client.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
