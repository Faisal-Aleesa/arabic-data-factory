# -*- coding: utf-8 -*-
"""Task 3, Group B - HTTP backend for the LLM judge. Provider-agnostic.

Implements `judge_client.JudgeBackend` over a plain HTTP POST, so any OpenAI-compatible
chat-completions endpoint or arbitrary JSON REST service can serve as the judge. Nothing
here is specific to one vendor: no vendor SDK, no vendor base URL, no vendor model id.

خلفية HTTP عامة للحَكَم: تعمل مع أي واجهة متوافقة مع OpenAI أو أي واجهة REST.

Why stdlib urllib and not a vendor SDK
--------------------------------------
The judging model will be a Qwen instance served over a hosted API, and the provider is
still being finalised. Hosted Qwen is commonly exposed OpenAI-compatibly (DashScope's
compatible mode, vLLM, Ollama, Together, OpenRouter, a self-hosted gateway), but not
always, and "OpenAI-compatible" is a family of near-misses rather than one wire format.

Binding to the `openai` package would pin the OpenAI request shape AND add a dependency
for what is, on the wire, one POST with a JSON body. `urllib.request` from the standard
library covers both the compatible case and a genuinely custom REST endpoint, keeps the
stack dependency-free, and leaves the team free to pick a provider without a code change.
If they later prefer a vendor SDK, it becomes another JudgeBackend beside this one - the
seam in `judge_client.py` is what makes that a local change.

Two adapters, and a third you write in a dict
---------------------------------------------
    OPENAI_CHAT   POST {base}/chat/completions with a messages array; reads
                  choices[0].message.content. Covers DashScope compatible mode, vLLM,
                  Ollama's OpenAI shim, Together, OpenRouter, LM Studio, and most
                  self-hosted gateways.
    ANTHROPIC_MSG kept only because the repo's own environment happens to have that SDK
                  installed; the wire shape differs (system is a top-level field, content
                  is a block list) and confusing the two is a silent empty-string bug.
    custom        any endpoint, by giving `text_path` / `usage_in_path` / `usage_out_path`
                  as key sequences into the response JSON.

Credentials never enter this file
---------------------------------
The API key is read from an environment variable BY NAME at call time. It is never a
constructor argument, never stored on the instance, never written to a transcript, and
`_redact()` strips it from every exception this module raises. A key that reaches a log
or a cached transcript is a leaked key, and this stack already writes transcripts to disk
on purpose.

Fail-closed, like everything else here
--------------------------------------
Every failure - connection refused, timeout, non-2xx, malformed envelope, an empty
completion, a refusal field, or `finish_reason: "length"` - raises `JudgeUnavailable`,
which the judge modules turn into ABSTAIN and the contract routes to HOLD_FOR_REVIEW.
Truncation matters especially: a response cut off at the token limit often still contains
parseable-looking JSON, and accepting it would let a half-formed verdict read as a real
one.

NOT CONFIGURED, DELIBERATELY
----------------------------
No default base_url, no default model, no default key variable. `from_env()` raises and
names exactly what is missing. Supplying those four values is the only step between this
module and a live run, and it belongs to whoever owns the endpoint.
"""

from __future__ import unicode_literals

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_client import JudgeBackend, JudgeResponse, JudgeUnavailable   # noqa: E402

try:                                  # pragma: no cover - trivial 2/3 split
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
except ImportError:                   # pragma: no cover
    from urllib2 import Request, urlopen, HTTPError, URLError


OPENAI_CHAT = 'openai_chat'
ANTHROPIC_MSG = 'anthropic_messages'
CUSTOM = 'custom'

# Environment variables read by from_env(). Names only - values are never read into
# module state, and the key variable's VALUE is never logged.
ENV_BASE_URL = 'JUDGE_API_BASE_URL'
ENV_MODEL = 'JUDGE_API_MODEL'
ENV_KEY_VAR = 'JUDGE_API_KEY_VAR'      # the NAME of the variable holding the key
ENV_DIALECT = 'JUDGE_API_DIALECT'      # openai_chat | anthropic_messages | custom


class EndpointConfig(object):
    """Everything needed to reach one judge endpoint. No defaults for the four essentials."""

    def __init__(self, base_url, model, api_key_var, dialect=OPENAI_CHAT,
                 path=None, auth_scheme='Bearer', auth_header='Authorization',
                 extra_headers=None, timeout_s=60.0, max_retries=2,
                 text_path=None, usage_in_path=None, usage_out_path=None,
                 extra_body=None):
        if not base_url:
            raise ValueError('base_url is required')
        if not model:
            raise ValueError('model is required')
        if not api_key_var:
            raise ValueError('api_key_var is required (the NAME of the env var, '
                             'not the key itself)')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key_var = api_key_var
        self.dialect = dialect
        self.path = path or _default_path(dialect)
        self.auth_scheme = auth_scheme
        self.auth_header = auth_header
        self.extra_headers = dict(extra_headers or {})
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.text_path = text_path
        self.usage_in_path = usage_in_path
        self.usage_out_path = usage_out_path
        self.extra_body = dict(extra_body or {})
        if dialect == CUSTOM and not text_path:
            raise ValueError('dialect="custom" requires text_path, e.g. '
                             '["output", "text"]')

    def url(self):
        return self.base_url + self.path

    def describe(self):
        """Safe to print: names the key variable, never reads it."""
        return {'base_url': self.base_url, 'model': self.model,
                'dialect': self.dialect, 'path': self.path,
                'api_key_var': self.api_key_var,
                'api_key_present': bool(os.environ.get(self.api_key_var)),
                'timeout_s': self.timeout_s, 'max_retries': self.max_retries}


def _default_path(dialect):
    if dialect == OPENAI_CHAT:
        return '/chat/completions'
    if dialect == ANTHROPIC_MSG:
        return '/v1/messages'
    return ''


def from_env():
    """Build a config from the environment, or raise naming exactly what is missing."""
    missing = [v for v in (ENV_BASE_URL, ENV_MODEL, ENV_KEY_VAR)
               if not os.environ.get(v)]
    if missing:
        raise JudgeUnavailable(
            'judge endpoint is not configured: %s not set. Required: %s (endpoint base '
            'URL), %s (model id), %s (the NAME of the env var holding the API key). '
            'Optional: %s (openai_chat | anthropic_messages | custom, default '
            'openai_chat).'
            % (', '.join(missing), ENV_BASE_URL, ENV_MODEL, ENV_KEY_VAR, ENV_DIALECT))
    key_var = os.environ[ENV_KEY_VAR]
    if not os.environ.get(key_var):
        raise JudgeUnavailable(
            '%s names %r, but %r is not set in the environment'
            % (ENV_KEY_VAR, key_var, key_var))
    return EndpointConfig(base_url=os.environ[ENV_BASE_URL],
                          model=os.environ[ENV_MODEL],
                          api_key_var=key_var,
                          dialect=os.environ.get(ENV_DIALECT, OPENAI_CHAT))


# ------------------------------------------------------------------- request / response

def build_body(cfg, system, user, max_tokens, temperature):
    """The JSON body for one call, in the configured dialect."""
    if cfg.dialect == OPENAI_CHAT:
        body = {'model': cfg.model,
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}],
                'max_tokens': max_tokens,
                'temperature': temperature}
    elif cfg.dialect == ANTHROPIC_MSG:
        # Different wire shape: system is top-level, not a message.
        body = {'model': cfg.model,
                'system': system,
                'messages': [{'role': 'user', 'content': user}],
                'max_tokens': max_tokens,
                'temperature': temperature}
    else:
        body = {'model': cfg.model,
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}],
                'max_tokens': max_tokens,
                'temperature': temperature}
    body.update(cfg.extra_body)
    return body


def _dig(obj, path):
    """Walk a key/index sequence into nested JSON; None if any hop is absent."""
    cur = obj
    for step in path:
        try:
            cur = cur[step]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def extract(cfg, payload):
    """(text, input_tokens, output_tokens) from a decoded response body.

    Raises JudgeUnavailable for envelopes that carry no usable completion - including a
    truncated one, which must never be parsed as if it were whole.
    """
    if not isinstance(payload, dict):
        raise JudgeUnavailable('response body is not a JSON object')

    # An error envelope returned with a 200, which several gateways do.
    err = payload.get('error')
    if err:
        raise JudgeUnavailable('endpoint returned an error envelope: %s'
                               % _clip(json.dumps(err, ensure_ascii=False)))

    if cfg.dialect == OPENAI_CHAT:
        choice = _dig(payload, ['choices', 0]) or {}
        finish = choice.get('finish_reason')
        msg = choice.get('message') or {}
        if msg.get('refusal'):
            raise JudgeUnavailable('model refused: %s' % _clip(str(msg['refusal'])))
        text = msg.get('content')
        if text is None:
            text = choice.get('text')          # legacy completions shape
        usage = payload.get('usage') or {}
        tin = usage.get('prompt_tokens', usage.get('input_tokens'))
        tout = usage.get('completion_tokens', usage.get('output_tokens'))
    elif cfg.dialect == ANTHROPIC_MSG:
        finish = payload.get('stop_reason')
        blocks = payload.get('content') or []
        text = ''.join(b.get('text', '') for b in blocks
                       if isinstance(b, dict) and b.get('type') == 'text')
        usage = payload.get('usage') or {}
        tin = usage.get('input_tokens')
        tout = usage.get('output_tokens')
    else:
        finish = None
        text = _dig(payload, cfg.text_path)
        tin = _dig(payload, cfg.usage_in_path) if cfg.usage_in_path else None
        tout = _dig(payload, cfg.usage_out_path) if cfg.usage_out_path else None

    # Truncation: a cut-off body often still looks parseable. Refuse it.
    if finish in ('length', 'max_tokens'):
        raise JudgeUnavailable(
            'completion was truncated (finish_reason=%r); a partial verdict must not be '
            'read as a whole one - raise max_tokens' % finish)
    if not text or not str(text).strip():
        raise JudgeUnavailable('response carried no completion text (finish_reason=%r)'
                               % finish)
    return str(text), tin, tout


def _clip(s, n=200):
    s = s or ''
    return s if len(s) <= n else s[:n] + '...'


def _urllib_transport(url, data, headers, timeout):
    req = Request(url, data=data, headers=headers)
    try:
        resp = urlopen(req, timeout=timeout)
    except HTTPError as exc:                     # non-2xx
        return exc.code, exc.read()
    except URLError as exc:
        raise JudgeUnavailable('connection failed: %s' % exc.reason)
    except Exception as exc:                     # socket timeout and friends
        raise JudgeUnavailable('transport error: %s' % exc)
    return getattr(resp, 'status', resp.getcode()), resp.read()


RETRYABLE = frozenset([408, 409, 425, 429, 500, 502, 503, 504])


class HttpBackend(JudgeBackend):
    """A judge over one HTTP endpoint. `transport` is injectable so tests run offline."""

    name = 'http'

    def __init__(self, config, transport=None, sleep=time.sleep):
        self.config = config
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self.model_id = config.model

    def _headers(self, key):
        h = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.config.auth_scheme:
            h[self.config.auth_header] = '%s %s' % (self.config.auth_scheme, key)
        else:
            h[self.config.auth_header] = key
        h.update(self.config.extra_headers)
        return h

    def _redact(self, text, key):
        """Strip the key from anything that might be surfaced or logged."""
        if not key or not text:
            return text
        return text.replace(key, '<REDACTED>')

    def generate(self, system, user, max_tokens=1024, temperature=0.0, probe_id=None):
        cfg = self.config
        key = os.environ.get(cfg.api_key_var)
        if not key:
            raise JudgeUnavailable(
                'no API key: environment variable %r is unset' % cfg.api_key_var)

        body = json.dumps(build_body(cfg, system, user, max_tokens, temperature),
                          ensure_ascii=False).encode('utf-8')
        headers = self._headers(key)

        last = None
        for attempt in range(cfg.max_retries + 1):
            started = time.time()
            try:
                status, raw = self._transport(cfg.url(), body, headers, cfg.timeout_s)
            except JudgeUnavailable as exc:
                last = JudgeUnavailable(self._redact(str(exc), key))
                if attempt < cfg.max_retries:
                    self._sleep(min(2 ** attempt, 8))
                    continue
                raise last
            latency = time.time() - started

            if status in RETRYABLE and attempt < cfg.max_retries:
                self._sleep(min(2 ** attempt, 8))
                continue
            if not (200 <= status < 300):
                snippet = self._redact(_clip(_decode(raw)), key)
                raise JudgeUnavailable('HTTP %s from the judge endpoint: %s'
                                       % (status, snippet))
            try:
                payload = json.loads(_decode(raw))
            except ValueError as exc:
                raise JudgeUnavailable(
                    'response was not JSON: %s' % self._redact(str(exc), key))
            try:
                text, tin, tout = extract(cfg, payload)
            except JudgeUnavailable as exc:
                raise JudgeUnavailable(self._redact(str(exc), key))
            return JudgeResponse(text, model_id=cfg.model, input_tokens=tin,
                                 output_tokens=tout, latency_s=latency)
        raise last or JudgeUnavailable('exhausted %d attempts' % (cfg.max_retries + 1))


def _decode(raw):
    if isinstance(raw, bytes):
        return raw.decode('utf-8', 'replace')
    return raw


# ------------------------------------------------------------------------------ self-test

def _fake(status, payload, record=None):
    """A transport returning a canned response, capturing what it was sent."""
    def _t(url, data, headers, timeout):
        if record is not None:
            record.append({'url': url, 'headers': headers, 'timeout': timeout,
                           'body': json.loads(_decode(data))})
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return status, body.encode('utf-8')
    return _t


def _cfg(**kw):
    base = dict(base_url='https://example.invalid/v1', model='qwen-test',
                api_key_var='JUDGE_TEST_KEY')
    base.update(kw)
    return EndpointConfig(**base)


def run_self_test():
    ok = True
    os.environ['JUDGE_TEST_KEY'] = 'sk-secret-value-123'
    KEY = os.environ['JUDGE_TEST_KEY']

    # --- OpenAI-compatible round trip ---------------------------------------------------
    seen = []
    be = HttpBackend(_cfg(), transport=_fake(200, {
        'choices': [{'message': {'content': '{"ok": true}'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 120, 'completion_tokens': 30}}, seen))
    r = be.generate('SYS', 'USER', max_tokens=256, temperature=0.0)
    if r.text != '{"ok": true}' or r.input_tokens != 120 or r.output_tokens != 30:
        print('  [FAIL] openai_chat round trip: %r %s/%s'
              % (r.text, r.input_tokens, r.output_tokens))
        ok = False
    sent = seen[0]
    if sent['url'] != 'https://example.invalid/v1/chat/completions':
        print('  [FAIL] wrong URL: %s' % sent['url'])
        ok = False
    msgs = sent['body']['messages']
    if msgs[0]['role'] != 'system' or msgs[0]['content'] != 'SYS' or \
            msgs[1]['role'] != 'user' or msgs[1]['content'] != 'USER':
        print('  [FAIL] openai messages array malformed: %s' % msgs)
        ok = False
    if sent['headers'].get('Authorization') != 'Bearer ' + KEY:
        print('  [FAIL] auth header not set as Bearer')
        ok = False

    # usage under the alternate key names some gateways use
    be = HttpBackend(_cfg(), transport=_fake(200, {
        'choices': [{'message': {'content': 'x'}, 'finish_reason': 'stop'}],
        'usage': {'input_tokens': 7, 'output_tokens': 3}}))
    r = be.generate('s', 'u')
    if (r.input_tokens, r.output_tokens) != (7, 3):
        print('  [FAIL] alternate usage key names not read: %s/%s'
              % (r.input_tokens, r.output_tokens))
        ok = False

    # --- the Anthropic shape is genuinely different -------------------------------------
    seen2 = []
    be = HttpBackend(_cfg(dialect=ANTHROPIC_MSG), transport=_fake(200, {
        'content': [{'type': 'text', 'text': '{"ok":1}'}], 'stop_reason': 'end_turn',
        'usage': {'input_tokens': 5, 'output_tokens': 2}}, seen2))
    r = be.generate('SYS', 'USER')
    if r.text != '{"ok":1}':
        print('  [FAIL] anthropic block extraction: %r' % r.text)
        ok = False
    b2 = seen2[0]['body']
    if b2.get('system') != 'SYS' or len(b2['messages']) != 1:
        print('  [FAIL] anthropic body shape: system must be top-level, one message')
        ok = False

    # --- custom REST via key paths ------------------------------------------------------
    be = HttpBackend(_cfg(dialect=CUSTOM, path='/generate',
                          text_path=['output', 'text'],
                          usage_in_path=['meta', 'in'], usage_out_path=['meta', 'out']),
                     transport=_fake(200, {'output': {'text': 'hello'},
                                           'meta': {'in': 11, 'out': 4}}))
    r = be.generate('s', 'u')
    if r.text != 'hello' or (r.input_tokens, r.output_tokens) != (11, 4):
        print('  [FAIL] custom path extraction: %r %s/%s'
              % (r.text, r.input_tokens, r.output_tokens))
        ok = False
    try:
        _cfg(dialect=CUSTOM)
        print('  [FAIL] custom dialect accepted without text_path')
        ok = False
    except ValueError:
        pass

    # --- every failure shape must raise, never return ------------------------------------
    fails = [
        ('truncated', 200, {'choices': [{'message': {'content': '{"partial":'},
                                         'finish_reason': 'length'}]}),
        ('empty completion', 200, {'choices': [{'message': {'content': ''},
                                                'finish_reason': 'stop'}]}),
        ('refusal', 200, {'choices': [{'message': {'refusal': 'cannot assist'},
                                       'finish_reason': 'stop'}]}),
        ('error envelope', 200, {'error': {'message': 'bad model'}}),
        ('not json', 200, '<html>gateway</html>'),
        ('not an object', 200, '[1,2,3]'),
        ('401', 401, {'error': 'unauthorized'}),
        ('400', 400, {'error': 'bad request'}),
    ]
    for label, status, payload in fails:
        be = HttpBackend(_cfg(max_retries=0), transport=_fake(status, payload))
        try:
            be.generate('s', 'u')
            print('  [FAIL] %s returned instead of raising' % label)
            ok = False
        except JudgeUnavailable:
            pass

    # Truncation must NOT be treated as a parseable answer - the whole point.
    be = HttpBackend(_cfg(max_retries=0), transport=_fake(200, {
        'choices': [{'message': {'content': '{"dimensions": {}, "confidence": "high"}'},
                     'finish_reason': 'length'}]}))
    try:
        be.generate('s', 'u')
        print('  [FAIL] a truncated but well-formed body was accepted')
        ok = False
    except JudgeUnavailable:
        pass

    # --- retries: retryable statuses retry, client errors do not -------------------------
    calls = {'n': 0}

    def flaky(url, data, headers, timeout):
        calls['n'] += 1
        if calls['n'] < 3:
            return 429, b'{"error":"rate limited"}'
        return 200, json.dumps({'choices': [{'message': {'content': 'ok'},
                                             'finish_reason': 'stop'}]}).encode()
    be = HttpBackend(_cfg(max_retries=2), transport=flaky, sleep=lambda s: None)
    if be.generate('s', 'u').text != 'ok' or calls['n'] != 3:
        print('  [FAIL] retry on 429 did not recover (calls=%d)' % calls['n'])
        ok = False

    calls4 = {'n': 0}

    def always400(url, data, headers, timeout):
        calls4['n'] += 1
        return 400, b'{"error":"bad"}'
    be = HttpBackend(_cfg(max_retries=2), transport=always400, sleep=lambda s: None)
    try:
        be.generate('s', 'u')
    except JudgeUnavailable:
        pass
    if calls4['n'] != 1:
        print('  [FAIL] a 400 was retried %d times; client errors must not retry'
              % calls4['n'])
        ok = False

    # --- the key must never appear in an exception ---------------------------------------
    def leaky(url, data, headers, timeout):
        return 401, ('{"error":"invalid key %s"}' % KEY).encode('utf-8')
    be = HttpBackend(_cfg(max_retries=0), transport=leaky)
    try:
        be.generate('s', 'u')
        print('  [FAIL] leaky transport did not raise')
        ok = False
    except JudgeUnavailable as exc:
        if KEY in str(exc):
            print('  [FAIL] THE API KEY LEAKED INTO AN EXCEPTION MESSAGE')
            ok = False
        if '<REDACTED>' not in str(exc):
            print('  [FAIL] redaction marker missing from the error')
            ok = False

    def leaky_conn(url, data, headers, timeout):
        raise JudgeUnavailable('connect to host with key %s failed' % KEY)
    be = HttpBackend(_cfg(max_retries=0), transport=leaky_conn, sleep=lambda s: None)
    try:
        be.generate('s', 'u')
    except JudgeUnavailable as exc:
        if KEY in str(exc):
            print('  [FAIL] the key leaked through a transport exception')
            ok = False

    # describe() must never carry the key itself
    d = _cfg().describe()
    if KEY in json.dumps(d) or d.get('api_key_present') is not True:
        print('  [FAIL] describe() leaked the key or misreported presence: %s' % d)
        ok = False

    # --- a missing key must abstain, not send an unauthenticated request -----------------
    del os.environ['JUDGE_TEST_KEY']
    sent_any = []
    be = HttpBackend(_cfg(), transport=_fake(200, {'choices': [
        {'message': {'content': 'x'}, 'finish_reason': 'stop'}]}, sent_any))
    try:
        be.generate('s', 'u')
        print('  [FAIL] generated with no API key set')
        ok = False
    except JudgeUnavailable:
        pass
    if sent_any:
        print('  [FAIL] an unauthenticated request was actually sent')
        ok = False

    # --- from_env() names what is missing rather than guessing ---------------------------
    for v in (ENV_BASE_URL, ENV_MODEL, ENV_KEY_VAR):
        os.environ.pop(v, None)
    try:
        from_env()
        print('  [FAIL] from_env() succeeded with nothing configured')
        ok = False
    except JudgeUnavailable as exc:
        for v in (ENV_BASE_URL, ENV_MODEL, ENV_KEY_VAR):
            if v not in str(exc):
                print('  [FAIL] from_env() error does not name %s' % v)
                ok = False

    # A key VARIABLE named but unset must be caught, not silently sent empty.
    os.environ[ENV_BASE_URL] = 'https://example.invalid/v1'
    os.environ[ENV_MODEL] = 'qwen-x'
    os.environ[ENV_KEY_VAR] = 'NOT_SET_ANYWHERE'
    try:
        from_env()
        print('  [FAIL] from_env() accepted a key var that is not set')
        ok = False
    except JudgeUnavailable:
        pass
    for v in (ENV_BASE_URL, ENV_MODEL, ENV_KEY_VAR):
        os.environ.pop(v, None)

    # --- no vendor SDK, and no hardcoded endpoint ----------------------------------------
    import io as _io
    import re as _re
    src = _io.open(__file__.replace('.pyc', '.py'), encoding='utf-8').read()
    for banned in ('import anthropic', 'import openai', 'from anthropic', 'from openai'):
        if _re.search(r'^\s*%s' % _re.escape(banned), src, _re.M):
            print('  [FAIL] vendor SDK import: %r' % banned)
            ok = False
    for host in ('dashscope', 'aliyuncs', 'api.openai.com', 'api.anthropic.com'):
        if _re.search(r'https?://[^\s\'"]*%s' % _re.escape(host), src):
            print('  [FAIL] a concrete provider endpoint is hardcoded: %r' % host)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='HTTP backend for the LLM judge.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--describe', action='store_true',
                    help='show the endpoint config from the environment, without the key')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    if a.describe:
        try:
            print(json.dumps(from_env().describe(), indent=2))
        except JudgeUnavailable as exc:
            sys.stderr.write('%s\n' % exc)
            sys.exit(2)
        sys.exit(0)
    ap.print_help()
