# -*- coding: utf-8 -*-
"""Pre-commit gate: refuse to commit anything that looks like a live credential.

Companion to audit_staged_arabic.py and built on the same philosophy. That audit exists
because source passages migrate into code during debugging and code is not gitignored;
this one exists because credentials migrate into TEMPLATES for the same reason, and a
template is the one file that is guaranteed to be committed.

Why it exists - the incident
----------------------------
On 2026-09-11 a commit replaced the placeholder in `.env.example` with a 73-character
value in the exact format of a live OpenRouter key, and pushed it to the public remote.
Every guard in the repo passed it. The pre-commit hook gates symbol resolution and the
licence audit; the licence audit scans for Arabic source text and correctly reported
`.env.example` as "0 runs, clean", because it was never a secrets scanner. `.gitignore`
covers `.env` but deliberately not `.env.example`, because the example IS the template.
The class of leak was entirely unprotected, and it is the class where the damage is done
the moment the push lands - a key in public history is compromised whether or not a
later commit removes it.

What is scanned
---------------
Every staged file, read from the INDEX (the staged version, not the working tree - the
two differ and only the index is about to be committed). Binary files are skipped.

Two kinds of check, because they answer different questions:

  STRUCTURAL   a prefix that only a real key from a known provider has. These fire on
               ANY file, since a key is a key wherever it lands:
                   openrouter   sk-or-v1-<40+ alnum>
                   anthropic    sk-ant-<20+ [alnum _ -]>
                   openai       sk-<20+ alnum>          (only when neither prefix above matched)
                   aws_access   AKIA<16 [A-Z0-9]>
               The OpenAI pattern is checked LAST and only where the two more specific
               prefixes did not match, so one key is reported once under its real name.

  TEMPLATE     `.env*` files should contain placeholders and nothing else. Every
               `KEY=value` line is inspected, and a value is reported when ALL of:
                   - it is 20+ characters
                   - it is not placeholder-shaped (see PLACEHOLDER_WORDS)
                   - it has no `/`, `:`, or whitespace (URLs and model names have those,
                     keys do not)
                   - its Shannon entropy is >= ENTROPY_MIN bits/char

Why the template check is not "just entropy" - MEASURED before writing this
---------------------------------------------------------------------------
Entropy alone does not separate placeholders from keys on this repo's own values:

    value                              len   H (bits/char)
    replace_with_your_openrouter_key    32   3.74      <- the real placeholder
    google/gemini-2.5-flash-lite        28   3.89      <- a real config value, NOT a key
    http://localhost:8000/v1            24   3.66
    sk-or-v1-<64 hex>                   73   4.21      <- the leaked shape
    AKIA<16>                            20   3.88
    <64 hex>                            64   3.89

The model name scores HIGHER than a raw hex key. A threshold that catches hex would
flag `OPENROUTER_MODEL`; one that spares the model name would miss the key. So entropy is
the LAST gate, not the first: the placeholder-word exemption and the no-`/`-no-`:` rule
remove the legitimate config values before entropy is consulted at all, and the
structural patterns catch the named providers regardless. ENTROPY_MIN is set at 3.5
because every key shape measured sits above it and every surviving non-key value is
removed by the earlier gates. That reasoning is pinned in run_self_test().

Fail-closed, exactly as the licence audit
-----------------------------------------
    0   nothing that looks like a credential is staged   -> commit proceeds
    1   a match was found                                -> commit BLOCKED
    2   the git index could not be read                  -> commit BLOCKED

Exit 2 BLOCKS here, unlike the licence audit's exit 2. The licence audit's "no corpus"
case is the normal state of a fresh clone, where there is nothing to leak. There is no
analogous innocent case for "could not read the staged files": if this scanner cannot see
what is about to be committed, it has no basis to say the commit is safe, and a check
that ran on nothing must not look like a pass.

What is reported
----------------
File, line number, pattern name, and the match REDACTED to its first 8 characters plus
its length. The full value is never printed: the audit's own output goes to a terminal
and sometimes into a log, and reproducing a secret while refusing it would be absurd.

No self-test literal may look like a key
----------------------------------------
The hook runs this scanner on this file whenever it is committed. So every key-shaped
value the self-test uses is built at RUNTIME from a hash of a fixed seed, never written
as a literal, and the self-test's last case is the scanner reading its own source and
finding nothing. If that case ever fails, someone has pasted a real-looking key into
the tests.
"""

import argparse
import collections
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------- patterns

# Ordered: the most specific prefixes first, so a key is named by its real provider.
# `openai` is handled separately in scan_text, because its prefix is a prefix of the
# other two and must only fire where they did not.
STRUCTURAL = [
    ('openrouter', re.compile(r'sk-or-v1-[A-Za-z0-9]{40,}')),
    ('anthropic',  re.compile(r'sk-ant-[A-Za-z0-9_\-]{20,}')),
    ('aws_access', re.compile(r'(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])')),
]
OPENAI = ('openai', re.compile(r'(?<![A-Za-z0-9_\-])sk-[A-Za-z0-9]{20,}'))

# A value containing any of these is a placeholder, whatever else it looks like.
PLACEHOLDER_WORDS = ('your', 'replace', 'placeholder', 'example', 'changeme',
                     'change_me', 'todo', 'xxx', 'dummy', 'sample', 'insert', 'fill',
                     'put_', 'goes_here', 'here')

ENV_LINE = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')
TEMPLATE_MIN_LEN = 20
ENTROPY_MIN = 3.5

REDACT_KEEP = 8


# ----------------------------------------------------------------------- helpers

def shannon(s):
    if not s:
        return 0.0
    counts = collections.Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def is_placeholder(value):
    v = value.lower().strip('\'"')
    if not v:
        return True
    if v.startswith('<') and v.endswith('>'):
        return True
    if any(w in v for w in PLACEHOLDER_WORDS):
        return True
    if len(set(v)) <= 2:                 # xxxxxxxx, ........, 00000000
        return True
    return False


def looks_like_key(value):
    """Structural pre-filter for the template check: keys have no URL/path/spacing chars."""
    v = value.strip('\'"')
    if len(v) < TEMPLATE_MIN_LEN:
        return False
    if any(ch in v for ch in '/:\\ \t@'):
        return False
    return True


def redact(match):
    return '%s...(%d chars)' % (match[:REDACT_KEEP], len(match))


def is_env_template(path):
    return os.path.basename(path).startswith('.env')


# -------------------------------------------------------------------------- scan

def scan_text(text, path):
    """Findings for one file's content. Pure - no IO."""
    findings = []
    lines = text.split('\n')
    for lineno, line in enumerate(lines, 1):
        covered = []                      # spans already claimed by a specific pattern
        for name, rx in STRUCTURAL:
            for m in rx.finditer(line):
                findings.append({'file': path, 'line': lineno, 'pattern': name,
                                 'match': redact(m.group(0))})
                covered.append((m.start(), m.end()))
        for m in OPENAI[1].finditer(line):
            if any(a <= m.start() < b for a, b in covered):
                continue
            findings.append({'file': path, 'line': lineno, 'pattern': OPENAI[0],
                             'match': redact(m.group(0))})

    if is_env_template(path):
        for lineno, line in enumerate(lines, 1):
            if line.lstrip().startswith('#'):
                continue
            m = ENV_LINE.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            if is_placeholder(value) or not looks_like_key(value):
                continue
            h = shannon(value.strip('\'"'))
            if h >= ENTROPY_MIN:
                already = any(f['line'] == lineno for f in findings)
                if not already:
                    findings.append({'file': path, 'line': lineno,
                                     'pattern': 'template_high_entropy',
                                     'match': redact(value),
                                     'detail': '%s= has entropy %.2f bits/char over %d chars; '
                                               '.env templates must hold placeholders only'
                                               % (key, h, len(value))})
    return findings


# ------------------------------------------------------------------- git plumbing

def staged_files():
    out = subprocess.run(['git', 'diff', '--cached', '--name-only',
                          '--diff-filter=ACMR'], cwd=REPO_ROOT,
                         capture_output=True, check=True).stdout.decode('utf-8')
    return [f for f in out.split('\n') if f.strip()]


def staged_content(path):
    """The INDEX version - what is actually about to be committed."""
    raw = subprocess.run(['git', 'show', ':' + path], cwd=REPO_ROOT,
                         capture_output=True, check=True).stdout
    if b'\x00' in raw[:8000]:
        return None                       # binary
    return raw.decode('utf-8', errors='replace')


def working_content(path):
    with open(os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path,
              'rb') as fh:
        raw = fh.read()
    if b'\x00' in raw[:8000]:
        return None
    return raw.decode('utf-8', errors='replace')


# --------------------------------------------------------------------- self-test

def _fake(prefix, seed, n, alphabet=None):
    """A key-SHAPED value built at runtime. Never a literal in this file."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    body = h if alphabet is None else ''.join(alphabet[int(c, 16) % len(alphabet)] for c in h)
    while len(body) < n:
        body += body
    return prefix + body[:n]


def run_self_test():
    ok = True

    def expect(text, path, want_patterns, why):
        nonlocal ok
        got = [f['pattern'] for f in scan_text(text, path)]
        if sorted(got) != sorted(want_patterns):
            print('  [FAIL] %s\n         got %r, expected %r' % (why, got, want_patterns))
            ok = False

    # --- structural patterns MUST fire, on any file, and be named correctly ---------
    expect('KEY=' + _fake('sk-or-v1-', 'a', 64), 'src/x.py', ['openrouter'],
           'openrouter key in a .py file')
    expect('token = "%s"' % _fake('sk-ant-api03-', 'b', 48), 'notebooks/n.ipynb',
           ['anthropic'], 'anthropic key in a notebook')
    expect('OPENAI_API_KEY=' + _fake('sk-', 'c', 48), 'config.txt', ['openai'],
           'openai-style key')
    expect('aws_access_key_id = ' + _fake('AKIA', 'd', 16, '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'),
           'deploy.sh', ['aws_access'], 'AWS access key id')
    # the leak's exact shape: a 73-char sk-or-v1- value in .env.example
    expect('OPENROUTER_API_KEY=' + _fake('sk-or-v1-', 'e', 64), '.env.example',
           ['openrouter'], 'the 2026-09-11 incident shape')

    # --- DISCRIMINATION: one key, one report, under its most specific name ----------
    # sk-or-v1-... also matches the generic sk- pattern; it must be reported ONCE.
    got = scan_text(_fake('sk-or-v1-', 'f', 64), 'a.txt', )
    if len(got) != 1 or got[0]['pattern'] != 'openrouter':
        print('  [FAIL] an openrouter key was double-reported or misnamed: %r'
              % [g['pattern'] for g in got]); ok = False
    got = scan_text(_fake('sk-ant-', 'g', 40), 'a.txt')
    if len(got) != 1 or got[0]['pattern'] != 'anthropic':
        print('  [FAIL] an anthropic key was double-reported or misnamed: %r'
              % [g['pattern'] for g in got]); ok = False

    # --- DISCRIMINATION: genuine placeholders must NOT fire, anywhere -----------------
    for ph in ('replace_with_your_openrouter_key', 'your_api_key_here', '<your-key>',
               'CHANGEME', 'sk-...', 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'your-openai-key',
               'example_key_123', 'placeholder', 'TODO', 'put_your_key_here', ''):
        expect('OPENROUTER_API_KEY=%s' % ph, '.env.example', [],
               'placeholder %r must not fire in a template' % ph)

    # --- DISCRIMINATION: legitimate non-key config values in .env.example ------------
    # These are the values the entropy measurement showed would be false positives
    # under a naive threshold; the structural pre-filter must remove them.
    for k, v in (('OPENROUTER_MODEL', 'google/gemini-2.5-flash-lite'),
                 ('BASE_URL', 'http://localhost:8000/v1'),
                 ('SOMETHING', 'C:\\Program Files\\Tool\\bin'),
                 ('EMAIL', 'someone@example.org'),
                 ('SHORT', 'abc123'),                     # under the length floor
                 ('DESCRIPTION', 'a sentence with spaces in it')):
        expect('%s=%s' % (k, v), '.env.example', [],
               'config value %s=%s must not fire' % (k, v))

    # --- the template check MUST fire on a high-entropy non-placeholder --------------
    expect('SOME_TOKEN=' + _fake('', 'h', 40), '.env.example',
           ['template_high_entropy'], 'raw hex secret in a template')
    expect('SOME_TOKEN=' + _fake('', 'i', 44, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+='),
           '.env.example', ['template_high_entropy'], 'base64-shaped secret in a template')
    # ...but the template check is SCOPED to .env* files: the same line in a .py file
    # is a hash or an id, not a leaked credential, and must not fire
    expect('SOME_TOKEN=' + _fake('', 'h', 40), 'src/x.py', [],
           'a hex value in ordinary code is not a template violation')
    # comments in a template are not values
    expect('# OPENROUTER_API_KEY=' + _fake('', 'j', 40), '.env.example', [],
           'a commented-out template line is not scanned for entropy')

    # --- entropy pins: the measured reasons the design is layered ---------------------
    # (1) real key shapes clear ENTROPY_MIN, so the entropy gate lets them through to be
    #     flagged. Hex caps at 4.0 bits/char (16 symbols) and measures ~3.9.
    hex64 = _fake('', 'k', 64)
    if shannon(hex64) < ENTROPY_MIN:
        print('  [FAIL] a 64-hex key scores %.2f, under ENTROPY_MIN=%.2f - real keys '
              'would slip through the entropy gate' % (shannon(hex64), ENTROPY_MIN))
        ok = False
    # (2) THE PLACEHOLDER ALSO CLEARS ENTROPY_MIN. This is the whole reason entropy is
    #     the last gate and not the first: on entropy alone the repo's own placeholder
    #     would be flagged. It is is_placeholder() that spares it. If this pin fails
    #     because someone raised ENTROPY_MIN above 3.74, the word exemption has become
    #     dead code and the docstring's argument is stale.
    ph = 'replace_with_your_openrouter_key'
    if not (shannon(ph) >= ENTROPY_MIN and is_placeholder(ph)):
        print('  [FAIL] the placeholder-word gate is no longer load-bearing '
              '(H=%.2f, is_placeholder=%s)' % (shannon(ph), is_placeholder(ph)))
        ok = False
    # (3) the model name scores above the hex key, so a threshold cannot separate them;
    #     it is looks_like_key() (the '/' rule) that spares it.
    model = 'google/gemini-2.5-flash-lite'
    if not (shannon(model) >= shannon(hex64) - 0.05 and not looks_like_key(model)):
        print('  [FAIL] the structural pre-filter is no longer what spares model names '
              '(H model=%.2f, H hex=%.2f, looks_like_key=%s)'
              % (shannon(model), shannon(hex64), looks_like_key(model)))
        ok = False

    # --- redaction: the full match never appears in output ----------------------------
    key = _fake('sk-or-v1-', 'l', 64)
    f = scan_text('K=' + key, 'z.txt')[0]
    if key in json.dumps(f) or len(f['match']) > REDACT_KEEP + 20:
        print('  [FAIL] a finding reproduced the secret it refused'); ok = False

    # --- binary content is skipped, not crashed on -----------------------------------
    # (exercised through the content readers; scan_text itself is text-only)

    # --- SELF-CONSISTENCY: this file, committed, must pass its own scanner -----------
    here = os.path.abspath(__file__)
    own = scan_text(io.open(here, encoding='utf-8').read(),
                    os.path.relpath(here, REPO_ROOT).replace('\\', '/'))
    if own:
        print('  [FAIL] this file trips its own scanner - a key-shaped literal has been '
              'pasted into it: %r' % [(f['line'], f['pattern']) for f in own])
        ok = False

    print('passed: %s' % ok)
    return ok


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description='Refuse to commit anything that looks like a live credential.')
    ap.add_argument('--files', nargs='*', help='scan these paths instead of the git index')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    findings = []
    try:
        if args.files:
            targets = [(p, working_content(p)) for p in args.files]
            source = 'working tree'
        else:
            targets = [(p, staged_content(p)) for p in staged_files()]
            source = 'git index (staged)'
    except (subprocess.CalledProcessError, OSError) as exc:
        # FAIL CLOSED. Not being able to see the staged content is not evidence that
        # the staged content is safe.
        print('[secrets] could not read %s: %s' % ('files' if args.files else 'the git index', exc))
        print('[secrets] EXIT 2 - nothing was checked, and that is not a pass.')
        sys.exit(2)

    scanned = 0
    for path, text in targets:
        if text is None:
            continue                      # binary
        scanned += 1
        findings += scan_text(text, path)

    if args.json:
        print(json.dumps({'source': source, 'files_scanned': scanned,
                          'findings': findings}, indent=2))
    else:
        print('[secrets] %d file(s) from %s' % (scanned, source))
        for f in findings:
            print('  %s:%d  %-22s %s' % (f['file'], f['line'], f['pattern'], f['match']))
            if f.get('detail'):
                print('      %s' % f['detail'])
        if findings:
            print('\n[secrets] FAIL - %d value(s) that look like credentials are staged.'
                  % len(findings))
            print('          Replace each with a placeholder. If one is a real key, it is')
            print('          already compromised the moment it is pushed: REVOKE it at the')
            print('          provider - removing it from the file does not un-leak it.')
        else:
            print('[secrets] PASS - nothing credential-shaped in the staged content.')

    sys.exit(1 if findings else 0)


if __name__ == '__main__':
    main()
