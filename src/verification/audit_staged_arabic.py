# -*- coding: utf-8 -*-
"""Licence containment audit - staged Arabic vs rights-pending source text.

REQUIRED before committing anything that touches dialect-derived content. See the
README's "Before you commit" section.

    git add ...
    python src/verification/audit_staged_arabic.py     # exit 0 = clean, 1 = needs triage

What problem this solves
------------------------
This project publishes the corpus it builds, not the sources it builds from, and the
dialect source's publication rights are still unconfirmed. The failure mode is not
committing `chunks.jsonl` by accident - `.gitignore` handles that. It is the quiet one:
verbatim source passages that migrate into CODE while you work. Docstring examples, a
regex you tuned against a real line, a self-test fixture pasted from a chunk you were
debugging. Those files are not gitignored, so they go public.

This has happened. Auditing `extract_facts.py` before its first commit found ten verbatim
dialect passages sitting in its docstrings and self-test cases - a definition, two
section headings, a footnote and two real poet names - all pasted in during debugging and
all about to be pushed to a public repo. They were replaced with invented placeholders.
That audit was hand-rolled three times before becoming this script.

How it classifies
-----------------
Every multi-word Arabic run in the staged content is checked against two populations:

  HELD    text whose publication rights are unconfirmed (the dialect corpus)
  PUBLIC  text already lawfully public (the classical corpus, CC BY-SA)

  HELD_ONLY       verbatim in a rights-pending source and in no public source.
                  This is the real finding. Replace it with an invented placeholder.
  ALSO_IN_PUBLIC  present in both. Usually a generic connective ("قال له") or shared
                  classical material. Review, but rarely a problem.
  (allowlisted)   previously triaged and recorded in the allowlist file; not reported.

Classification is data-driven rather than a hardcoded list of "safe" strings, because a
hardcoded list goes stale the moment a corpus is added. Occurrence counts are printed
alongside each hit: a phrase appearing hundreds of times is a connective, one appearing
once or twice is distinctive and worth a closer look.

Deliberately conservative
-------------------------
Runs of a single word are not reported. A lone word - a place name, a headword, a root -
is not meaningfully "source text" and reporting them buried the real findings in noise.
This means a single hallucinated-looking lemma in a docstring will pass; that is an
accepted limit, not an oversight.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..'))
ALLOWLIST = 'docs/arabic_audit_allowlist.txt'

# Rights-pending sources. Each entry: (label, on-disk path, git fallback ref:path).
# The fallback matters because these files are gitignored and local-only - on a fresh
# clone the on-disk copy does not exist, and an audit that silently finds nothing to
# check against is worse than one that fails loudly.
HELD_SOURCES = [
    ('saudi_dialect', 'data/processed/chunks.jsonl',
     'held/dialect-corpus-release:data/processed/chunks.jsonl'),
]
PUBLIC_SOURCES = [
    ('classical_lexicon', 'data/processed/chunks_asas_albalagha.jsonl', None),
]

ARABIC = r'؀-ۿ'
_DIA = re.compile(r'[ً-ْٰ]')
RUN = re.compile(r'[' + ARABIC + r'][' + ARABIC + r'\s]{4,}')
MIN_WORDS = 2


def norm(s):
    """Normalise for comparison: drop diacritics, strip non-Arabic, collapse spaces."""
    s = _DIA.sub('', s)
    s = re.sub(r'[^' + ARABIC + r'\s]', ' ', s)
    return ' '.join(s.split())


# ------------------------------------------------------------------ corpus loading

def _load_corpus(path, fallback):
    """Return normalised text of a corpus, from disk or from a git ref."""
    raw = None
    full = os.path.join(REPO_ROOT, path)
    if os.path.exists(full):
        raw = io.open(full, encoding='utf-8', errors='replace').read()
    elif fallback:
        try:
            raw = subprocess.run(['git', 'show', fallback], cwd=REPO_ROOT,
                                 capture_output=True, check=True).stdout.decode('utf-8')
        except subprocess.CalledProcessError:
            return None
    if raw is None:
        return None
    parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts.append(json.loads(line).get('chunk_text', ''))
        except ValueError:
            parts.append(line)
    return norm(' '.join(parts))


def load_populations():
    held, public, missing = OrderedDict(), OrderedDict(), []
    for label, path, fb in HELD_SOURCES:
        t = _load_corpus(path, fb)
        if t is None:
            missing.append((label, path))
        else:
            held[label] = t
    for label, path, fb in PUBLIC_SOURCES:
        t = _load_corpus(path, fb)
        if t is not None:
            public[label] = t
    return held, public, missing


# ------------------------------------------------------------------ staged content

def staged_files():
    out = subprocess.run(['git', 'diff', '--cached', '--name-only',
                          '--diff-filter=ACMR'], cwd=REPO_ROOT,
                         capture_output=True, check=True).stdout.decode('utf-8')
    return [f for f in out.split('\n') if f.strip()]


def staged_content(path):
    """Read the STAGED version (the index), not the working tree - that is what is
    actually about to be committed, and the two can differ."""
    try:
        return subprocess.run(['git', 'show', ':' + path], cwd=REPO_ROOT,
                              capture_output=True, check=True).stdout.decode(
                                  'utf-8', errors='replace')
    except subprocess.CalledProcessError:
        return ''


def load_allowlist():
    p = os.path.join(REPO_ROOT, ALLOWLIST)
    if not os.path.exists(p):
        return set()
    out = set()
    for line in io.open(p, encoding='utf-8'):
        line = line.split('#')[0].strip()
        if line:
            out.add(norm(line))
    return out


# ------------------------------------------------------------------------ the scan

def arabic_runs(text, min_words=MIN_WORDS):
    runs = set()
    for m in RUN.finditer(text):
        t = norm(m.group(0))
        if len(t.split()) >= min_words:
            runs.add(t)
    return runs


def scan_text(text, held, public, allow=frozenset(), min_words=MIN_WORDS):
    """Core check. Returns a list of findings, worst first.

    Kept free of git and file IO so it can be self-tested directly.
    """
    findings = []
    for run in sorted(arabic_runs(text, min_words), key=len, reverse=True):
        if run in allow:
            continue
        hit_held = [(lbl, t.count(run)) for lbl, t in held.items() if run in t]
        if not hit_held:
            continue
        hit_pub = [(lbl, t.count(run)) for lbl, t in public.items() if run in t]
        findings.append({
            'run': run,
            'words': len(run.split()),
            'held': hit_held,
            'public': hit_pub,
            'verdict': 'ALSO_IN_PUBLIC' if hit_pub else 'HELD_ONLY',
        })
    findings.sort(key=lambda f: (f['verdict'] != 'HELD_ONLY', -f['words']))
    return findings


# ------------------------------------------------------------------------ self-test

def run_self_test():
    """Two layers.

    1. Synthetic: a fabricated held corpus and a fabricated staged file that share a
       phrase. Proves the detector fires, and that it stays quiet on unrelated text and
       on allowlisted text. Uses invented Arabic only - a self-test that hardcoded a real
       held passage would itself be the leak this script exists to prevent.

    2. Live, if the real dialect corpus is reachable: a multi-word phrase is SAMPLED
       from the corpus at run time and fed in as though it were staged. This re-verifies
       the script still catches exactly the class of leak it found in extract_facts.py's
       debugging artifacts, without any real text being written into this file.
    """
    ok = True
    held = {'fake_held': norm('الجملة التجريبية المحجوزة للاختبار وكلام آخر')}
    public = {'fake_public': norm('عبارة عامة مشتركة بين المصدرين')}

    cases = [
        ('الجملة التجريبية المحجوزة', 1, 'HELD_ONLY', 'verbatim held phrase must fire'),
        ('نص مخترع لا صلة له بشيء', 0, None, 'unrelated text must stay quiet'),
        ('عبارة عامة مشتركة', 0, None, 'phrase only in the public corpus is not a hit'),
        ('كلمة', 0, None, 'single words are never reported'),
    ]
    for text, n, verdict, note in cases:
        f = scan_text(text, held, public)
        if len(f) != n:
            print('  [FAIL] %s: expected %d finding(s), got %d' % (note, n, len(f)))
            ok = False
        elif n and f[0]['verdict'] != verdict:
            print('  [FAIL] %s: expected %s, got %s' % (note, verdict, f[0]['verdict']))
            ok = False

    # a phrase present in BOTH populations downgrades to ALSO_IN_PUBLIC
    both = {'fake_held': norm('عبارة عامة مشتركة بين المصدرين')}
    f = scan_text('عبارة عامة مشتركة', both, public)
    if not f or f[0]['verdict'] != 'ALSO_IN_PUBLIC':
        print('  [FAIL] shared phrase should classify as ALSO_IN_PUBLIC')
        ok = False

    # allowlist suppresses a known-triaged phrase
    if scan_text('الجملة التجريبية المحجوزة', held, public,
                 allow={norm('الجملة التجريبية المحجوزة')}):
        print('  [FAIL] allowlisted phrase should not be reported')
        ok = False

    # ---- layer 2: live regression against the real held corpus, sampled at run time
    real_held, real_public, missing = load_populations()
    if not real_held:
        print('  [warn] held corpus unavailable (%s) - live regression skipped'
              % ', '.join('%s:%s' % m for m in missing))
    else:
        label, text = next(iter(real_held.items()))
        words = text.split()
        sample = None
        for i in range(200, min(len(words), 4000)):
            cand = ' '.join(words[i:i + 4])
            if len(cand.split()) == 4 and text.count(cand) == 1:
                sample = cand
                break
        if sample is None:
            print('  [warn] could not sample a distinctive phrase - live check skipped')
        else:
            f = scan_text(sample, real_held, real_public)
            if not f or f[0]['verdict'] != 'HELD_ONLY':
                print('  [FAIL] live regression: a real held passage was NOT caught')
                ok = False
            else:
                print('  live regression: sampled a 4-word phrase from %s -> caught as %s'
                      % (label, f[0]['verdict']))
            # control: the same phrase, allowlisted, must fall silent
            if scan_text(sample, real_held, real_public, allow={sample}):
                print('  [FAIL] live regression: allowlist did not suppress')
                ok = False

    print('passed: %s' % ok)
    return ok


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description='Audit staged Arabic text against rights-pending corpora.')
    ap.add_argument('--files', nargs='*',
                    help='audit these paths instead of the git index')
    ap.add_argument('--min-words', type=int, default=MIN_WORDS,
                    help='minimum words in a run before it is reported (default 2)')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    ap.add_argument('--self-test', action='store_true',
                    help='verify the detector and exit')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    held, public, missing = load_populations()
    if missing and not held:
        sys.stderr.write(
            '[audit] REFUSING to pass: no rights-pending corpus could be loaded\n'
            '    (%s).\n'
            '    An audit with nothing to compare against would report "clean" for any\n'
            '    input, which is worse than not running it. Restore the corpus or check\n'
            '    out the held branch, then re-run.\n'
            % ', '.join('%s -> %s' % m for m in missing))
        sys.exit(2)

    allow = load_allowlist()
    if args.files is not None:
        targets = [(f, io.open(f, encoding='utf-8', errors='replace').read())
                   for f in args.files if os.path.exists(f)]
        source = 'working tree'
    else:
        targets = [(f, staged_content(f)) for f in staged_files()]
        source = 'git index (staged)'

    if not targets:
        print('[audit] nothing to audit (%s)' % source)
        sys.exit(0)

    print('[audit] %d file(s) from %s' % (len(targets), source))
    print('[audit] held: %s   public: %s   allowlisted: %d'
          % (', '.join(held) or 'none', ', '.join(public) or 'none', len(allow)))
    print()

    report, worst = [], 0
    for path, text in targets:
        findings = scan_text(text, held, public, allow, args.min_words)
        runs = len(arabic_runs(text, args.min_words))
        held_only = [f for f in findings if f['verdict'] == 'HELD_ONLY']
        worst = max(worst, 1 if held_only else 0)
        report.append({'file': path, 'runs': runs, 'findings': findings})
        status = ('%d HELD_ONLY' % len(held_only)) if held_only else (
            '%d to review' % len(findings) if findings else 'clean')
        print('  %-58s %3d runs  %s' % (path, runs, status))
        for f in findings:
            counts = ' '.join('%s x%d' % (l, c) for l, c in f['held'])
            pub = ' '.join('%s x%d' % (l, c) for l, c in f['public'])
            print('     %-14s %s' % (f['verdict'], f['run'][:60]))
            print('     %-14s held: %s%s' % ('', counts, ('   public: ' + pub) if pub else ''))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    print()
    if worst:
        print('[audit] FAIL - verbatim rights-pending text is staged. Replace each')
        print('        HELD_ONLY run with an invented placeholder, or add it to')
        print('        %s if triage shows it is generic.' % ALLOWLIST)
        sys.exit(1)
    print('[audit] PASS - no rights-pending source text in the staged content.')
    sys.exit(0)


if __name__ == '__main__':
    main()
