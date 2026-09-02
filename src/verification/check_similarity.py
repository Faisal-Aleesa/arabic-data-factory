# -*- coding: utf-8 -*-
"""Task 3 Group A item 3 - Semantic similarity between a response and its source chunk.

Input : a candidate response + its source_chunk_id
        data/processed/chunks.jsonl                  (--corpus saudi_dialect)
        data/processed/chunks_asas_albalagha.jsonl   (--corpus classical_lexicon)
Output: similarity scores per candidate; no pass/fail verdict (see THRESHOLDS below)

Why this exists alongside check_facts.py
----------------------------------------
check_facts.py matches specifics - years, page references, names, lemmas - exactly or
fuzzily. It is precise about what it checks and blind to everything else. A response can
pass it cleanly while still being badly grounded: assert no wrong year, name no invented
poet, and simply drift into vague paraphrase that the source does not support. Nothing in
a fact table catches that, because there is no fact to contradict.

This measures the other axis: how close the response is, in embedding space, to the text
it was supposedly generated from. The two checks are complements, not alternatives. A
response should pass both; failing either is a different kind of problem.

NOT deterministic in the pipeline's sense
-----------------------------------------
The README's "no LLM calls anywhere" applies to the Data Engineering stage. This module
is in the verification stage and does call a model. Embeddings are deterministic given
fixed weights in eval mode - the same input yields the same vector - but the scores are
model-dependent, and swapping the model changes every number. The model id is recorded in
every report for that reason.

Windowing, and the bug that made this measure worthless until it was fixed
--------------------------------------------------------------------------
Chunks are 200-800 whitespace tokens; the default encoder here accepts 128. Anything
longer is truncated INSIDE the model, silently, with no error raised.

The first version of this module used 900-character windows chosen by eye, on the
assumption of a 512-token limit. Arabic tokenizes at roughly 2.3 characters per token on
this model, so each 900-char window was 393 tokens and 67% of it was discarded before
being embedded. The resulting scores were not merely noisy, they were anti-correlated:
across the synthetic candidates the CORRECT chunk ranked in the bottom quartile of its
own corpus (288/337, 371/394), and a hallucinating response outscored a grounded one.

Windows are therefore built from the model's own tokenizer (`embedder.split`), never
from a character count, so the budget is right for whatever model is passed to --model.
The character windower survives only for the dependency-free baseline.

Each chunk is split into overlapping windows and the response compared against all:

  max_similarity   the best-matching window. This is the primary signal. A dialect chunk
                   bundles ~15 dictionary entries; a response about one of them SHOULD
                   score low against the other fourteen, and max is what asks "is this
                   grounded in some part of the source" rather than "does it summarise
                   the whole chunk".
  mean_similarity  average across windows. Lower by construction, and useful as a
                   contrast: a response tightly grounded in one entry shows a high max
                   with a low mean, while vague topical paraphrase shows both mid-range.

Read PERCENTILE, not raw cosine
-------------------------------
e5 compresses cosine into a narrow band - unrelated Arabic sits around 0.84, a verbatim
excerpt of the chunk itself around 0.93 - and the correct chunk can score BELOW the
corpus median while still being correct. Raw cosine is therefore not interpretable on
its own and no threshold should ever be set on it. --discriminate reports two derived
statistics, and the percentile is the operative one:

  margin      own max cosine minus the median over a random baseline of wrong chunks.
              Directionally right but tiny in absolute terms (+0.06 is a strong result).
  percentile  where the own-chunk score falls within that baseline. This is the number
              to reason about.

Observed behaviour on the content-bearing probes (tests/fixtures/similarity_*.jsonl,
n=6 per corpus, e5-small):

  group          saudi_dialect          classical_lexicon
  grounded       pct 99, 100            pct 100, 100, 99
  ambiguous      pct 20                 pct 96
  drift          pct 17                 pct 85
  hallucinated   pct 19, 1              pct 40

Grounded separates strongly and consistently on both corpora. Hallucinated sits well
below. Two honest caveats:

1. Drift and hallucination are NOT separable from each other. Both read as ungrounded,
   which is the correct outcome - this module answers "is this grounded", not "how did
   it fail". Use it with check_facts.py, which distinguishes the failure modes it can
   see.
2. A correct paraphrase with no shared wording is unreliable: the ambiguous probe scored
   pct 96 on classical but pct 20 on dialect, indistinguishable there from hallucination.
   So the measure is partly tracking lexical overlap, not purely semantics, and a
   low score is evidence of drift rather than proof of it. Treat low scores as "route to
   review", never as an automatic reject.

THRESHOLDS - deliberately not set
---------------------------------
No pass/fail cutoff is defined here and none should be guessed from n=6 hand-written
probes. Those probes establish that the measure orders cases sensibly and discriminates;
they say nothing about where a real cutoff sits, and caveat 2 above is a direct warning
against setting one tightly. Run --distribution on real reconstructed output and decide
then.

The oracle check behind all this: feeding a verbatim 40-word excerpt of a chunk back as
the response retrieves that chunk at rank 1 in 15 of 16 trials across both corpora
(top-10 in 16/16). If a change to this module breaks that, the change is wrong.
"""

import argparse
import hashlib
import random
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNKS = {
    'saudi_dialect':     'data/processed/chunks.jsonl',
    'classical_lexicon': 'data/processed/chunks_asas_albalagha.jsonl',
}
FIXTURES = {
    'saudi_dialect':     'tests/fixtures/candidates_saudi_dialect.jsonl',
    'classical_lexicon': 'tests/fixtures/candidates_classical_lexicon.jsonl',
}

# Multilingual RETRIEVAL encoder, 512-token context, small enough for CPU.
#
# The model class matters more than the model. Response-vs-chunk is an ASYMMETRIC
# retrieval problem - a short response against a long passage - and a symmetric
# paraphrase model is the wrong instrument for it. Measured on the same probes:
#
#   paraphrase-multilingual-MiniLM-L12-v2   classical median rank 379/394
#   intfloat/multilingual-e5-small          classical median rank  15/394
#
# e5 REQUIRES "query: " / "passage: " prefixes. Without them the inputs are out of
# distribution and the scores look plausible while meaning nothing - the same silent
# failure mode as the truncation bug. Prefixing is applied automatically for e5 models
# and must be added here for any other model family that expects it.
DEFAULT_MODEL = 'intfloat/multilingual-e5-small'

WINDOW_CHARS = 900          # ~200 whitespace tokens of Arabic, inside the encoder limit
WINDOW_STRIDE = 450         # 50% overlap so an entry is never split across every window


# ------------------------------------------------------------------------- embedders

class SentenceTransformerEmbedder(object):
    """The real backend. Imported lazily so the module loads without torch installed."""

    kind = 'sentence-transformers'

    def __init__(self, model_id=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self._m = SentenceTransformer(model_id)
        self._m.eval()
        self.max_tokens = int(self._m.max_seq_length)
        self._tok = self._m.tokenizer
        self.needs_prefix = 'e5' in model_id.lower()

    def _encode(self, texts, prefix):
        if self.needs_prefix:
            texts = [prefix + t for t in texts]
        v = self._m.encode(list(texts), convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def encode_query(self, texts):
        return self._encode(texts, 'query: ')

    def encode_passage(self, texts):
        return self._encode(texts, 'passage: ')

    # Back-compat for callers that do not distinguish the two sides.
    def encode(self, texts):
        return self.encode_passage(texts)

    def split(self, text):
        """Window by the model's OWN tokens, not by characters.

        This is not a refinement, it is a correctness fix. Character windows sized by
        eye silently exceed the encoder's limit and are truncated inside the model with
        no error: with 900-char windows on this model (max_seq_length 128) an Arabic
        window tokenized to 393 tokens and 67% of every window was discarded before it
        was ever embedded. Scores computed that way were close to meaningless - the correct
        chunk ranked in the bottom quartile of the corpus.

        Arabic tokenizes densely (~2.3 chars/token here), and the ratio differs per
        model and per script, so the budget is taken from the tokenizer rather than
        assumed.
        """
        ids = self._tok.encode(text, add_special_tokens=False)
        budget = max(16, self.max_tokens - 2)          # leave room for [CLS]/[SEP]
        if len(ids) <= budget:
            return [text]
        stride = max(1, budget // 2)                   # 50% overlap
        out = []
        for i in range(0, len(ids), stride):
            piece = ids[i:i + budget]
            if not piece:
                break
            out.append(self._tok.decode(piece, skip_special_tokens=True))
            if i + budget >= len(ids):
                break
        return out or [text]


class HashingEmbedder(object):
    """Dependency-free lexical baseline: hashed character 4-grams, L2-normalised.

    This is NOT a semantic model and must not be used to judge grounding. It exists so
    the pipeline, the windowing and the self-test can run on a machine without torch, and
    so CI can exercise the code path. Character n-grams do capture Arabic morphological
    overlap, which makes it a reasonable smoke test and a poor grounding check: it cannot
    see the paraphrase drift this module was built to catch.
    """

    kind = 'hashing-baseline'
    model_id = 'hashing-char4gram'

    def __init__(self, dim=512, n=4):
        self.dim, self.n = dim, n

    needs_prefix = False

    def split(self, text):
        return windows(text)

    def encode_query(self, texts):
        return self.encode(texts)

    def encode_passage(self, texts):
        return self.encode(texts)

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = ' '.join(str(t).split())
            for j in range(max(1, len(s) - self.n + 1)):
                g = s[j:j + self.n]
                h = int(hashlib.md5(g.encode('utf-8')).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_embedder(model_id, allow_fallback=False):
    try:
        return SentenceTransformerEmbedder(model_id)
    except ImportError:
        if not allow_fallback:
            raise
        sys.stderr.write(
            '[sim] sentence-transformers unavailable; falling back to the lexical\n'
            '      baseline. Scores are NOT semantic and must not be used to judge\n'
            '      grounding - see HashingEmbedder. Install with:\n'
            '        python -m pip install sentence-transformers\n')
        return HashingEmbedder()


# ------------------------------------------------------------ comparing percentiles
#
# A percentile is a RANK against a baseline of wrong chunks. That makes it meaningful
# only when at least one side is anchored to the source: between two responses that are
# both ungrounded, nothing ties either of them to the chunk and the ordering is noise.
#
# النسبة المئوية رتبة مقابل عينة من المقاطع الخاطئة، فلا معنى لها إلا إذا كان أحد
# الطرفين مسندا إلى المصدر. وبين نصين غير مسندين لا شيء يربط أيا منهما بالمقطع،
# فالترتيب ضجيج لا دليل.
#
# This was a real bug, found twice. Two format probes flipped verdict BETWEEN corpora on
# identical logic - one AUTO_CONFIRM here and FLAG_SUSPICIOUS there, and the reverse for
# the other probe - every flip driven by a percentile ordering between two ungrounded
# halves. It was then found to exist a second time in the SFT comparison path, because
# the fix had been applied at the caller rather than at the source.
#
# So groundedness is a REQUIRED argument of this function, not an optional guard a caller
# has to remember. A caller that has not decided whether its candidates are grounded
# cannot call it at all, and comparing `result['percentile']` values by hand is a visible
# departure from the module's own API rather than the path of least resistance.
#
# This module deliberately does NOT define what "grounded" means - that is a fact-layer
# judgement. verify_sft.is_grounded() supplies it.

PERCENTILE_TIE_MARGIN = 5.0        # points; below this the two are indistinguishable

A_BETTER = 'a_better'
B_BETTER = 'b_better'
TIE = 'tie'
INCOMPARABLE_UNGROUNDED = 'incomparable_ungrounded'


def compare_percentiles(a_pct, b_pct, a_grounded, b_grounded,
                        tie_margin=PERCENTILE_TIE_MARGIN):
    """Compare two similarity percentiles, refusing when neither side is grounded.

    `a_grounded` and `b_grounded` are REQUIRED positional arguments precisely so that
    the groundedness question cannot be skipped by a future caller.

    مقارنة نسبتي تشابه، مع الامتناع عن الحكم إذا لم يكن أي من الطرفين مسندا.
    وسيطا الإسناد إلزاميان تحديدا كي لا يتجاوزهما مستدعٍ لاحق.

    Returns one of: 'a_better', 'b_better', 'tie', 'incomparable_ungrounded'.
    """
    if not (a_grounded or b_grounded):
        return INCOMPARABLE_UNGROUNDED
    if a_pct is None or b_pct is None:
        return TIE
    if a_pct - b_pct > tie_margin:
        return A_BETTER
    if b_pct - a_pct > tie_margin:
        return B_BETTER
    return TIE


# -------------------------------------------------------------------------- scoring

def windows(text, size=WINDOW_CHARS, stride=WINDOW_STRIDE):
    s = ' '.join(str(text).split())
    if len(s) <= size:
        return [s]
    out = []
    for i in range(0, len(s), stride):
        w = s[i:i + size]
        if w.strip():
            out.append(w)
        if i + size >= len(s):
            break
    return out


def _summarise(sims, n):
    return {'max': float(np.max(sims)), 'mean': float(np.mean(sims)),
            'min': float(np.min(sims)), 'n_windows': n,
            'best_window': int(np.argmax(sims))}


def score(response, chunk_text, embedder):
    """Cosine similarity of the response against every window of the chunk.

    Simple uncached path, used by the self-test and by one-off --response calls.
    """
    wins = embedder.split(chunk_text)
    qv = embedder.encode_query([response])[0]
    pv = embedder.encode_passage(wins)
    return _summarise(pv @ qv, len(wins))     # all vectors are L2-normalised


class WindowIndex(object):
    """Caches window embeddings per chunk.

    --discriminate scores every candidate against the same set of wrong chunks, so
    without this the same windows are re-embedded once per candidate. On CPU that is the
    difference between seconds and minutes; the vectors are identical either way.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self._chunks = {}
        self._responses = {}

    def chunk_vecs(self, chunk_id, text):
        if chunk_id not in self._chunks:
            wins = self.embedder.split(text)
            self._chunks[chunk_id] = (len(wins), self.embedder.encode_passage(wins))
        return self._chunks[chunk_id]

    def response_vec(self, response):
        if response not in self._responses:
            self._responses[response] = self.embedder.encode_query([response])[0]
        return self._responses[response]

    def score(self, response, chunk_id, chunk_text):
        n, wv = self.chunk_vecs(chunk_id, chunk_text)
        return _summarise(wv @ self.response_vec(response), n)


def load_chunks(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            out[r['chunk_id']] = r
    return out


# ------------------------------------------------------------------------ self-test

def run_self_test():
    """Two layers, mirroring audit_staged_arabic.py.

    1. Backend-independent: windowing, normalisation and the cosine path are exercised
       with the deterministic hashing embedder, so this runs anywhere. Uses invented
       Arabic only - no source passages are hardcoded here.
    2. Live, if a real model is installed: confirms an identical string scores ~1.0 and
       that unrelated text scores materially lower. Skipped with a notice otherwise.
    """
    ok = True
    emb = HashingEmbedder()

    # windowing
    if len(windows('x' * 100)) != 1:
        print('  [FAIL] short text should be one window'); ok = False
    w = windows('y' * 3000)
    if len(w) < 3 or any(len(x) > WINDOW_CHARS for x in w):
        print('  [FAIL] long text windowing wrong: %d windows' % len(w)); ok = False

    # identity and ordering
    a = 'جملة تجريبية مخترعة للاختبار وحدها'
    b = 'نص مختلف تماما لا صلة له بالجملة الأولى إطلاقا'
    s_self = score(a, a, emb)
    s_other = score(a, b, emb)
    if s_self['max'] < 0.99:
        print('  [FAIL] identical text should score ~1.0, got %.3f' % s_self['max']); ok = False
    if not (s_self['max'] > s_other['max']):
        print('  [FAIL] identical should outscore unrelated (%.3f vs %.3f)'
              % (s_self['max'], s_other['max'])); ok = False

    # max >= mean >= min, and a response matching ONE window is not dragged to zero
    long_chunk = (b + ' ') * 30 + a
    s = score(a, long_chunk, emb)
    if not (s['max'] >= s['mean'] >= s['min'] - 1e-6):
        print('  [FAIL] max/mean/min ordering broken: %r' % s); ok = False
    if s['n_windows'] < 2:
        print('  [FAIL] expected multiple windows for a long chunk'); ok = False
    if s['max'] <= s['mean']:
        print('  [FAIL] a response matching one window should lift max above mean'); ok = False

    # determinism
    if score(a, long_chunk, emb) != s:
        print('  [FAIL] scoring is not deterministic'); ok = False

    # compare_percentiles: the guard lives here now, so it is tested here
    cmp_cases = [
        (90.0, 20.0, True,  True,  A_BETTER),
        (20.0, 90.0, True,  True,  B_BETTER),
        (70.0, 68.0, True,  True,  TIE),                 # inside the margin
        (90.0, 20.0, False, False, INCOMPARABLE_UNGROUNDED),
        (20.0, 90.0, False, False, INCOMPARABLE_UNGROUNDED),   # symmetric
        (90.0, 20.0, True,  False, A_BETTER),            # one grounded side is enough
        (90.0, 20.0, False, True,  A_BETTER),
        (None, 20.0, True,  True,  TIE),                 # a missing score is not a win
    ]
    # NB: deliberately not named a/b - the live layer below reuses those names, and an
    # earlier version of this loop shadowed them and fed None to the tokenizer.
    for pa, pb, ag, bg, want in cmp_cases:
        got = compare_percentiles(pa, pb, ag, bg)
        if got != want:
            print('  [FAIL] compare_percentiles(%s, %s, %s, %s) = %s, expected %s'
                  % (pa, pb, ag, bg, got, want))
            ok = False
    # groundedness must be positional and required - omitting it is a TypeError, which
    # is the whole point of the design
    try:
        compare_percentiles(90.0, 20.0)
    except TypeError:
        pass
    else:
        print('  [FAIL] groundedness args are not required'); ok = False

    # live layer
    try:
        real = SentenceTransformerEmbedder()
    except ImportError:
        print('  [skip] sentence-transformers not installed - live layer skipped')
    else:
        rs = score(a, a, real)
        ro = score(a, b, real)
        # Deliberately NOT an absolute floor. Retrieval encoders compress cosine into a
        # narrow band (e5 puts unrelated Arabic around 0.86), so a hardcoded 0.95 would
        # pass or fail on the model rather than on the code. Assert the ordering and a
        # minimum separation instead, which holds across model families.
        if rs['max'] - ro['max'] < 0.02:
            print('  [FAIL] live: identical (%.3f) barely separated from unrelated (%.3f)'
                  % (rs['max'], ro['max'])); ok = False
        if rs['max'] <= ro['max']:
            print('  [FAIL] live: identical (%.3f) did not beat unrelated (%.3f)'
                  % (rs['max'], ro['max'])); ok = False
        else:
            print('  live: identical %.3f vs unrelated %.3f (model=%s)'
                  % (rs['max'], ro['max'], real.model_id))

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------- reporting

def _pct(vals, p):
    return float(np.percentile(np.array(vals, dtype=np.float64), p)) if vals else float('nan')


def report_distribution(rows):
    print()
    print('=' * 78)
    print('SCORE DISTRIBUTION (no threshold applied - see module docstring)')
    print('=' * 78)
    if rows and 'margin' in rows[0]:
        print('  (margin = own max cosine minus the median over a random baseline of')
        print('   wrong chunks; positive means closer to its own source than to others)')
    groups = {}
    for r in rows:
        groups.setdefault(r.get('group', 'all'), []).append(r['max'])
    print('  %-34s %5s %6s %6s %6s %6s' % ('group', 'n', 'min', 'p50', 'max', 'mean'))
    for g, v in sorted(groups.items()):
        print('  %-34s %5d %6.3f %6.3f %6.3f %6.3f'
              % (g, len(v), min(v), _pct(v, 50), max(v), float(np.mean(v))))
    allv = [r['max'] for r in rows]
    if allv:
        print('  %-34s %5d %6.3f %6.3f %6.3f %6.3f'
              % ('ALL', len(allv), min(allv), _pct(allv, 50), max(allv), float(np.mean(allv))))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description='Semantic similarity between a candidate response and its chunk.')
    ap.add_argument('--corpus', choices=sorted(CHUNKS),
                    help='which corpus the chunk belongs to; required')
    ap.add_argument('--chunks', help='override chunks .jsonl')
    ap.add_argument('--model', default=DEFAULT_MODEL, help='sentence-transformers model id')
    ap.add_argument('--response', help='a single candidate response')
    ap.add_argument('--chunk-id', help='source_chunk_id for --response')
    ap.add_argument('--in', dest='inp', help='candidates .jsonl')
    ap.add_argument('--fixtures', action='store_true',
                    help='score the synthetic candidate fixtures for --corpus')
    ap.add_argument('--discriminate', type=int, default=0, metavar='N',
                    help='also score each candidate against N wrong chunks and report '
                         'the margin (0 = off)')
    ap.add_argument('--distribution', action='store_true',
                    help='print the score distribution summary')
    ap.add_argument('--allow-fallback', action='store_true',
                    help='use the lexical baseline if no model is installed (NOT semantic)')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    if not args.corpus:
        sys.stderr.write(
            '[sim] REFUSING to run: --corpus is required. The chunk id space differs\n'
            '    per corpus and a wrong lookup would silently score against the wrong\n'
            '    text.\n    corpora available:\n      '
            + '\n      '.join(sorted(CHUNKS)) + '\n')
        sys.exit(1)

    path = args.chunks or CHUNKS[args.corpus]
    if not os.path.exists(path):
        sys.stderr.write('[sim] chunks file not found: %s\n' % path)
        sys.exit(1)
    chunks = load_chunks(path)

    if args.response:
        if not args.chunk_id:
            sys.stderr.write('[sim] --response requires --chunk-id\n')
            sys.exit(1)
        cases = [{'response': args.response, 'source_chunk_id': args.chunk_id}]
    else:
        src = args.inp or (FIXTURES[args.corpus] if args.fixtures else None)
        if not src:
            sys.stderr.write('[sim] give --response/--chunk-id, --in, or --fixtures\n')
            sys.exit(1)
        if not os.path.exists(src):
            sys.stderr.write('[sim] candidates file not found: %s\n' % src)
            sys.exit(1)
        cases = [json.loads(l) for l in open(src, encoding='utf-8') if l.strip()]

    embedder = build_embedder(args.model, args.allow_fallback)
    print('[sim] corpus=%s  backend=%s  model=%s'
          % (args.corpus, embedder.kind, embedder.model_id))
    mt = getattr(embedder, 'max_tokens', None)
    print('[sim] windows: %s' % ('%d model tokens, 50%% overlap' % mt if mt
                                 else '%d chars, stride %d' % (WINDOW_CHARS, WINDOW_STRIDE)))
    print()

    index = WindowIndex(embedder)
    all_ids = sorted(chunks)
    baseline_ids = []
    if args.discriminate:
        rng = random.Random(0)                      # fixed seed: reproducible baseline
        baseline_ids = rng.sample(all_ids, min(args.discriminate, len(all_ids)))
    rows = []
    for case in cases:
        cid = case['source_chunk_id']
        rec = chunks.get(cid)
        if rec is None:
            print('  [UNKNOWN CHUNK] %s' % cid)
            continue
        s = index.score(case['response'], cid, rec['chunk_text'])
        row = {'label': case.get('label'), 'chunk_id': cid,
               'group': case.get('group') or (case.get('expect') or 'ungrouped'),
               'max': s['max'], 'mean': s['mean'], 'n_windows': s['n_windows']}

        if args.discriminate:
            # Reference distribution: the SAME randomly-sampled wrong chunks for every
            # candidate, so scores are comparable across the set and the window cache
            # stays effective. Random, not the first N sorted - sorting by chunk_id
            # groups by region, and the dialect corpus's first chunks are prose
            # introductions that any fluent sentence resembles, which biased an earlier
            # version of this comparison.
            wsc = [index.score(case['response'], w, chunks[w]['chunk_text'])['max']
                   for w in baseline_ids if w != cid]
            arr = np.array(wsc, dtype=np.float64)
            row['baseline_n'] = int(arr.size)
            row['baseline_median'] = float(np.median(arr))
            row['baseline_max'] = float(arr.max())
            # Raw cosine is not interpretable on its own: retrieval encoders compress
            # everything into a narrow band, and the correct chunk can sit BELOW the
            # corpus median while still being correct. Margin over the baseline median
            # and percentile within the baseline are what carry signal.
            row['margin'] = row['max'] - row['baseline_median']
            row['percentile'] = float((arr < row['max']).mean() * 100.0)

        rows.append(row)
        head = '  max=%.3f mean=%.3f (%d win)' % (s['max'], s['mean'], s['n_windows'])
        if args.discriminate:
            head += '   baseline med=%.3f   margin=%+.3f   pct=%.0f%%' % (
                row['baseline_median'], row['margin'], row['percentile'])
        print('%-58s' % (case.get('label') or cid)[:58])
        print(head)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.distribution:
        report_distribution(rows)


if __name__ == '__main__':
    main()
