# -*- coding: utf-8 -*-
"""Task 3 Group D item 14 - Diversity analysis over a set of SFT records.

Input : an SFT records .jsonl (instruction, response, source_chunk_id, source_region,
        format_type, model_version)
Output: a markdown report - phrasing variety, region spread, format spread

شرح بالعربية
------------
قياس تنوّع مجموعة سجلات التدريب: تنوّع صياغة التعليمات، وتوزيع المناطق، وتوزيع
أنواع الشكل. يمتد هذا على أدوات التحليل الاستكشافي القائمة في eda.py ولا يبدأ من
الصفر: يعيد استعمال md_table وpercentile وrtl وقائمة المناطق منها.

المقاييس الأساسية لا تحتاج أي نموذج - إحصاء n-gram فقط - وطبقة التضمين اختيارية.

Extends eda.py rather than starting fresh
-----------------------------------------
`md_table`, `percentile`, `rtl` and `DIALECT_REGIONS` are imported from
data_engineering/eda.py, so the report renders the same way the corpus EDA does and the
region vocabulary cannot drift between the two.

Which metrics, and why these
----------------------------
Checked what is actually installed before choosing (requirements-verification.txt plus
what sentence-transformers pulls in): numpy, scipy, scikit-learn, torch and
sentence-transformers are all present. So embedding clustering was available - and is
still not the primary metric.

PRIMARY, dependency-free, deterministic:

  distinct-n (n=1..4)     unique n-grams / total n-grams over all instructions. The
                          standard lexical-diversity measure. Falls as phrasing repeats.
  nearest-neighbour       for each instruction, the highest difflib token-sequence ratio
  similarity              against any OTHER instruction. This is the one that catches
                          templating directly. Trigram Jaccard was tried first and MISSED
                          it - on 8-token instructions a single noun swap destroys 3 of 6
                          trigrams, scoring 0.333 on an unmistakably templated set. See
                          nearest_neighbour_similarity().
  near-template rate      share of instructions whose nearest neighbour is above the
                          band. Reported ALONGSIDE the mean because the mean dilutes: a
                          batch that is half templated averages below the band while the
                          rate still reports 50%. Real batches are mixed, so the rate is
                          the metric that catches a templated SUBSET.
  template share          the most common leading 3-gram, as a share of the set. A high
                          value means most instructions open the same way.

OPTIONAL, embedding layer (--embed):

  mean pairwise cosine and a near-duplicate rate over sentence embeddings.

The n-gram metrics are primary deliberately. They are deterministic, need no model, run
in CI, and are not sensitive to which encoder is installed - the same reasoning the rest
of the pipeline uses. The embedding layer adds a semantic view that n-grams cannot give
(two instructions with no shared wording that nonetheless ask the same thing), but its
numbers move with the model, so it enriches rather than decides.

BALANCE, dependency-free:

  normalised Shannon entropy over region and over format_type, 0 = everything in one
  bucket, 1 = perfectly uniform. Reported with the max share and the empty-bucket count,
  because entropy alone hides WHICH bucket is starved.

What this CANNOT measure
------------------------
- It is only as good as the probes until real reconstructed output exists. The synthetic
  set in tests/fixtures/ was built by hand to contain a known-repetitive half, so it can
  show the tool separates repetitive from varied. It says nothing about what real
  generation will look like.
- PHRASING variety is not TASK variety. Twenty differently-worded instructions can all
  ask for the same operation. Nothing here detects that; it needs an instruction-type
  taxonomy that does not exist yet.
- A region-spread warning is not automatically a generation defect. `western` is the
  corpus's known under-resourced region (18,079 tokens, 48% of the median), so a record
  set drawn proportionally from the corpus WILL look imbalanced. That is a sourcing fact,
  not a sampling error, and the report says so rather than implying it can be fixed by
  resampling.
- No thresholds are settled. The bands below are provisional and derived from the
  synthetic set; real data should set them.
- distinct-n is length-sensitive: short instructions produce fewer n-grams, so a set of
  terse instructions scores lower without necessarily being less varied.
"""

import argparse
import difflib
import json
import math
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_engineering'))

import eda as _eda                                              # noqa: E402

DIALECT_REGIONS = _eda.DIALECT_REGIONS
KNOWN_FORMATS = ('dictionary_entry', 'prose', 'narrative_paragraph', 'verse',
                 'footnote_block', 'list')

# Provisional bands, derived from the synthetic set only. NOT calibrated.
# عتبات مبدئية مستمدة من المجموعة الاصطناعية وحدها، وليست معايرة.
LOW_DISTINCT_2 = 0.45          # below this, phrasing repeats heavily
HIGH_NN_SIMILARITY = 0.55         # above this, instructions are near-templates
HIGH_TEMPLATE_SHARE = 0.35     # above this, most instructions open identically
HIGH_NEAR_TEMPLATE_RATE = 0.25 # share of records having a near-twin
LOW_ENTROPY = 0.75             # below this, the set leans hard on a few buckets


# ------------------------------------------------------------------------ n-gram core

def tokens(text):
    return (text or '').split()


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)] if len(toks) >= n else []


def distinct_n(texts, n):
    """unique n-grams / total n-grams across the set. 1.0 = never repeats."""
    total, uniq = 0, set()
    for t in texts:
        g = ngrams(tokens(t), n)
        total += len(g)
        uniq.update(g)
    return (len(uniq) / float(total)) if total else 0.0


def nearest_neighbour_similarity(texts):
    """For each text, the highest token-sequence similarity against any OTHER text.

    لكل تعليمة، أعلى تشابه تسلسلي مع أي تعليمة أخرى. هذا ما يكشف القوالب المعادة.

    This is the templating detector, and the measure behind it was CHOSEN BY MEASUREMENT
    after the obvious one failed.

    Trigram Jaccard was tried first and does not work on instructions this short. On a
    real templated pair - one 8-token frame with a single noun swapped - a one-token swap
    destroys 3 of the 6 trigrams, so Jaccard reads:

        1-gram 0.778    2-gram 0.556    3-gram 0.333    difflib token ratio 0.875

    At 0.333 the trigram measure misses a set that is unmistakably templated. difflib's
    sequence ratio counts matched tokens in order (2*M/T), so a single substitution costs
    proportionally rather than wiping out every window that overlaps it. It is also
    already a project dependency - check_facts.py uses it for near-miss names.
    """
    toks = [tokens(t) for t in texts]
    out = []
    for i, a in enumerate(toks):
        best = 0.0
        for j, b in enumerate(toks):
            if i != j:
                best = max(best, difflib.SequenceMatcher(None, a, b).ratio())
        out.append(best)
    return out


def template_share(texts, n=3):
    """Share of the set opening with the single most common leading n-gram."""
    heads = [tuple(tokens(t)[:n]) for t in texts if len(tokens(t)) >= n]
    if not heads:
        return 0.0, None
    head, count = Counter(heads).most_common(1)[0]
    return count / float(len(heads)), ' '.join(head)


# --------------------------------------------------------------------------- balance

def normalised_entropy(counts, universe_size=None):
    """Shannon entropy / log(k). 0 = one bucket, 1 = uniform across k buckets."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    k = universe_size or len(counts)
    if k <= 1:
        return 1.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    # max(0.0, ...) avoids reporting "-0.00" for a single-bucket set: the sum is a
    # signed zero there, which renders as negative and reads like a bug.
    return max(0.0, h / math.log(k))


def spread(values, universe):
    counts = Counter(values)
    total = sum(counts.values()) or 1
    empty = [u for u in universe if counts.get(u, 0) == 0]
    top, top_n = (counts.most_common(1)[0] if counts else (None, 0))
    return {
        'counts': dict(counts),
        'n': total,
        'entropy': normalised_entropy(counts, len(universe)),
        'max_share': top_n / float(total),
        'max_bucket': top,
        'empty_buckets': empty,
    }


# ------------------------------------------------------------------------- the report

def analyse(records, embedder=None):
    instructions = [r.get('instruction', '') for r in records]
    nn = nearest_neighbour_similarity(instructions)
    tmpl_share, tmpl_head = template_share(instructions)
    out = {
        'n_records': len(records),
        'n_unique_instructions': len(set(instructions)),
        'distinct_1': distinct_n(instructions, 1),
        'distinct_2': distinct_n(instructions, 2),
        'distinct_3': distinct_n(instructions, 3),
        'distinct_4': distinct_n(instructions, 4),
        'nn_similarity_mean': (sum(nn) / len(nn)) if nn else 0.0,
        'nn_similarity_max': max(nn) if nn else 0.0,
        'near_template_rate': (sum(1 for x in nn if x >= HIGH_NN_SIMILARITY) / float(len(nn)))
                              if nn else 0.0,
        'template_share': tmpl_share,
        'template_head': tmpl_head,
        'region': spread([r.get('source_region') for r in records], DIALECT_REGIONS),
        'format': spread([r.get('format_type') for r in records], KNOWN_FORMATS),
    }

    if embedder is not None:
        import numpy as np
        v = embedder.encode_query(instructions) if hasattr(embedder, 'encode_query') \
            else embedder.encode(instructions)
        sims = v @ v.T
        n = len(instructions)
        off = [sims[i][j] for i in range(n) for j in range(n) if i != j]
        out['embed_mean_cosine'] = float(sum(off) / len(off)) if off else 0.0
        out['embed_near_dup_rate'] = (sum(1 for x in off if x >= 0.95) / float(len(off))
                                      if off else 0.0)
        out['embed_model'] = getattr(embedder, 'model_id', '?')
    return out


def flags(a):
    """Provisional warnings. Each names the metric AND the band that produced it."""
    out = []
    if a['distinct_2'] < LOW_DISTINCT_2:
        out.append('LOW PHRASING VARIETY: distinct-2 %.2f < %.2f'
                   % (a['distinct_2'], LOW_DISTINCT_2))
    if a['nn_similarity_mean'] >= HIGH_NN_SIMILARITY:
        out.append('TEMPLATED INSTRUCTIONS: mean nearest-neighbour similarity %.2f >= %.2f'
                   % (a['nn_similarity_mean'], HIGH_NN_SIMILARITY))
    # The RATE survives dilution where the mean does not. Validation on the synthetic set
    # showed a mixed 50/50 batch averaging 0.515 - just under the mean band - while the
    # rate correctly reported 50%, i.e. exactly the templated half. A real batch will be
    # mixed, so the mean alone would routinely miss a templated SUBSET.
    # المعدل يصمد أمام التخفيف بخلاف المتوسط: دفعة مختلطة تُخفي القالب في المتوسط.
    if a['near_template_rate'] >= HIGH_NEAR_TEMPLATE_RATE:
        out.append('TEMPLATED SUBSET: %.0f%% of instructions have a near-twin (>= %.0f%%)'
                   % (100 * a['near_template_rate'], 100 * HIGH_NEAR_TEMPLATE_RATE))
    if a['template_share'] >= HIGH_TEMPLATE_SHARE:
        out.append('SHARED OPENING: %.0f%% of instructions start with %r (>= %.0f%%)'
                   % (100 * a['template_share'], a['template_head'],
                      100 * HIGH_TEMPLATE_SHARE))
    if a['n_unique_instructions'] < a['n_records']:
        out.append('EXACT DUPLICATES: %d of %d instructions are repeats'
                   % (a['n_records'] - a['n_unique_instructions'], a['n_records']))
    if a['region']['entropy'] < LOW_ENTROPY:
        out.append('REGION IMBALANCE: normalised entropy %.2f < %.2f (%r holds %.0f%%)'
                   % (a['region']['entropy'], LOW_ENTROPY, a['region']['max_bucket'],
                      100 * a['region']['max_share']))
    if a['region']['empty_buckets']:
        out.append('REGIONS ABSENT: %s' % ', '.join(a['region']['empty_buckets']))
    if a['format']['entropy'] < LOW_ENTROPY:
        out.append('FORMAT IMBALANCE: normalised entropy %.2f < %.2f (%r holds %.0f%%)'
                   % (a['format']['entropy'], LOW_ENTROPY, a['format']['max_bucket'],
                      100 * a['format']['max_share']))
    return out


def render(a, title='SFT diversity report'):
    L = ['# %s' % title, '',
         '%d record(s), %d unique instruction(s).'
         % (a['n_records'], a['n_unique_instructions']), '',
         '## Instruction phrasing', '',
         _eda.md_table(['metric', 'value', 'reading'], [
             ['distinct-1', '%.3f' % a['distinct_1'], 'unique unigrams / total'],
             ['distinct-2', '%.3f' % a['distinct_2'], 'lower = more repetition'],
             ['distinct-3', '%.3f' % a['distinct_3'], ''],
             ['distinct-4', '%.3f' % a['distinct_4'], ''],
             ['nearest-neighbour similarity (mean)', '%.3f' % a['nn_similarity_mean'],
              'higher = more templated'],
             ['near-template rate', '%.0f%%' % (100 * a['near_template_rate']),
              'share with a near-twin'],
             ['shared opening 3-gram', '%.0f%%' % (100 * a['template_share']),
              repr(a['template_head'])],
         ]), '']
    if 'embed_mean_cosine' in a:
        L += ['## Instruction phrasing - embedding layer', '',
              _eda.md_table(['metric', 'value'], [
                  ['mean pairwise cosine', '%.3f' % a['embed_mean_cosine']],
                  ['near-duplicate rate (cos >= 0.95)',
                   '%.0f%%' % (100 * a['embed_near_dup_rate'])],
                  ['model', a['embed_model']],
              ]), '']
    for label, key, universe in (('Region', 'region', DIALECT_REGIONS),
                                 ('Format', 'format', KNOWN_FORMATS)):
        s = a[key]
        rows = [[u, s['counts'].get(u, 0),
                 '%.0f%%' % (100 * s['counts'].get(u, 0) / float(s['n']))]
                for u in universe]
        L += ['## %s spread' % label, '',
              'normalised entropy **%.2f** (1.00 = uniform), max share %.0f%% in %r'
              % (s['entropy'], 100 * s['max_share'], s['max_bucket']), '',
              _eda.md_table(['%s' % label.lower(), 'records', 'share'], rows), '']
    f = flags(a)
    L += ['## Flags', '']
    L += ['- ' + x for x in f] if f else ['None raised at the current provisional bands.']
    L += ['', '_Bands are provisional and derived from a synthetic set; real data should',
          'set them. A region warning may reflect the corpus\'s own known imbalance',
          '(`western` is under-resourced at 48% of the median) rather than a sampling',
          'defect._']
    return '\n'.join(L)


# ------------------------------------------------------------------------- self-test

def run_self_test():
    ok = True
    # Perfectly repetitive: one phrase repeated. Every metric should sit at its worst.
    rep = ['اشرح معنى الكلمة في هذه المادة'] * 8
    # Genuinely varied: different openings, different structures, different lengths.
    var = ['اشرح معنى الكلمة في هذه المادة',
           'ما الدلالة التي يوردها المعجم هنا',
           'لخص مضمون المدخل بإيجاز شديد',
           'قارن بين الاستعمالين المذكورين',
           'استخرج الشواهد الشعرية الواردة',
           'بيّن أصل اللفظة ومشتقاتها',
           'اذكر النظائر في مناطق أخرى',
           'كيف يفرق النص بين المعنيين']

    a_rep, a_var = analyse([{'instruction': t} for t in rep]), \
                   analyse([{'instruction': t} for t in var])
    if not a_rep['distinct_2'] < a_var['distinct_2']:
        print('  [FAIL] distinct-2 did not separate repetitive from varied (%.3f vs %.3f)'
              % (a_rep['distinct_2'], a_var['distinct_2'])); ok = False
    if not a_rep['nn_similarity_mean'] > a_var['nn_similarity_mean']:
        print('  [FAIL] nearest-neighbour similarity did not separate (%.3f vs %.3f)'
              % (a_rep['nn_similarity_mean'], a_var['nn_similarity_mean'])); ok = False
    if a_rep['nn_similarity_mean'] < 0.99:
        print('  [FAIL] identical instructions should score ~1.0, got %.3f'
              % a_rep['nn_similarity_mean']); ok = False
    if not flags(a_rep):
        print('  [FAIL] a fully repetitive set raised no flag'); ok = False
    if a_rep['n_unique_instructions'] != 1:
        print('  [FAIL] unique-instruction count wrong'); ok = False

    # Templating that distinct-n alone would miss: one frame, one noun swapped.
    tmpl = ['اشرح معنى كلمة %s في هذه المادة المعجمية' % w
            for w in ('ألف', 'باء', 'تاء', 'ثاء', 'جيم', 'حاء', 'خاء', 'دال')]
    a_t = analyse([{'instruction': t} for t in tmpl])
    if a_t['n_unique_instructions'] != len(tmpl):
        print('  [FAIL] templated set should have no exact duplicates'); ok = False
    if a_t['nn_similarity_mean'] < HIGH_NN_SIMILARITY:
        print('  [FAIL] templating not detected: nn similarity %.3f' % a_t['nn_similarity_mean'])
        ok = False
    if not any('TEMPLATED' in f or 'SHARED OPENING' in f for f in flags(a_t)):
        print('  [FAIL] templated set raised no templating flag'); ok = False

    # A MIXED batch must still be flagged: the mean dilutes, the rate does not.
    mixed = [{'instruction': t} for t in tmpl] + [{'instruction': t} for t in var]
    a_m = analyse(mixed)
    if not any('TEMPLATED SUBSET' in f for f in flags(a_m)):
        print('  [FAIL] a half-templated batch raised no subset flag (rate %.2f, mean %.2f)'
              % (a_m['near_template_rate'], a_m['nn_similarity_mean'])); ok = False
    if not (0.4 <= a_m['near_template_rate'] <= 0.6):
        print('  [FAIL] near-template rate should identify ~half, got %.2f'
              % a_m['near_template_rate']); ok = False

    # entropy: one bucket vs uniform
    if normalised_entropy(Counter({'a': 10}), 5) != 0.0:
        print('  [FAIL] single-bucket entropy should be 0'); ok = False
    uni = normalised_entropy(Counter({k: 4 for k in 'abcde'}), 5)
    if abs(uni - 1.0) > 1e-9:
        print('  [FAIL] uniform entropy should be 1, got %.4f' % uni); ok = False
    sk = spread(['najdi'] * 9 + ['western'], DIALECT_REGIONS)
    if sk['entropy'] >= 0.75 or sk['max_bucket'] != 'najdi':
        print('  [FAIL] skewed spread not detected: %r' % sk); ok = False
    if sorted(sk['empty_buckets']) != ['eastern', 'northern', 'southern']:
        print('  [FAIL] empty buckets wrong: %r' % sk['empty_buckets']); ok = False

    # a balanced, varied set must raise no phrasing flag
    bal = [{'instruction': t, 'source_region': r, 'format_type': f}
           for t, r, f in zip(var, list(DIALECT_REGIONS) + list(DIALECT_REGIONS[:3]),
                              ['dictionary_entry', 'prose', 'list', 'verse',
                               'footnote_block', 'narrative_paragraph',
                               'dictionary_entry', 'prose'])]
    ab = analyse(bal)
    if any('LOW PHRASING VARIETY' in f or 'TEMPLATED' in f for f in flags(ab)):
        print('  [FAIL] a varied set was flagged as repetitive: %r' % flags(ab)); ok = False

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description='Diversity analysis over SFT records.')
    ap.add_argument('--in', dest='inp', help='SFT records .jsonl')
    ap.add_argument('--out', help='write the markdown report here')
    ap.add_argument('--title', default='SFT diversity report')
    ap.add_argument('--subset', choices=('all', 'repetitive', 'varied'), default='all',
                    help='filter by a `diversity_group` field, for validation runs')
    ap.add_argument('--embed', action='store_true',
                    help='add the embedding layer (needs requirements-verification.txt)')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)
    if not args.inp:
        sys.stderr.write('[diversity] --in <records.jsonl> is required\n')
        sys.exit(1)

    records = [json.loads(l) for l in open(args.inp, encoding='utf-8') if l.strip()]
    if args.subset != 'all':
        records = [r for r in records if r.get('diversity_group') == args.subset]
        if not records:
            sys.stderr.write('[diversity] no records with diversity_group=%r\n'
                             % args.subset)
            sys.exit(1)

    embedder = None
    if args.embed:
        try:
            import check_similarity as csim
            embedder = csim.SentenceTransformerEmbedder()
        except ImportError:
            sys.stderr.write('[diversity] sentence-transformers missing; n-gram only\n')

    a = analyse(records, embedder)
    if args.json:
        print(json.dumps(a, ensure_ascii=False, indent=2))
        return
    report = render(a, args.title)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(report + '\n')
        print('[diversity] wrote %s' % args.out)
    print(report)


if __name__ == '__main__':
    main()
