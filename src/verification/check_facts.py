# -*- coding: utf-8 -*-
"""Task 3 Group A item 2 - Fact-consistency checker.

Input : a candidate response + its source_chunk_id
        data/processed/facts_saudi_dialect.jsonl      (--corpus saudi_dialect)
        data/processed/facts_classical_lexicon.jsonl  (--corpus classical_lexicon)
Output: a verdict per candidate, and per-assertion detail

Fully deterministic - regex, string and difflib only. No LLM calls, consistent with
extract_facts.py and the rest of the pipeline.

NOTE ON THE FACTS FILENAME: the classical facts file is `facts_classical_lexicon.jsonl`,
named after the --corpus value, not `facts_asas_albalagha.jsonl`. Only the *chunks* file
carries the asas_albalagha name.

What this does
--------------
Given a candidate SFT response and the chunk it was generated from, it pulls the
checkable specifics out of the response - years, page references, cited authorities,
quoted lexical items, region claims - and tests each against that chunk's extracted
facts. Two failure modes are reported separately, because they mean different things:

  CONTRADICTED  the response asserts something the source also talks about, but with a
                different value. A wrong year, a corrupted name. This is the dangerous
                class: it looks grounded and is not.
  UNSUPPORTED   the response asserts a specific the source says nothing about at all.
                Classic hallucinated detail.

Why four verdict levels and not two
-----------------------------------
A binary pass/fail would mark a correct paraphrase as a failure. The extractor stores
what the SOURCE said, in the source's wording; a good response may say the same thing
differently - Arabic-Indic digits, a name with the article dropped, a role noun where the
source named the person. So matching is done on a NORMALIZED key, and a response value
that is a sub- or super-string of a real fact is reported as PARTIAL rather than failed.
PARTIAL means "a human should look", not "wrong".

  SUPPORTED   exact match to a known fact (after normalization)
  PARTIAL     sub/superstring of a known fact - paraphrase or partial name
  UNSUPPORTED no fact of this type in the chunk to match against
  CONTRADICTED conflicts with a fact of the same type that IS present

The response's overall verdict is the worst of its assertions.

Per-type policy, and why it differs
-----------------------------------
PRECISE types (hijri_year, gregorian_year, year_unmarked_era, page_reference) are
closed values: if the chunk carries years at all and the response's year matches none of
them, that is a contradiction, not a gap. Sources do not silently omit the year they are
discussing.

OPEN types (citation_authority, entry_headword, entry_root) are not closed: a chunk
holds a handful of names out of a large space, so a non-matching name is usually an
invention rather than a conflict. It is only reported as CONTRADICTED when it is a NEAR
miss of a real fact (difflib ratio >= 0.72), i.e. a corrupted version of a name the
source does carry, rather than an unrelated one.

Normalization is a MATCHING KEY, never output
---------------------------------------------
`norm_key` folds alef/ya, drops diacritics and tatweel, and maps Arabic-Indic digits to
ASCII. That is deliberately lossy, exactly as `clean.py`'s folded variant is: it is right
for matching and wrong for text you keep. Nothing normalized here is ever written into a
verdict's quoted value - the original response substring is reported.

Known limits, stated rather than hidden
---------------------------------------
- Lexical items are only detected when the response marks them - parentheses, quotes or
  Arabic quotation marks. A bare unmarked mention of a headword is not detected, so a
  hallucinated lemma written as plain prose can pass. This is conservative on purpose:
  scanning every word against the lemma space produced far more noise than signal.
- The checker inherits extract_facts.py's recall. A fact the extractor missed is not in
  the table, so a response asserting it reads as UNSUPPORTED. That is the correct
  behaviour for a ground-truth check, but it means UNSUPPORTED counts should be read
  alongside docs/citation_review_*.csv rather than as a pure hallucination rate.
- Region claims are matched against the chunk's own region and its
  cross_dialect_reference facts. A response may legitimately mention another region if
  the source does; it is flagged only when the source does not.
"""

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_facts as ef                                    # noqa: E402

FACTS = {
    'saudi_dialect':     'data/processed/facts_saudi_dialect.jsonl',
    'classical_lexicon': 'data/processed/facts_classical_lexicon.jsonl',
}
FIXTURES = {
    'saudi_dialect':     'tests/fixtures/candidates_saudi_dialect.jsonl',
    'classical_lexicon': 'tests/fixtures/candidates_classical_lexicon.jsonl',
}

SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED = (
    'SUPPORTED', 'PARTIAL', 'UNSUPPORTED', 'CONTRADICTED')
SEVERITY = {SUPPORTED: 0, PARTIAL: 1, UNSUPPORTED: 2, CONTRADICTED: 3}

PRECISE_TYPES = {'hijri_year', 'gregorian_year', 'year_unmarked_era', 'page_reference'}
OPEN_TYPES = {'citation_authority', 'entry_headword', 'entry_root'}
NEAR_MISS_RATIO = 0.72


# --------------------------------------------------------------------- normalization

ARABIC_INDIC = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
_DIA = re.compile(r'[ً-ْٰـ]')


def norm_key(s):
    """Lossy matching key. Never used as output - see module docstring."""
    s = s.translate(ARABIC_INDIC)
    s = _DIA.sub('', s)
    s = re.sub(r'[أإآٱ]', 'ا', s)
    s = s.replace('ى', 'ي').replace('ة', 'ه')
    return ' '.join(s.split())


# ------------------------------------------------------- assertions from the response

# Region vocabulary, used only for the dialect corpus. Maps surface forms a response
# might use onto the five region codes the corpus is partitioned by.
REGION_TERMS = {
    'نجد': 'najdi', 'النجديه': 'najdi', 'نجديه': 'najdi', 'القصيم': 'najdi',
    'الحجاز': 'western', 'الحجازيه': 'western', 'حجازيه': 'western',
    'جده': 'western', 'المدينه': 'western', 'ينبع': 'western',
    'الشمال': 'northern', 'الشماليه': 'northern', 'الجوف': 'northern',
    'الجنوب': 'southern', 'الجنوبيه': 'southern', 'عسير': 'southern',
    'الشرقيه': 'eastern', 'القطيف': 'eastern', 'الاحساء': 'eastern',
}
# Only count a region term when the response is ATTRIBUTING the entry to it.
REGION_ATTRIB = re.compile(
    r'(?:في|لهجة|لهجات|عند|أهل|من)\s+(?:أهل\s+)?(' +
    '|'.join(sorted({re.escape(k) for k in REGION_TERMS}, key=len, reverse=True)) + r')')

# A lexical item the response marks as a cited word: (x), "x", «x».
LEXICAL_MARKED = re.compile(r'[\(\"«“]([^\)\"»”\n]{1,30})[\)\"»”]')


def response_assertions(text, corpus):
    """Pull checkable specifics out of a candidate response.

    Reuses extract_facts.py's patterns rather than restating them, so the two stages
    cannot drift apart in what counts as a year, a page reference or a citation.
    """
    out = []

    def add(t, v):
        v = v.strip()
        if v:
            out.append({'type': t, 'value': v})

    page_spans = []
    for m in ef.DIA_PAGE_REF.finditer(text):
        page_spans.append((m.start(1), m.end(1)))
        add('page_reference', m.group(1))

    hijri_at = {m.start(1) for m in ef.DIA_HIJRI.finditer(text)}
    for m in ef.DIA_YEAR_ANY.finditer(text):
        s = m.start(1)
        if any(a <= s < b for a, b in page_spans):
            continue
        t = ef._year_type(m.group(1), s in hijri_at, bool(m.group(2)))
        if t:
            add(t, m.group(1))

    for m in ef.DIA_CITATION.finditer(text):
        name = ef._citation_value(m.group(1))
        if name:
            add('citation_authority', name)

    lex_type = 'entry_root' if corpus == 'classical_lexicon' else 'entry_headword'
    for m in LEXICAL_MARKED.finditer(text):
        inner = m.group(1).strip()
        # a marked span that is itself a number or a citation is already covered above
        if re.fullmatch(r'[\d\s.,-]+', inner):
            continue
        add(lex_type, inner)

    if corpus == 'saudi_dialect':
        for m in REGION_ATTRIB.finditer(text):
            add('region_claim', m.group(1))

    return out


# ------------------------------------------------------------------------ the check

def _judge(assertion, by_type, region_ctx):
    """Verdict for one assertion against the chunk's facts of the same type."""
    a_type, a_val = assertion['type'], assertion['value']
    key = norm_key(a_val)

    if a_type == 'region_claim':
        code = REGION_TERMS.get(norm_key(a_val))
        if code and code == region_ctx['region']:
            return SUPPORTED, 'matches the chunk region'
        for v in region_ctx['crossrefs']:
            if key and key in norm_key(v):
                return SUPPORTED, 'source itself references this region'
        return CONTRADICTED, 'attributes the entry to region %r; chunk region is %r' % (
            code or a_val, region_ctx['region'])

    known = by_type.get(a_type, [])
    known_keys = [norm_key(v) for v in known]

    for orig, k in zip(known, known_keys):
        if k == key:
            return SUPPORTED, 'exact match: %s' % orig

    for orig, k in zip(known, known_keys):
        if key and k and (key in k or k in key):
            return PARTIAL, 'sub/superstring of known fact: %s' % orig

    if a_type in PRECISE_TYPES:
        if known:
            return CONTRADICTED, 'chunk has %s %s; response says %s' % (
                a_type, '/'.join(known[:4]), a_val)
        return UNSUPPORTED, 'chunk carries no %s at all' % a_type

    best, ratio = None, 0.0
    for orig, k in zip(known, known_keys):
        r = difflib.SequenceMatcher(None, key, k).ratio()
        if r > ratio:
            best, ratio = orig, r
    if best is not None and ratio >= NEAR_MISS_RATIO:
        return CONTRADICTED, 'near-miss of known fact %r (ratio %.2f)' % (best, ratio)
    return UNSUPPORTED, 'no %s in chunk matches' % a_type


def check_response(response, chunk_id, index, corpus):
    row = index.get(chunk_id)
    if row is None:
        return {'source_chunk_id': chunk_id, 'verdict': 'UNKNOWN_CHUNK',
                'assertions': [], 'error': 'chunk_id not found in %s' % FACTS[corpus]}

    by_type = {}
    for f in row['extracted_facts']:
        by_type.setdefault(f['type'], []).append(f['value'])
    region_ctx = {'region': row['source_region'],
                  'crossrefs': by_type.get('cross_dialect_reference', [])}

    results = []
    for a in response_assertions(response, corpus):
        verdict, why = _judge(a, by_type, region_ctx)
        results.append({'type': a['type'], 'value': a['value'],
                        'verdict': verdict, 'reason': why})

    if not results:
        overall = 'NO_CHECKABLE_CLAIMS'
    else:
        overall = max((r['verdict'] for r in results), key=lambda v: SEVERITY[v])
    return {'source_chunk_id': chunk_id, 'source_region': row['source_region'],
            'verdict': overall, 'assertions': results}


def load_index(path):
    return {json.loads(l)['source_chunk_id']: json.loads(l)
            for l in open(path, encoding='utf-8') if l.strip()}


# ------------------------------------------------------------------------- self-test
#
# Synthetic, self-contained cases. Arabic here is invented placeholder text or the
# classical corpus's public structural markers - never a verbatim dialect passage; see
# extract_facts.py's licence-audit note.

def _fake_index():
    return {'t_c0001': {'source_chunk_id': 't_c0001', 'source_region': 'najdi',
                        'extracted_facts': [
                            {'type': 'hijri_year', 'value': '1426', 'position_in_text': 0},
                            {'type': 'page_reference', 'value': '39', 'position_in_text': 1},
                            {'type': 'citation_authority', 'value': 'مرزوق الفلاني',
                             'position_in_text': 2},
                            {'type': 'entry_headword', 'value': 'ططط', 'position_in_text': 3},
                            {'type': 'cross_dialect_reference', 'value': 'الحجاز',
                             'position_in_text': 4}]}}


SELF_TEST_CASES = [
    # (response, expected overall verdict, note)
    ('طبع المرجع سنة 1426هـ.', SUPPORTED, 'exact year match'),
    ('طبع المرجع سنة ١٤٢٦هـ.', SUPPORTED, 'Arabic-Indic digits normalise to the same year'),
    ('طبع المرجع سنة 1436هـ.', CONTRADICTED, 'chunk has a hijri year; this is a different one'),
    ('انظر المرجع ص.39 للاختبار.', SUPPORTED, 'exact page match'),
    ('انظر المرجع ص.77 للاختبار.', CONTRADICTED, 'chunk has a page ref; this is a different one'),
    ('قال مرزوق الفلاني كلاما.', SUPPORTED, 'exact authority match'),
    ('قال مرزوق الفلانى كلاما.', SUPPORTED, 'ya/alef-maqsura folding makes this the same name'),
    ('قال مرزوق الغلاني كلاما.', CONTRADICTED, 'near-miss of a real name = corrupted, not invented'),
    ('قال سعدون الخيالي كلاما.', UNSUPPORTED, 'unrelated name, no near match'),
    ('الكلمة (ططط) معناها كذا.', SUPPORTED, 'marked lexical item matches a headword'),
    ('الكلمة (زززز) معناها كذا.', UNSUPPORTED, 'marked lexical item not in the chunk'),
    ('هذه الكلمة في لهجة نجد.', SUPPORTED, 'region claim matches the chunk region'),
    ('هذه الكلمة في لهجة الحجاز.', SUPPORTED, 'source itself cross-references this region'),
    ('هذه الكلمة في لهجة القطيف.', CONTRADICTED, 'attributes to a region the source does not'),
    ('هذا كلام عام بلا تفاصيل.', 'NO_CHECKABLE_CLAIMS', 'nothing specific to check'),
]


def run_self_test():
    idx = _fake_index()
    ok = True
    for i, (resp, want, note) in enumerate(SELF_TEST_CASES, 1):
        got = check_response(resp, 't_c0001', idx, 'saudi_dialect')
        if got['verdict'] != want:
            print('  [FAIL] case %d: expected %s got %s  (%s)'
                  % (i, want, got['verdict'], note))
            for a in got['assertions']:
                print('           %-22s %-12s %s' % (a['type'], a['verdict'], a['reason']))
            ok = False
    # an unknown chunk id must be reported, not silently pass
    if check_response('نص', 'nope', idx, 'saudi_dialect')['verdict'] != 'UNKNOWN_CHUNK':
        print('  [FAIL] unknown chunk_id not reported')
        ok = False
    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def _render(case, res):
    mark = {SUPPORTED: 'OK  ', PARTIAL: 'PART', UNSUPPORTED: 'UNSUP',
            CONTRADICTED: 'CONTRA', 'NO_CHECKABLE_CLAIMS': '----',
            'UNKNOWN_CHUNK': '????'}
    print('-' * 78)
    print('[%s] %s' % (res['verdict'], case.get('label', '')))
    print('  chunk   : %s (%s)' % (res['source_chunk_id'], res.get('source_region', '?')))
    print('  response: %s' % case['response'][:150].replace('\n', ' '))
    if case.get('expect'):
        flag = 'as expected' if res['verdict'] == case['expect'] else \
               'MISMATCH - expected %s' % case['expect']
        print('  expected: %s  (%s)' % (case['expect'], flag))
    for a in res['assertions']:
        print('    %-6s %-22s %-24s %s'
              % (mark.get(a['verdict'], '?'), a['type'], a['value'][:22], a['reason'][:60]))
    if not res['assertions']:
        print('    (no checkable assertions)')


def main():
    ap = argparse.ArgumentParser(description='Fact-consistency checker (Task 3).')
    ap.add_argument('--corpus', choices=sorted(FACTS),
                    help='which corpus the chunk belongs to; required (no auto-detection)')
    ap.add_argument('--facts', help='override facts .jsonl')
    ap.add_argument('--response', help='a single candidate response to check')
    ap.add_argument('--chunk-id', help='source_chunk_id for --response')
    ap.add_argument('--in', dest='inp', help='candidates .jsonl to check in bulk')
    ap.add_argument('--fixtures', action='store_true',
                    help='run the synthetic candidate fixtures for --corpus')
    ap.add_argument('--json', action='store_true', help='emit JSON instead of a report')
    ap.add_argument('--self-test', action='store_true',
                    help='verify the checker against synthetic cases and exit')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    if not args.corpus:
        sys.stderr.write(
            '[check] REFUSING to run: --corpus is required. The two corpora have\n'
            '    different fact vocabularies (entry_headword/region vs entry_root/\n'
            '    figurative_sense); guessing would check against the wrong table.\n'
            '    corpora available:\n      ' + '\n      '.join(sorted(FACTS)) + '\n')
        sys.exit(1)

    facts_path = args.facts or FACTS[args.corpus]
    if not os.path.exists(facts_path):
        sys.stderr.write('[check] facts file not found: %s\n'
                         '    run: python src/verification/extract_facts.py --corpus %s\n'
                         % (facts_path, args.corpus))
        sys.exit(1)
    index = load_index(facts_path)

    if args.response:
        if not args.chunk_id:
            sys.stderr.write('[check] --response requires --chunk-id\n')
            sys.exit(1)
        cases = [{'response': args.response, 'source_chunk_id': args.chunk_id}]
    else:
        path = args.inp or (FIXTURES[args.corpus] if args.fixtures else None)
        if not path:
            sys.stderr.write('[check] give --response/--chunk-id, --in, or --fixtures\n')
            sys.exit(1)
        if not os.path.exists(path):
            sys.stderr.write('[check] candidates file not found: %s\n' % path)
            sys.exit(1)
        cases = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]

    results = []
    for c in cases:
        results.append(check_response(c['response'], c['source_chunk_id'],
                                      index, args.corpus))

    if args.json:
        for c, r in zip(cases, results):
            print(json.dumps({'label': c.get('label'), **r}, ensure_ascii=False))
    else:
        for c, r in zip(cases, results):
            _render(c, r)
        print('-' * 78)
        tally = Counter(r['verdict'] for r in results)
        print('%d candidate(s): %s' % (len(results), '  '.join(
            '%s=%d' % (k, v) for k, v in tally.most_common())))
        exp = [(c, r) for c, r in zip(cases, results) if c.get('expect')]
        if exp:
            hit = sum(1 for c, r in exp if c['expect'] == r['verdict'])
            print('expectations: %d/%d matched%s'
                  % (hit, len(exp), '' if hit == len(exp) else '   <-- REVIEW'))


if __name__ == '__main__':
    main()
