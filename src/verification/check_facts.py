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
  PARTIAL     a whole-token abbreviation or extension of a known fact
  UNSUPPORTED no fact of this type in the chunk to match against
  CONTRADICTED conflicts with a fact of the same type that IS present

Sub/superstring is NOT symmetric
-------------------------------
An earlier version treated any character containment as PARTIAL, which conflated two
opposite things. Dropping material from a real name ("الحارث بن مرارة" for
"الحارث بن مرارة الحنظلي") invents nothing. ADDING material does, and it is precisely the
shape the DPO stage generates deliberately as an "unsupported additions" weakness -
so reading it as a legitimate fuller form was exactly backwards.

The two are separated by TOKEN ALIGNMENT rather than character containment:

  abbreviation  response tokens are a contiguous run of the fact's tokens -> PARTIAL
  extension     the fact's tokens survive intact and whole tokens are added -> PARTIAL,
                or CONTRADICTED when chunk_text is supplied and an added token does not
                occur in the source at all
  corruption    containment holds at character level but a token was altered, i.e. junk
                welded onto a word ("التغلبي" -> "التغلبيّي") -> CONTRADICTED

`chunk_text` is optional. Without it an extension stays PARTIAL, because there is no
corpus evidence to justify a hard call; verify_sft.py supplies it.

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
    'najdi_popular':     'data/processed/facts_najdi_popular.jsonl',
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


def response_assertions(text, corpus, instruction=None):
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
        # A MULTI-WORD marked span cannot be an entry_ROOT and must not be typed as one.
        # It is emitted below as `quoted_lexical_item` instead, judged against the chunk
        # text rather than against a table of bare triliteral roots.
        #
        # This was a real defect, not tidying. Measured: every classical entry_root fact
        # is exactly 3 characters with no space, so a phrase could never match one and
        # always came back UNSUPPORTED - and since a record takes the WORST of its
        # assertions, that bogus assertion masked a correct PARTIAL from the same span.
        # The morphology fix could not reach it either: there is no root to inflect.
        #
        # CLASSICAL ONLY. The dialect corpus's `entry_headword` values ARE frequently
        # multi-word (`DIA_HEADWORD` allows up to 40 characters including spaces), and
        # the abbreviation/extension cases depend on that - excluding phrases there broke
        # four existing self-test cases, which is how this scoping was found.
        if lex_type == 'entry_root' and len(inner.split()) > 1:
            continue
        add(lex_type, inner)

    if corpus == 'saudi_dialect':
        for m in REGION_ATTRIB.finditer(text):
            add('region_claim', m.group(1))

    # Quoted lexical items. The INSTRUCTION is read as well as the response, and
    # that asymmetry is measured, not stylistic: of the 3,333 records the corpus
    # rules found nothing in, 98% quote the item in the instruction and only 43%
    # repeat it in the response. Response-only extraction is capped at 43% coverage
    # by construction.
    #
    # The CLAIM is still the response's. The instruction merely names what the
    # response is glossing; _judge() checks the response's own wording against the
    # source around that item. An item taken from the instruction is never by itself
    # evidence about the response.
    for item in ef.quoted_lexical_items(text):
        add('quoted_lexical_item', item)
    if instruction:
        for item in ef.quoted_lexical_items(instruction):
            add('quoted_lexical_item', item)

    return out


# ------------------------------------------------------------------------ the check

# ------------------------------------------------ sub/superstring relationship analysis
#
# Plain containment conflates two opposite situations, and probe C-E showed the cost:
#
#   response SHORTER than the fact   "الحارث بن مرارة" vs "الحارث بن مرارة الحنظلي"
#       The response drops the nisba. It asserts a subset of what the source says, so
#       nothing has been invented. PARTIAL is right.
#
#   response LONGER than the fact    "التغلبيّي" vs "التغلبي"
#       The response ADDS material. This is the shape the DPO stage deliberately
#       generates as an "unsupported additions" weakness, and reading it as a
#       legitimate fuller form is exactly backwards.
#
# The two are separated by TOKEN ALIGNMENT rather than by character containment:
#
#   extension    every token of the known fact survives intact and whole new tokens are
#                appended - a plausible fuller name (الحنظلي added to الحارث بن مرارة)
#   corruption   containment holds at the character level but a token has been altered,
#                so the added characters are junk welded onto an existing word
#
# Character containment cannot tell these apart; token alignment can, and it does not
# need a corpus lookup to do it. When the chunk text IS available (verify_sft.py passes
# it) an extension is checked further: added tokens absent from the source are an
# unsupported addition, not a fuller form.

def _token_run(big, small):
    """Index where `small` occurs as a contiguous run inside `big`, else -1."""
    if not small or len(small) > len(big):
        return -1
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return i
    return -1


# Gloss window around a located quoted item, in normalised characters. A dictionary
# entry's explanation follows its headword, so the window reaches mostly forward; the
# small backward reach catches a preposed gloss. 200/40 was the measured configuration
# (84% local overlap vs 6% against an unrelated window) - widening it raises the local
# rate and the unrelated rate together, which buys nothing.
GLOSS_WINDOW, GLOSS_BACK = 200, 40

# Function words carry no evidence: they appear in every Arabic sentence, so counting
# them as shared content would make the overlap test fire on everything. Deliberately
# short - a long hand-tuned stop list would be fitted to this one delivery.
GLOSS_STOPWORDS = frozenset(norm_key(w) for w in (
    'في من على عن الى هو هي هذا هذه ذلك التي الذي ما لا و او ان انه مع عند بعد قبل '
    'كل بين لكن زي مثل يكون تكون صار يعني معنى معناها يقصدون القصد كان لما اللي شي '
    'شيء واحد الناس نقول يقول قال').split())

# 4+ letters: shorter Arabic tokens are overwhelmingly particles and pronouns.
_CONTENT_WORD = re.compile(r'[ء-ي]{4,}')


def _content_words(text):
    """Content-word set of a text, normalised, stop-words removed."""
    return {w for w in _CONTENT_WORD.findall(norm_key(text or ''))
            if w not in GLOSS_STOPWORDS}


# A bare root. Measured: every entry_root fact in the classical corpus is exactly 3
# characters. The range is 2-4 so a quadriliteral root in a future corpus is handled, but
# anything longer is a phrase and must not be treated as a root.
ROOT_MIN_LEN, ROOT_MAX_LEN = 2, 4


def _is_bare_root(k):
    return k and ' ' not in k and ROOT_MIN_LEN <= len(k) <= ROOT_MAX_LEN


def _subsequence(root, word):
    """True if `root`'s characters occur IN ORDER inside `word`, not necessarily adjacent.

    Handles derivation the substring test cannot: أثف inside أثفية is contiguous, but a
    hollow root's letters are separated by an infixed vowel and only this test finds them.
    """
    i = 0
    for ch in word:
        if i < len(root) and ch == root[i]:
            i += 1
    return i == len(root)


def _root_subsequence_hit(resp_key, known, known_keys):
    """The first chunk root the response inflects, or None.

    Returns the ORIGINAL root string (not the normalised key) so the reason line names
    what a reviewer would search the chunk for.
    """
    toks = [t for t in resp_key.split() if t]
    for orig, k in zip(known, known_keys):
        if not _is_bare_root(k):
            continue
        for t in toks:
            if len(t) >= len(k) and _subsequence(k, t):
                return orig
    return None


def _relation(resp_key, known_key):
    """Classify how a response value relates to a known fact value.

    Returns 'equal', 'abbreviation', 'extension', 'corruption', or None.
    """
    if resp_key == known_key:
        return 'equal'
    rt, kt = resp_key.split(), known_key.split()
    if _token_run(kt, rt) >= 0:
        return 'abbreviation'          # response is a whole-token subset of the fact
    if _token_run(rt, kt) >= 0:
        return 'extension'             # fact survives intact, whole tokens added
    if resp_key in known_key or known_key in resp_key:
        return 'corruption'            # characters welded onto/into a token
    return None


def _judge(assertion, by_type, region_ctx, chunk_tokens=None,
           chunk_text=None, response=None):
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

    # ---------------------------------------- quoted_lexical_item: does the source say it,
    #                                           and does the response gloss it plausibly?
    #
    # Not judged against the fact table at all - it is judged against the CHUNK TEXT,
    # because the claim is "the source contains this span", and the table holds roots and
    # authorities rather than arbitrary phrases.
    #
    # Two stages, and both are needed. Stage one asks whether the item exists in the
    # source. Stage two asks whether the response's own wording overlaps what the source
    # says AROUND that item. Stage one alone would mark 3,290 records as carrying a
    # checkable claim that essentially always passes, which is worse than silence: it
    # looks like verification and detects nothing but a fabricated headword.
    #
    # MEASURED, 2026-09-09, on 3,333 records:
    #   item present in its own chunk            94%   (1% in a random other chunk)
    #   multi-word item, own chunk               87%   (0% in a random other chunk)
    #   response overlaps the local gloss window 84%   (6% against an unrelated window)
    #
    # PARTIAL on success, never SUPPORTED. 84/6 is real discrimination but it is far
    # weaker than the morphology rule's 1% false-hit rate, and a single shared content
    # word is thin evidence that a gloss is CORRECT. PARTIAL means "a human should look",
    # which is what this measurement supports.
    #
    # الاقتباس موجود في المصدر، وشرحُه يتقاطع مع سياقه - وهذا يستدعي مراجعة، لا قبولًا.
    if a_type == 'quoted_lexical_item':
        if not chunk_text:
            return (PARTIAL,
                    'quoted item %r cannot be located: no chunk text supplied' % a_val)
        nct, nkey = norm_key(chunk_text), key
        pos = nct.find(nkey) if nkey else -1
        if pos < 0:
            return (UNSUPPORTED,
                    'the source does not contain the quoted item %r' % a_val)
        if not response:
            return (PARTIAL, 'quoted item %r occurs in the source' % a_val)
        window = nct[max(0, pos - GLOSS_BACK):pos + len(nkey) + GLOSS_WINDOW]
        shared = _content_words(response) & _content_words(window)
        if shared:
            return (PARTIAL,
                    'quoted item %r occurs in the source and the response shares %d '
                    'content word(s) with the surrounding gloss - needs a human'
                    % (a_val, len(shared)))
        return (UNSUPPORTED,
                'quoted item %r occurs in the source, but the response shares no content '
                'word with what the source says around it' % a_val)

    # ------------------------------------------------ entry_root: morphology, not corruption
    #
    # MEASURED on 4,645 real reconstructed responses (2026-09-08): of the entry_root
    # assertions this function flagged, 87% of CONTRADICTED and 78% of UNSUPPORTED were
    # strings that appear VERBATIM in their own source chunk. That is not what a
    # contradiction looks like.
    #
    # The cause is a type mismatch, not a bug in the logic below. Every entry_root fact is
    # a BARE TRILITERAL ROOT - measured: 3,372 facts, all exactly 3 characters. Responses
    # cite the inflected surface form: root أثف appears as أثفية, مدد as مددت, جلح as
    # الأجلح. The generic analysis then sees characters welded onto a known value and
    # returns 'corruption', which is the right reading for a multi-word NAME ('جابر بن حني'
    # with extra letters is suspicious) and the wrong reading for a root, where added
    # letters are ordinary Arabic derivation.
    #
    # الجذر يُصرَّف، فالحروف الزائدة اشتقاق لا تحريف.
    #
    # So entry_root gets its own rule, applied BEFORE the generic analysis because the
    # generic analysis is what produces the false positive. A chunk root whose letters
    # appear IN ORDER within a response token means the response is discussing a root the
    # chunk actually covers.
    #
    # PARTIAL, not SUPPORTED, and the distinction is deliberate: naming a root the chunk
    # covers is not the same as asserting the source's claim ABOUT that root. The response
    # may still say something false about it. PARTIAL means "a human should look", which is
    # exactly the epistemic state here.
    #
    # Subsequence rather than substring: substring recovers 72% of the flagged
    # CONTRADICTED but 0% of the UNSUPPORTED, because UNSUPPORTED is by construction the
    # set where containment already failed. Those are hollow and defective roots whose
    # letters are separated by infixed vowels. Subsequence reaches 89% and 32%.
    #
    # Specificity is measured, not assumed: against a RANDOM OTHER chunk's roots the rule
    # fires on 1% of the same tokens (10/1510). run_self_test() pins that.
    if a_type == 'entry_root':
        hit = _root_subsequence_hit(key, known, known_keys)
        if hit is not None:
            return (PARTIAL,
                    'response cites an inflected form of the chunk root %r; naming a root '
                    'the chunk covers is not an assertion about it - needs a human' % hit)
        # No chunk root matches at all. That IS the suspicious case, so fall through to
        # the generic analysis and let it return UNSUPPORTED/CONTRADICTED as before.

    # Directional analysis, strongest signal first: a corruption anywhere outranks an
    # abbreviation elsewhere, because the corrupted claim is the one that is wrong.
    relations = []
    for orig, k in zip(known, known_keys):
        if key and k:
            rel = _relation(key, k)
            if rel:
                relations.append((rel, orig, k))

    for rel, orig, k in relations:
        if rel == 'corruption':
            return (CONTRADICTED,
                    'appended/altered characters on a known fact %r - the source token '
                    'was modified, not extended' % orig)

    for rel, orig, k in relations:
        if rel == 'extension':
            added = [t for t in key.split() if t not in k.split()]
            if chunk_tokens is not None and added:
                unsupported = [t for t in added if t not in chunk_tokens]
                if unsupported:
                    return (CONTRADICTED,
                            'extends known fact %r with token(s) absent from the source: '
                            '%s' % (orig, ' '.join(unsupported)))
            return (PARTIAL,
                    'whole-token extension of known fact: %s (added: %s)'
                    % (orig, ' '.join(added) or '-'))

    for rel, orig, k in relations:
        if rel == 'abbreviation':
            return PARTIAL, 'whole-token abbreviation of known fact: %s' % orig

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


def check_response(response, chunk_id, index, corpus, chunk_text=None,
                   instruction=None):
    row = index.get(chunk_id)
    if row is None:
        return {'source_chunk_id': chunk_id, 'verdict': 'UNKNOWN_CHUNK',
                'assertions': [], 'error': 'chunk_id not found in %s' % FACTS[corpus]}

    by_type = {}
    for f in row['extracted_facts']:
        by_type.setdefault(f['type'], []).append(f['value'])
    region_ctx = {'region': row['source_region'],
                  'crossrefs': by_type.get('cross_dialect_reference', [])}

    chunk_tokens = set(norm_key(chunk_text).split()) if chunk_text else None
    results = []
    for a in response_assertions(response, corpus, instruction):
        verdict, why = _judge(a, by_type, region_ctx, chunk_tokens,
                              chunk_text=chunk_text, response=response)
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
                            {'type': 'citation_authority', 'value': 'سالم بن حمد الفلاني',
                             'position_in_text': 5},
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
    # --- sub/superstring direction: the C-E append-corruption regression ---
    # A shorter form of a real name drops material; nothing is invented.
    ('قال سالم بن حمد كلاما.', PARTIAL,
     'whole-token abbreviation of a longer known name stays PARTIAL'),
    # A longer form that leaves every known token intact is a plausible fuller name.
    # Uses the lexical path, whose capture is the whole marked span - the citation regex
    # caps how much of a name it will take, so it cannot express this case.
    ('الكلمة (ططط الكبير) معناها كذا.', PARTIAL,
     'whole-token extension of a known headword stays PARTIAL'),
    # Characters welded onto a known token are a corruption, NOT a fuller form. This is
    # probe C-E's shape (التغلبي -> التغلبيّي) and it must not read as PARTIAL.
    ('الكلمة (طططط) معناها كذا.', CONTRADICTED,
     'appended characters on a known token are a corruption'),
    ('الكلمة (اططط) معناها كذا.', CONTRADICTED,
     'prepended characters on a known token are a corruption'),
    ('قال سالم بن حمد الفلانيي كلاما.', CONTRADICTED,
     'corruption of the final token of a longer name - the C-E shape'),

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
    # --- extension validated against the chunk text, when it is available ---
    # An added token the SOURCE also contains is a plausible fuller form.
    r = check_response('الكلمة (ططط الكبير) معناها كذا.', 't_c0001', idx, 'saudi_dialect',
                       chunk_text='نص فيه ططط الكبير مذكور صراحة')
    if r['verdict'] != PARTIAL:
        print('  [FAIL] extension with a source-supported token should be PARTIAL, got %s'
              % r['verdict'])
        ok = False
    # An added token absent from the source is an unsupported addition - the DPO
    # "unsupported additions" weakness shape.
    r = check_response('الكلمة (ططط الخيالي) معناها كذا.', 't_c0001', idx, 'saudi_dialect',
                       chunk_text='نص فيه ططط وحده دون اي اضافة')
    if r['verdict'] != CONTRADICTED:
        print('  [FAIL] extension with an unsourced token should be CONTRADICTED, got %s'
              % r['verdict'])
        ok = False
    # Without chunk_text the same case stays PARTIAL - no corpus evidence, no hard call.
    r = check_response('الكلمة (ططط الخيالي) معناها كذا.', 't_c0001', idx, 'saudi_dialect')
    if r['verdict'] != PARTIAL:
        print('  [FAIL] extension without chunk_text should stay PARTIAL, got %s'
              % r['verdict'])
        ok = False

    # relation classifier, directly
    for a, b, want in ((norm_key('الحارث بن مرارة'), norm_key('الحارث بن مرارة الحنظلي'),
                        'abbreviation'),
                       (norm_key('الحارث بن مرارة الحنظلي'), norm_key('الحارث بن مرارة'),
                        'extension'),
                       (norm_key('التغلبيي'), norm_key('التغلبي'), 'corruption'),
                       (norm_key('الاعشى'), norm_key('الاعشى'), 'equal'),
                       (norm_key('النابغه'), norm_key('الفرزدق'), None)):
        got = _relation(a, b)
        if got != want:
            print('  [FAIL] _relation(%r, %r) = %r, expected %r' % (a, b, got, want))
            ok = False

    # an unknown chunk id must be reported, not silently pass
    if check_response('نص', 'nope', idx, 'saudi_dialect')['verdict'] != 'UNKNOWN_CHUNK':
        print('  [FAIL] unknown chunk_id not reported')
        ok = False

    # ---------------------------------------------- entry_root morphology (invented roots)
    #
    # All Arabic below is INVENTED. No source passage is hardcoded in a self-test - the
    # licence audit found ten real dialect passages that had migrated into this stack's
    # docstrings during debugging, and a test fixture is exactly where that happens.
    if not _is_bare_root('ططط'):
        print('  [FAIL] a 3-letter token is not recognised as a bare root'); ok = False
    for bad in ('ططط ططط', 'طططططط', '', 'ط'):
        if _is_bare_root(bad):
            print('  [FAIL] %r accepted as a bare root' % bad); ok = False

    # derivation: the root's letters appear in order inside the inflected form
    for root, word, want in (('ططط', 'ططط', True),        # bare
                             ('ططط', 'المططط', True),      # prefixed
                             ('ططط', 'ططظة', False),       # only two of the three letters
                             ('ططط', 'طاطاط', True),       # infixed (hollow-root shape)
                             ('ططط', 'طط', False),         # too short to contain it
                             ('ططط', 'طظطظط', True)):      # separated but in order
        if _subsequence(root, word) != want:
            print('  [FAIL] _subsequence(%r, %r) != %s' % (root, word, want)); ok = False

    # order matters - the same letters reversed must NOT match
    if _subsequence('طظع', 'عظط'):
        print('  [FAIL] subsequence matched letters in the wrong order'); ok = False

    known = ['ططط', 'ظظظ']
    kk = [norm_key(v) for v in known]
    if _root_subsequence_hit('المططط', known, kk) != 'ططط':
        print('  [FAIL] inflected form did not resolve to its root'); ok = False
    if _root_subsequence_hit('عععع', known, kk) is not None:
        print('  [FAIL] unrelated token matched a root'); ok = False
    # a multi-word known value is a phrase, not a root, and must never match this way
    if _root_subsequence_hit('ططط', ['ططط ظظظ'], [norm_key('ططط ظظظ')]) is not None:
        print('  [FAIL] a multi-word known value was treated as a root'); ok = False

    # SPECIFICITY REGRESSION PIN.
    # MEASURED on the real 2026-09-08 intake: the same response tokens tested against a
    # RANDOM OTHER chunk's roots fired on 10 of 1510 (1%). The rule is discriminative, and
    # that is the property most at risk if someone later loosens it - e.g. to letter-set
    # overlap, or by dropping the in-order requirement. This pins the shape of that
    # measurement on invented data: unrelated roots must not match unrelated tokens.
    alien_roots = ['ظظظ', 'عععا', 'غغغ', 'فففا', 'قققا']
    alien_keys = [norm_key(v) for v in alien_roots]
    tokens = ['المططط', 'مططط', 'تططط', 'ططططة', 'ططاط']
    hits = sum(1 for t in tokens
               if _root_subsequence_hit(t, alien_roots, alien_keys) is not None)
    if hits:
        print('  [FAIL] specificity: %d of %d unrelated tokens matched an alien root; '
              'the in-order requirement has been weakened' % (hits, len(tokens)))
        ok = False
    # ...and the matching root must still be found when it IS present
    if _root_subsequence_hit('المططط', alien_roots + ['ططط'],
                             alien_keys + [norm_key('ططط')]) != 'ططط':
        print('  [FAIL] specificity pin suppressed a genuine root match'); ok = False

    # ------------------------------------------- quoted_lexical_item (invented Arabic)
    #
    # All Arabic invented, as above. `ططط ظظظ` stands in for a quoted lexical phrase.
    src_chunk = 'ططط ظظظ: عععع غغغغ ففففف. وقققق كككك.'
    idx2 = {'q_c1': {'source_chunk_id': 'q_c1', 'source_region': 'classical',
                     'extracted_facts': []}}

    # the item exists AND the response overlaps the gloss around it -> PARTIAL
    r = check_response('يعني "ططط ظظظ" هي عععع غغغغ.', 'q_c1', idx2,
                       'classical_lexicon', chunk_text=src_chunk)
    if r['verdict'] != PARTIAL:
        print('  [FAIL] located item + gloss overlap -> %s, expected PARTIAL'
              % r['verdict']); ok = False

    # the item does NOT exist in the source -> UNSUPPORTED
    r = check_response('يعني "خخخخ ذذذذ" هي عععع.', 'q_c1', idx2,
                       'classical_lexicon', chunk_text=src_chunk)
    if r['verdict'] != UNSUPPORTED:
        print('  [FAIL] absent quoted item -> %s, expected UNSUPPORTED'
              % r['verdict']); ok = False

    # the item exists but the response shares NO content word with its gloss
    r = check_response('يعني "ططط ظظظ" هي شششش تتتتت.', 'q_c1', idx2,
                       'classical_lexicon', chunk_text=src_chunk)
    if r['verdict'] != UNSUPPORTED:
        print('  [FAIL] located item with no gloss overlap -> %s, expected UNSUPPORTED'
              % r['verdict']); ok = False

    # THE INSTRUCTION PATH: the item is quoted only in the question, which is the 98%
    # case. Without instruction= the record must stay silent; with it, it must be judged.
    resp = 'هي عععع غغغغ.'
    r = check_response(resp, 'q_c1', idx2, 'classical_lexicon', chunk_text=src_chunk)
    if r['verdict'] != 'NO_CHECKABLE_CLAIMS':
        print('  [FAIL] response-only should find nothing here, got %s' % r['verdict'])
        ok = False
    r = check_response(resp, 'q_c1', idx2, 'classical_lexicon', chunk_text=src_chunk,
                       instruction='وش معنى "ططط ظظظ"؟')
    if r['verdict'] != PARTIAL:
        print('  [FAIL] instruction-sourced item -> %s, expected PARTIAL' % r['verdict'])
        ok = False

    # BACKWARD COMPATIBILITY: instruction is optional and its absence must not raise.
    try:
        check_response('نص', 'q_c1', idx2, 'classical_lexicon')
    except TypeError as exc:
        print('  [FAIL] check_response is no longer callable without instruction: %s' % exc)
        ok = False

    # extraction shape
    if ef.quoted_lexical_items('وش معنى "ططط ظظظ"؟') != ['ططط ظظظ']:
        print('  [FAIL] double-quoted span not extracted'); ok = False
    if ef.quoted_lexical_items("وش معنى 'ططط ظظظ'؟") != ['ططط ظظظ']:
        print('  [FAIL] SINGLE-quoted span not extracted - this was the 53% case')
        ok = False
    if ef.quoted_lexical_items('وش معنى "ططط"؟'):
        print('  [FAIL] single-token span extracted; min is %d Arabic tokens'
              % ef.QUOTED_MIN_TOKENS); ok = False
    if ef.quoted_lexical_items("قال '1426 هـ' في النص"):
        print('  [FAIL] a quoted YEAR became a lexical item; years have their own types')
        ok = False
    if ef.quoted_lexical_items("it's a plain english don't"):
        print('  [FAIL] a Latin apostrophe pair produced an Arabic lexical item')
        ok = False

    # SPECIFICITY / DISCRIMINATION PIN.
    # MEASURED 2026-09-09 on the real intake: a multi-word quoted item appears in its own
    # chunk 87% of the time and in a RANDOM OTHER chunk 0% of the time; single-token spans
    # were 2%, which is why QUOTED_MIN_TOKENS is 2. This pins the shape of that result:
    # multi-word items must not match unrelated sources. Lowering the minimum to 1, or
    # matching on any shared token instead of the whole span, breaks this first.
    unrelated = 'خخخخ ذذذذ: شششش تتتتت. ونننن مممم.'
    idx3 = {'u_c1': {'source_chunk_id': 'u_c1', 'source_region': 'classical',
                     'extracted_facts': []}}
    for probe in ('ططط ظظظ', 'عععع غغغغ', 'وقققق كككك'):
        rr = check_response('يعني "%s" شيء.' % probe, 'u_c1', idx3,
                            'classical_lexicon', chunk_text=unrelated)
        if rr['verdict'] != UNSUPPORTED:
            print('  [FAIL] specificity: %r matched an unrelated chunk -> %s'
                  % (probe, rr['verdict'])); ok = False
    # ...and the same probes must still resolve against their OWN chunk
    rr = check_response('يعني "ططط ظظظ" هي عععع.', 'q_c1', idx2,
                        'classical_lexicon', chunk_text=src_chunk)
    if rr['verdict'] != PARTIAL:
        print('  [FAIL] specificity pin suppressed a genuine item match'); ok = False

    # A quoted item must never reach SUPPORTED - 84%/6% does not justify it.
    for a in rr.get('assertions', []):
        if a['type'] == 'quoted_lexical_item' and a['verdict'] == SUPPORTED:
            print('  [FAIL] quoted_lexical_item reached SUPPORTED'); ok = False

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
