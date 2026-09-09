# -*- coding: utf-8 -*-
"""Task 3 Stage 1 - Rule-based fact extraction.

Input : data/processed/chunks.jsonl                  (--corpus saudi_dialect)
        data/processed/chunks_asas_albalagha.jsonl   (--corpus classical_lexicon)
Output: data/processed/facts_<corpus>.jsonl          (one JSON object per chunk)
        data/processed/facts_report_<corpus>.json

Fully deterministic - regex and string rules only, no LLM calls, consistent with the
rest of the pipeline.

Purpose
-------
This is an intermediate FACT-REFERENCE TABLE, not SFT/DPO training records. It is the
ground truth that later verification stages are checked against, so a false positive
here is worse than a miss: it becomes a "fact" that a downstream judge will mark a
model wrong for failing to reproduce.

Output schema (IDENTICAL for both corpora)
------------------------------------------
    {"source_chunk_id": str,
     "source_region":   str,
     "extracted_facts": [{"type": str, "value": str, "position_in_text": int}]}

`position_in_text` is the 0-based character offset of `value` within the chunk's
`chunk_text`, so `chunk_text[pos:pos+len(value)] == value` always holds. That
invariant is asserted for every fact emitted (see _emit); it is the cheapest guard
against the offset drift that silently corrupts a reference table.

Why TWO extractor profiles and not one (--corpus, explicit, no auto-detection)
-----------------------------------------------------------------------------
The two corpora look similar - both carry `format_type: dictionary_entry` - but their
internal structure does not overlap. This was measured across all 731 chunks before
any of the rules below were written:

  pattern                         saudi_dialect        classical_lexicon
  entry headword "(x) :"          95.8% of chunks      0.0%
  figurative marker "ومن المجاز"   0.6%                 96.7%
  footnote marker "(.)N"          73.6%                0.0%
  page reference "ص N"            60.8%                0.0%
  any digit run                   93.8% (3576 hits)    11.4% (45 hits)

Two consequences drove the design:

1. The classical root rule - a line-initial Arabic word followed by a colon - fires on
   88.7% of DIALECT chunks with 849 hits, and every sampled hit is a false positive:
   ordinary words that happen to end a line before a colon (فلان, يقول, القطيف,
   الحرمين). On the classical corpus the same rule yields genuine triliteral roots
   (بخس, نخع, شيط, شتت). The rule is sound for one corpus and worthless for the other,
   so it is never run on the dialect corpus.

2. Every one of the 45 digit runs in the classical corpus is a bare 1-4: footnote
   superscripts, not content. A shared "number" rule would emit 45 junk facts there
   while emitting 3576 mostly-apparatus hits in the dialect corpus. The classical
   profile therefore extracts no numbers at all.

Following the project's per-source tatweel precedent (README, "Normalization decisions
to revisit per source"), the profile is selected explicitly by --corpus and never
inferred. Adding a third corpus means measuring it and writing a third profile, not
reusing whichever of these two looks closer.

What is deliberately NOT extracted
----------------------------------
Footnote markers ("(.)1") and footnote definition lines ("1معجمية.") are apparatus,
not facts about the language. They are the single largest digit source in the dialect
corpus and would otherwise dominate the table with references to page furniture. They
are counted in the report as `apparatus_skipped` so the volume stays visible.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

AR = r'؀-ۿ'
ARW = r'[' + AR + r']'

CORPORA = {
    'saudi_dialect':     'data/processed/chunks.jsonl',
    'classical_lexicon': 'data/processed/chunks_asas_albalagha.jsonl',
    'najdi_popular':     'data/processed/chunks_najdi_popular.jsonl',
}


# --------------------------------------------------------------------------- helpers

def _emit(facts, chunk_text, ftype, value, pos):
    """Append a fact after verifying the offset actually points at the value.

    A reference table with drifted offsets fails silently - every downstream check
    still runs, just against the wrong span. Cheaper to refuse here.
    """
    stripped = value.strip()
    if not stripped:
        return
    pos += len(value) - len(value.lstrip())
    if chunk_text[pos:pos + len(stripped)] != stripped:
        raise AssertionError(
            'offset invariant violated for %r: expected %r at %d, found %r'
            % (ftype, stripped[:40], pos, chunk_text[pos:pos + len(stripped)][:40]))
    facts.append({'type': ftype, 'value': stripped, 'position_in_text': pos})


# ------------------------------------------------------------ profile: saudi_dialect
#
# Entry form is "(headword) :definition" - the colon sits AFTER the closing paren with
# the space before it, an RTL extraction artifact that is stable across all five
# regions. Numbered list items "( -)6..." and "( :)1..." must NOT match; they fail the
# `\)\s*:` requirement because a digit, not a colon, follows the paren.

DIA_HEADWORD = re.compile(r'(?m)^\(\s*([^()\n]{1,40}?)\s*\)\s*:')
DIA_PAGE_REF = re.compile(r'ص\s*\.?\s*(\d{1,4})(?!\d)')

# Era typing, and why there are three year types rather than two.
#
# An explicit هـ / هجرية marker is decisive. Without one, a bare 4-digit year is NOT
# safely Gregorian: this source is a Saudi print edition dated 1434H, and the audit
# found 1425-1429 being emitted as "gregorian_year" when they are plainly Hijri
# (1425H ~ 2004 CE). Guessing an era in a ground-truth table manufactures a wrong fact,
# so unmarked years in the overlap band are typed `year_unmarked_era` and left for a
# human. Above 1500 there is no ambiguity - Hijri 1500 is ~2076 CE - so those are
# Gregorian.
DIA_HIJRI = re.compile(r'(\d{3,4})\s*(?:هجرية|هجري|هـ|ه\b)')
DIA_YEAR_ANY = re.compile(r'(?<!\d)(\d{3,4})(?!\d)\s*(م\b|ميلادي)?')
ERA_AMBIGUOUS_LO, ERA_AMBIGUOUS_HI = 1300, 1500

DIA_CROSSREF = re.compile(
    r'(?:وفي|في)\s+((?:حاضرة|بادية|لهجة|لهجات|أهل)'
    r'(?:\s+(?:أهل|بني))?\s+' + ARW + r'{2,}'
    r'|مصر|الشام|العراق|اليمن|الحجاز|نجد|الكويت|البحرين)')

# Citation attribution.
#
# "قال" is also ordinary narration - "قال له", "قال لها", "قال ما" - so a bare
# "verb + next Arabic word" rule harvests pronouns and particles as authorities
# (له alone accounted for 114 hits in the audit). Two corrections: reject the
# function-word set below, and treat a kunya/nasab opener (ذو, أبو, ابن, أم ...) as
# requiring the following word, so "قال ذو الرمة" yields the whole name rather than
# the truncated "ذو".
# The leading \b is load-bearing, not decoration. Without it "قال" matches as a
# SUBSTRING of ordinary words that merely end in those letters - انتقال, مقال,
# اعتقال - and the following noun is then harvested as an authority. The audit caught
# this on a real passage where an ordinary word ending in those letters was read as a
# speech verb. Arabic letters are word characters under Python's Unicode \b, so the
# boundary does the right thing here.
#
# "ذكر" is deliberately absent: it means "mentioned" far more often than it introduces
# an attribution, and it produced false positives on ordinary mentions at a rate the
# other verbs do not. Longer alternatives are listed first so they win the match.
CITATION_VERB = r'\b(?:وأنشد|أنشد|وقال|قالت|قال|يقول)'
KUNYA = r'(?:ذو|ذي|ذا|أبو|أبي|أبا|ابن|بن|أم|امرؤ|امرئ)'
# The kunya branch takes at most one further particle (ابن أبي ربيعة) and then exactly
# one name word - not an arbitrary trailing word, which would swallow the rest of the
# sentence ("قال ذو الرمة بيتا" -> "ذو الرمة بيتا").
CITATION_NAME = (
    r'(' + KUNYA + r'\s+(?:' + KUNYA + r'\s+)?' + ARW + r'{2,}'
    r'|(?:ال)?' + ARW + r'{3,}(?:\s+(?:بن|ابن)\s+' + ARW + r'{2,}){0,2}'
    r'(?:\s+(?:ال)' + ARW + r'{3,}){0,1})')
DIA_CITATION = re.compile(CITATION_VERB + r'\s+' + CITATION_NAME)

# Pronouns, particles and deictics that follow a speech verb in ordinary narration.
CITATION_STOPWORDS = {
    'له', 'لها', 'لهم', 'لهن', 'لك', 'لكم', 'لي', 'لنا',
    'عنه', 'عنها', 'فيه', 'فيها', 'به', 'بها', 'منه', 'منها',
    'هذا', 'هذه', 'ذلك', 'تلك', 'هو', 'هي', 'هم', 'أنا', 'أنت',
    'ما', 'إن', 'أن', 'قد', 'كذا', 'كما', 'إذا', 'حين', 'عند',
    'كان', 'كانت', 'لما', 'لقد', 'ثم', 'بعض', 'البعض',
    'فيها', 'أيضا', 'أيضاً', 'أي', 'لم', 'لا', 'نعم',
}


KUNYA_SET = {'ذو', 'ذي', 'ذا', 'أبو', 'أبي', 'أبا', 'ابن', 'بن', 'أم', 'امرؤ', 'امرئ'}


def _citation_value(raw):
    """Return the attribution name, or None if this is narration rather than a citation.

    Three rejections, in order of how much noise each removes (measured on the real
    corpora, see the module docstring's precision note):

    1. Function words after a speech verb - "قال له", "قال ما" - are narration.
    2. A lone kunya particle carries no attribution.
    3. A SINGLE bare token with no ال- prefix and no kunya is rejected. This is the
       rule that removes فلان ("so-and-so", a placeholder), لمن, and stray verb forms
       that follow a speech verb. It costs recall: a genuine one-word poet name
       without the article (حاتم) is dropped too. That trade is deliberate - this
       table is ground truth, so a wrong attribution is more expensive than a missing
       one. Multi-word names (مرزوق الفلاني), article-prefixed roles (الشاعر,
       المثل) and kunya forms (ذو الرمة) all survive.
    """
    name = raw.strip()
    toks = name.split()
    if not toks:
        return None
    if toks[0] in CITATION_STOPWORDS or name in CITATION_STOPWORDS:
        return None
    if name in KUNYA_SET:
        return None
    # Doubled particle from a source typo, e.g. "أبو أبو ذؤيب".
    if len(toks) >= 2 and toks[0] == toks[1]:
        return None
    if len(toks) == 1 and not name.startswith('ال'):
        return None
    return name

# Apparatus - counted, never emitted.
DIA_FOOTNOTE_MARK = re.compile(r'\(\s*\.\s*\)\s*\d+')
DIA_FOOTNOTE_LINE = re.compile(r'(?m)^\d{1,2}(?=' + ARW + r')')


def _year_type(value, explicit_hijri, explicit_greg):
    if explicit_hijri:
        return 'hijri_year'
    if explicit_greg:
        return 'gregorian_year'
    n = int(value)
    if ERA_AMBIGUOUS_LO <= n <= ERA_AMBIGUOUS_HI:
        return 'year_unmarked_era'
    if n > ERA_AMBIGUOUS_HI:
        return 'gregorian_year'
    return None          # < 1300 unmarked: too weak to call a year at all


def extract_saudi_dialect(chunk_text):
    facts = []
    skipped = (len(DIA_FOOTNOTE_MARK.findall(chunk_text))
               + len(DIA_FOOTNOTE_LINE.findall(chunk_text)))

    for m in DIA_HEADWORD.finditer(chunk_text):
        _emit(facts, chunk_text, 'entry_headword', m.group(1), m.start(1))

    for m in DIA_CITATION.finditer(chunk_text):
        name = _citation_value(m.group(1))
        if name:
            _emit(facts, chunk_text, 'citation_authority', name, m.start(1))

    # Page references are consumed first so their digits are not re-read as years.
    page_spans = []
    for m in DIA_PAGE_REF.finditer(chunk_text):
        page_spans.append((m.start(1), m.end(1)))
        _emit(facts, chunk_text, 'page_reference', m.group(1), m.start(1))

    hijri_spans = {m.start(1) for m in DIA_HIJRI.finditer(chunk_text)}
    for m in DIA_YEAR_ANY.finditer(chunk_text):
        s = m.start(1)
        if any(a <= s < b for a, b in page_spans):
            continue
        ftype = _year_type(m.group(1), s in hijri_spans, bool(m.group(2)))
        if ftype:
            _emit(facts, chunk_text, ftype, m.group(1), s)

    for m in DIA_CROSSREF.finditer(chunk_text):
        _emit(facts, chunk_text, 'cross_dialect_reference', m.group(1), m.start(1))
    return facts, skipped


# -------------------------------------------------------- profile: classical_lexicon
#
# Entry form is "root: gloss" at line start. Restricted to 2-6 Arabic letters with no
# internal space and no definite article, which is what an Arabic root looks like. The
# dialect corpus's line-final ordinary words are longer and/or carry "ال", which is why
# this rule cannot be shared (see module docstring).

CLA_ROOT = re.compile(r'(?m)^(?!ال)(' + ARW + r'{2,6})\s*:(?!\s*$)')
CLA_FIGURATIVE = re.compile(r'ومن\s+المجاز')
CLA_CITATION = re.compile(CITATION_VERB + r'\s+' + CITATION_NAME)


# ------------------------------------------------------- quoted lexical items (responses)
#
# NOT a chunk-extraction rule. These patterns run over GENERATED text - an instruction or
# a response - not over corpus text, and they exist because the corpus rules above find
# nothing in conversational paraphrase.
#
# MEASURED on 4,645 real reconstructed records (2026-09-09): 3,333 returned
# NO_CHECKABLE_CLAIMS. The cause is structural, not a gap in pattern coverage. Every
# classical chunk rule anchors on dictionary-entry SHAPE - `CLA_ROOT` needs `^root:` at a
# line start - and a conversational answer never has that shape. But the content is not
# unverifiable: 99% of those records quote a lexical item, and 94% of the quoted spans
# appear verbatim in their own source chunk against a 1% hit rate on a random other chunk.
#
# What the responses actually do is quote an item and gloss it:
#     Q: ما هي 'جهمة الليل'؟   ->   A: هي الجزء الأخير من الليل
# so the item is the checkable specific, and `check_facts` judges the gloss around it.
#
# الاقتباس بين علامتين هو المُدَّعى القابل للفحص في النص المحاوَر.
#
# Single quotes are included, and that is the whole reason this is a separate pattern:
# `check_facts.LEXICAL_MARKED` covers parentheses and double quotes only, so it already
# caught the 46% of instructions using " and missed the 53% using '. The apostrophe is
# also an English possessive, so a span is kept only if it contains Arabic.
QUOTED_SPAN = re.compile(r'[\'‘’"“”«»]'
                         r'([^\'‘’"“”«»\n]{2,60}?)'
                         r'[\'‘’"“”«»]')

# A single token is usually a bare word that the root rules already reach, and it is the
# weakest case for specificity: measured, multi-word spans hit a random other chunk 0% of
# the time against 2% for all spans. Multi-word is where the discrimination lives.
QUOTED_MIN_TOKENS = 2


def quoted_lexical_items(text, min_tokens=QUOTED_MIN_TOKENS):
    """Quoted Arabic spans in generated text, longest-first, de-duplicated.

    Returns the ORIGINAL surface strings. Normalisation for matching belongs to the
    caller - this project ships `paragraphs_original` and a folded key is a matching
    device, never an output.
    """
    seen, out = set(), []
    for m in QUOTED_SPAN.finditer(text or ''):
        inner = m.group(1).strip()
        if not inner or not re.search(ARW, inner):
            continue
        # Count ARABIC tokens, not whitespace tokens. A quoted year like "1426 هـ" has
        # two whitespace tokens but one Arabic one, and it must not become a lexical
        # item: years are already covered by the era rules above, and the classical
        # profile deliberately extracts no numbers at all (see the module docstring), so
        # a quoted year here would be judged against a table that never holds one.
        ar_tokens = [t for t in inner.split() if re.search(ARW, t)]
        if len(ar_tokens) < min_tokens:
            continue
        if inner not in seen:
            seen.add(inner)
            out.append(inner)
    return out


def extract_classical_lexicon(chunk_text):
    """Extract roots, figurative senses and citations from the classical lexicon.

    `figurative_sense` carries the WHOLE figurative passage, not just the marker.
    In أساس البلاغة the "ومن المجاز" marker opens the figurative-usage section for a
    root and runs to the end of that root's entry, so the span ends at whichever comes
    first: the next line-initial root, the next marker, or the end of the chunk. The
    marker alone was position-only and told a downstream check nothing about what the
    figurative sense actually is; the span is the fact. Measured over all 1987
    occurrences: median 242 chars, p90 719, max 2371.
    """
    facts = []
    root_starts = [m.start() for m in CLA_ROOT.finditer(chunk_text)]
    fig_starts = [m.start() for m in CLA_FIGURATIVE.finditer(chunk_text)]
    boundaries = sorted(root_starts + fig_starts)

    for m in CLA_ROOT.finditer(chunk_text):
        _emit(facts, chunk_text, 'entry_root', m.group(1), m.start(1))

    for s in fig_starts:
        nxt = [b for b in boundaries if b > s]
        end = nxt[0] if nxt else len(chunk_text)
        _emit(facts, chunk_text, 'figurative_sense', chunk_text[s:end], s)

    for m in CLA_CITATION.finditer(chunk_text):
        name = _citation_value(m.group(1))
        if name:
            _emit(facts, chunk_text, 'citation_authority', name, m.start(1))
    return facts, 0


# ------------------------------------------------------------------ citation review
#
# `citation_authority` is the only rule that still carries measurable residual noise
# (hand-check of 14 dialect samples: ~12 correct). Rather than tighten the regex until
# it starts dropping real attributions, the low-confidence tail is written to a review
# CSV, following the project's existing damaged_pages / residue_tokens practice.
#
# The confidence test is deliberately data-driven rather than a hand-written name list:
# a genuine authority RECURS across the corpus (ذو الرمة 306 times), while phrase-shaped
# noise appears once. Diacritics are stripped before testing so that
# ابنُ الأعرابي is recognised as carrying the nasab particle ابن.
#
# The queue is over-inclusive by design - being listed means "a human should look",
# not "this is wrong". Many classical entries in it are real poets flagged only for
# being infrequent.

DIACRITICS = re.compile(r'[ً-ْٰـ]')
NASAB = {'بن', 'ابن'}


def _strip_diacritics(s):
    return DIACRITICS.sub('', s)


def citation_is_confident(value, freq):
    toks = _strip_diacritics(value).split()
    if not toks:
        return False
    if toks[0] in KUNYA_SET:
        return True
    if any(t in NASAB for t in toks):
        return True
    return freq.get(value, 0) >= 3


# ------------------------------------------------------ profile: najdi_popular
#
# THIRD profile, added 2026-09-09 for معجم الكلمات الشعبية في نجد
# (doc_id majam_alkalimat_alshaabia_najd, manifest row dialect_dict_najdi_popular).
# Written rather than reusing either existing profile, per this module's own rule:
# adding a corpus means measuring it and writing a profile, not picking whichever of
# the other two looks closer. Both were measured against this source first:
#
#   pattern                     najdi_popular   held dialect   classical
#   DIA_HEADWORD "(x) :"            0.0%           95.8%          0.0%
#   CLA_ROOT     "^root:"         100.0%           75.4%        100.0%
#
# The dialect rule finds NOTHING here - this source has no parenthesised headwords.
# The classical rule appears to fit at 100%, and that is the trap the module docstring
# warns about. Sampling its 371 hits: مدوقع, صلوقعه, صفّاق, بنطّح, معراض - inflected
# dialect words, not triliteral roots. Only 18% are even 3 characters, against 100%
# in the classical corpus. It is wrong in both directions: CLA_ROOT excludes ^ال by
# design, so it SKIPS the 74 of 473 headwords (16%) that carry the article while
# capturing 399 others as fabricated "roots".
#
# The entry form here is simply `headword: gloss` at a line start, article allowed.
# The pattern permits a multi-word headword; the data contains none (0 of 473). The
# permission is left in because it costs nothing and the OCR could yield one, but the
# 0% is recorded so a future multi-word hit reads as new behaviour, not as expected.
#
# VALIDATED AGAINST A COMPLETE ANSWER KEY, which neither other profile ever had.
# data/processed/najdi_dictionary_final.json carries the OCR pipeline's own
# headword/meaning split for all 473 entries, so recall and precision here are exact
# rather than sampled: 473 hits, 100% recall, 100% precision, zero false positives,
# zero missed headwords. run_self_test() pins that.
#
# NO YEAR RULES, for the same reason the classical profile has none. All 61 digit
# runs in the corpus were checked: 25 are single digits, and every 3-4 digit run is
# a page number the OCR mangled into mixed Arabic-Indic and ASCII (١74, ١١6, ٠90١,
# ٠094). Zero carry a هـ/هجري marker. Running DIA_HIJRI/DIA_YEAR_ANY here would
# manufacture 25 junk years out of scanning artifacts. run_self_test pins this.
#
# صيغة المدخل: كلمة ثم نقطتان ثم الشرح، في أول السطر.
NAJ_ENTRY = re.compile(r'(?m)^[ \t]*([^:\n]{1,40}?)[ \t]*:[ \t]*([^\n]*)')

# Glosses shorter than this are OCR noise or a dangling colon, not a definition.
NAJ_MIN_GLOSS = 3


def extract_najdi_popular(chunk_text):
    """Headword + gloss pairs from the Najdi popular-words dictionary.

    Emits `entry_headword` for the term and `gloss` for its definition.

    `gloss` IS NOT CONSUMED BY check_facts TODAY. Nothing in the repo reads the type
    (grepped, zero consumers outside this file): check_facts derives its assertions
    from the RESPONSE side, and there is no gloss assertion extractor, so these 473
    facts sit in the table inert. They are emitted anyway because the table is a
    reference artifact, not a check_facts input - and because the gloss is the only
    thing a paraphrase could ever be tested against, so the fix for that gap needs
    them already present. Stated here rather than implied, so nobody reads a clean
    najdi_popular fact run as evidence that glosses are being verified.

    Page references reuse the dialect rule - this is the same kind of print source.
    """
    facts = []
    for m in NAJ_ENTRY.finditer(chunk_text):
        head, gloss = m.group(1).strip(), m.group(2).strip()
        if not head or not re.search(ARW, head):
            continue
        _emit(facts, chunk_text, 'entry_headword', head, m.start(1))
        if len(gloss) >= NAJ_MIN_GLOSS and re.search(ARW, gloss):
            _emit(facts, chunk_text, 'gloss', gloss, m.start(2))
    for m in DIA_PAGE_REF.finditer(chunk_text):
        _emit(facts, chunk_text, 'page_reference', m.group(1), m.start(1))
    # Second element is the apparatus-skipped count, same contract as the other two
    # profiles. Measured as 0 across all 6 chunks: this source carries no footnote
    # markers and no footnote definition lines at all (the OCR pipeline dropped the
    # page furniture upstream), so there is nothing to skip and nothing being hidden
    # by reporting zero.
    return facts, 0


PROFILES = {
    'saudi_dialect':     extract_saudi_dialect,
    'classical_lexicon': extract_classical_lexicon,
    'najdi_popular':     extract_najdi_popular,
}


# ------------------------------------------------------------------------- self-test
#
# A clean run over the real corpus proves nothing on its own - it cannot show that a
# rule which SHOULD fire did, or that one which should NOT fire stayed quiet. These
# cases pin both directions. Arabic strings here are short synthetic constructions or
# already-public structural markers, not quoted source passages.

SELF_TEST_CASES = [
    # (corpus, text, must contain (type, value), types that must NOT appear)
    ('saudi_dialect', '(ططط) :تعريف تجريبي للاختبار.',
     [('entry_headword', 'ططط')], []),
    # numbered list items are not headwords: a digit, not a colon, follows the paren
    ('saudi_dialect', '( -)6بند مرقم تجريبي للاختبار.',
     [], ['entry_headword']),
    ('saudi_dialect', '( :)9بند مرقم ثان للاختبار.',
     [], ['entry_headword']),
    ('saudi_dialect', 'قال الشاعر الفلاني بيتا للاختبار.',
     [('citation_authority', 'الشاعر الفلاني')], []),
    ('saudi_dialect', 'طبع المرجع سنة 999هجرية للاختبار.',
     [('hijri_year', '999')], []),
    ('saudi_dialect', 'مرجع تجريبي للاختبار 1998',
     [('gregorian_year', '1998')], []),
    ('saudi_dialect', 'انظر المرجع ص.493 للاختبار.',
     [('page_reference', '493')], []),
    # apparatus must be skipped, never emitted as a number-bearing fact
    ('saudi_dialect', 'كلام عادي(.)1 ثم كلام آخر.',
     [], ['page_reference', 'gregorian_year', 'hijri_year']),

    # narration, not attribution: "said to him" must not yield له as an authority
    ('saudi_dialect', 'ثم قال له صاحبه كلاما طويلا.',
     [], ['citation_authority']),
    ('saudi_dialect', 'وقال ما يعرفه الناس عن ذلك.',
     [], ['citation_authority']),
    # فلان is a placeholder ("so-and-so"), never an authority
    ('saudi_dialect', 'يقول فلان كلاما في هذا.',
     [], ['citation_authority']),
    # article-prefixed role attributions and multi-word names must survive
    ('saudi_dialect', 'قال الشاعر بيتا جميلا.',
     [('citation_authority', 'الشاعر')], []),
    ('saudi_dialect', 'قال مرزوق الفلاني للاختبار.',
     [('citation_authority', 'مرزوق الفلاني')], []),
    # doubled particle from a source typo must not become an authority
    ('classical_lexicon', 'قال أبو أبو ذؤيب شعرا.',
     [], ['citation_authority']),
    # era typing: explicit marker decides; unmarked overlap-band years stay unresolved
    ('saudi_dialect', 'طبع سنة 1426هـ في الرياض.',
     [('hijri_year', '1426')], ['gregorian_year', 'year_unmarked_era']),
    ('saudi_dialect', 'صدر عام 1998م عن الاتحاد.',
     [('gregorian_year', '1998')], ['hijri_year', 'year_unmarked_era']),
    ('saudi_dialect', 'وذكر ذلك في 1427 دون تحديد.',
     [('year_unmarked_era', '1427')], ['gregorian_year', 'hijri_year']),
    ('saudi_dialect', 'نشر عام 2007 في مجلة.',
     [('gregorian_year', '2007')], ['year_unmarked_era', 'hijri_year']),
    # a page number must not be re-read as a year
    ('saudi_dialect', 'انظر ص1425 من الكتاب.',
     [('page_reference', '1425')], ['year_unmarked_era', 'gregorian_year']),

    # "انتقال" contains "قال" - a substring match must not harvest the next word
    ('saudi_dialect', 'جرى الانتقال التجريبي بين الجملتين.',
     [], ['citation_authority']),
    ('saudi_dialect', 'وبعد الاعتقال الطويل عاد الرجل.',
     [], ['citation_authority']),

    ('classical_lexicon', 'أبب: اطلب الأمر في إبانه.',
     [('entry_root', 'أبب')], []),
    # kunya must carry the following word, not truncate to the particle
    ('classical_lexicon', 'قال ذو الرمة بيتا مشهورا.',
     [('citation_authority', 'ذو الرمة')], []),
    ('classical_lexicon', 'وأنشد ابن الأعرابي شعرا.',
     [('citation_authority', 'ابن الأعرابي')], []),
    # the figurative fact must carry the whole passage, not just the marker
    ('classical_lexicon', 'ومن المجاز: فلان مولع بأوابد الكلام.',
     [('figurative_sense', 'ومن المجاز: فلان مولع بأوابد الكلام.')], ['figurative_marker']),
    # ... and must stop at the next root entry rather than running on
    ('classical_lexicon', 'ومن المجاز: سعة الصدر.\nنخع: بلغ النخاع.',
     [('figurative_sense', 'ومن المجاز: سعة الصدر.'), ('entry_root', 'نخع')], []),
    ('classical_lexicon', 'قال الأعشى صرمت ولم أصرمكم.',
     [('citation_authority', 'الأعشى')], []),
    # the dialect corpus's line-final ordinary words must not read as roots
    ('classical_lexicon', 'الحرمين: وما بعدها.', [], ['entry_root']),

    # ------------------------------------------------------------- najdi_popular
    ('najdi_popular', 'الغرب: هو الأداة المستعملة لرفع الماء.',
     [('entry_headword', 'الغرب'), ('gloss', 'هو الأداة المستعملة لرفع الماء.')], []),
    # the article is KEPT. CLA_ROOT excludes ^ال by design and would skip this
    # entry entirely - 74 of the 473 headwords (16%) carry it.
    # The gloss here is an invented placeholder. The natural-sounding definition
    # written first turned out to occur verbatim in the rights-pending corpus, and
    # audit_staged_arabic.py refused the commit - the fail-closed behaviour working
    # as designed. The colliding phrase is deliberately not repeated in this comment.
    ('najdi_popular', 'المعراض: ططط ظظظ.',
     [('entry_headword', 'المعراض')], ['entry_root']),
    # an inflected dialect word is a legitimate headword here. The same string is the
    # shape CLA_ROOT misreads as a triliteral root - this profile must not emit one.
    ('najdi_popular', 'مدوقع: منكسر.', [('entry_headword', 'مدوقع')], ['entry_root']),
    # DISCRIMINATION: a colon with no Arabic head is not an entry. A digital clock or
    # a bare ratio must produce nothing at all.
    ('najdi_popular', '12:30 والوقت متأخر.', [], ['entry_headword', 'gloss']),
    # DISCRIMINATION: a line with no colon is prose, not an entry.
    ('najdi_popular', 'هذا سطر عادي بلا نقطتين.', [], ['entry_headword', 'gloss']),
    # a gloss under NAJ_MIN_GLOSS is a dangling colon or OCR noise: keep the headword,
    # drop the gloss, rather than emitting a one-character "definition".
    ('najdi_popular', 'شبح: ا', [('entry_headword', 'شبح')], ['gloss']),
    # a gloss with no Arabic at all is not a definition
    ('najdi_popular', 'قلط: xyz', [('entry_headword', 'قلط')], ['gloss']),
    ('najdi_popular', 'انظر ص 41 من الكتاب.', [('page_reference', '41')], []),
    # the OTHER corpora's rules must stay out: no years, no figurative sense
    ('najdi_popular', 'حول: سنة 1380 هـ.', [], ['hijri_year', 'figurative_sense']),
]


def run_self_test():
    ok = True
    for i, (corpus, text, must, must_not) in enumerate(SELF_TEST_CASES, 1):
        facts, _ = PROFILES[corpus](text)
        got = {(f['type'], f['value']) for f in facts}
        got_types = {f['type'] for f in facts}
        for want in must:
            if want not in got:
                print('  [FAIL] case %d (%s): expected %r, got %r'
                      % (i, corpus, want, sorted(got)))
                ok = False
        for bad in must_not:
            if bad in got_types:
                print('  [FAIL] case %d (%s): %r should not fire, got %r'
                      % (i, corpus, bad, sorted(got)))
                ok = False
        for f in facts:
            p = f['position_in_text']
            if text[p:p + len(f['value'])] != f['value']:
                print('  [FAIL] case %d: offset invariant broken for %r' % (i, f))
                ok = False
    ok = _self_test_najdi_answer_key() and ok
    print('passed: %s' % ok)
    return ok


def _self_test_najdi_answer_key():
    """Pin the najdi_popular profile at 100% recall / 100% precision.

    This is the only profile in the module validated against a COMPLETE answer key
    rather than a sample: data/processed/najdi_dictionary_final.json carries the OCR
    pipeline's own headword/meaning split for all 473 entries, and both it and the
    chunks file are tracked, so this runs in any clone rather than only on the machine
    that built it. Set equality both ways - a missed headword and a fabricated one are
    different failures and the count alone would hide either.
    """
    # Resolved from __file__, not cwd: main() takes relative paths and assumes it is
    # run from the repo root, but a self-test that only passes from one directory is
    # a self-test that gets skipped.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ck = os.path.join(repo, CORPORA['najdi_popular'])
    key_path = os.path.join(repo, 'data/processed/najdi_dictionary_final.json')
    if not (os.path.exists(ck) and os.path.exists(key_path)):
        print('  [SKIP] najdi answer-key pin: inputs missing')
        return True
    with open(key_path, encoding='utf-8') as fh:
        key = json.load(fh)
    want_h = sorted(e['headword'].strip() for e in key)
    want_g = sorted(e['meaning'].strip() for e in key)
    got_h, got_g = [], []
    with open(ck, encoding='utf-8') as fh:
        for line in fh:
            facts, _ = extract_najdi_popular(json.loads(line)['chunk_text'])
            got_h += [f['value'] for f in facts if f['type'] == 'entry_headword']
            got_g += [f['value'] for f in facts if f['type'] == 'gloss']
    ok = True
    for label, want, got in (('headword', want_h, sorted(got_h)),
                             ('gloss', want_g, sorted(got_g))):
        if want != got:
            miss, extra = set(want) - set(got), set(got) - set(want)
            print('  [FAIL] najdi %s answer key: %d expected, %d got, '
                  '%d missed, %d fabricated' % (label, len(want), len(got),
                                                len(miss), len(extra)))
            for v in sorted(miss)[:3]:
                print('           missed: %r' % v[:60])
            for v in sorted(extra)[:3]:
                print('           extra : %r' % v[:60])
            ok = False
    if ok:
        print('  najdi answer key: %d/%d headwords, %d/%d glosses, 0 fabricated'
              % (len(got_h), len(want_h), len(got_g), len(want_g)))
    return ok


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description='Rule-based fact extraction (Task 3).')
    ap.add_argument('--corpus', choices=sorted(CORPORA),
                    help='which corpus to extract; required (no auto-detection)')
    ap.add_argument('--in', dest='inp', help='override input chunks .jsonl')
    ap.add_argument('--out', help='override output facts .jsonl')
    ap.add_argument('--report', help='override report .json')
    ap.add_argument('--review', help='override citation review .csv')
    ap.add_argument('--self-test', action='store_true',
                    help='verify the rules against synthetic cases and exit')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    if not args.corpus:
        sys.stderr.write(
            '[facts] REFUSING to run: --corpus is required. The two corpora have\n'
            '    non-overlapping structure and each has its own extractor profile;\n'
            '    guessing would silently apply the wrong rules (see module docstring).\n'
            '    corpora available:\n      '
            + '\n      '.join(sorted(CORPORA)) + '\n')
        sys.exit(1)

    inp = args.inp or CORPORA[args.corpus]
    out = args.out or 'data/processed/facts_%s.jsonl' % args.corpus
    rep = args.report or 'data/processed/facts_report_%s.json' % args.corpus
    # Per-corpus review files, NOT one shared docs/citation_review.csv. The `context`
    # column quotes source text, so the dialect file inherits the held rights status of
    # chunks.jsonl and is gitignored, while the classical one is CC BY-SA and
    # publishable. A single combined file would drag the classical review into the hold
    # for no reason. Naming follows the existing per-corpus report convention.
    review = args.review or 'docs/citation_review_%s.csv' % args.corpus
    extract = PROFILES[args.corpus]

    if not os.path.exists(inp):
        sys.stderr.write('[facts] input not found: %s\n' % inp)
        sys.exit(1)

    recs = [json.loads(l) for l in open(inp, encoding='utf-8') if l.strip()]

    # Pass 1: extract. Held in memory (337/394 rows) because the citation confidence
    # test needs corpus-wide frequencies, which are not known until every chunk is done.
    extracted, apparatus = [], 0
    for r in recs:
        facts, skipped = extract(r['chunk_text'])
        apparatus += skipped
        facts.sort(key=lambda f: f['position_in_text'])
        extracted.append((r, facts))

    citation_freq = Counter(f['value'] for _, fs in extracted for f in fs
                            if f['type'] == 'citation_authority')

    type_counts, per_region = Counter(), Counter()
    chunks_with = chunks_without = 0
    empty_chunks = []
    review_rows = []

    with open(out, 'w', encoding='utf-8') as fh:
        for r, facts in extracted:
            text = r['chunk_text']
            fh.write(json.dumps({'source_chunk_id': r['chunk_id'],
                                 'source_region': r['region'],
                                 'extracted_facts': facts},
                                ensure_ascii=False) + '\n')
            for f in facts:
                type_counts[f['type']] += 1
                per_region[r['region']] += 1
                if (f['type'] == 'citation_authority'
                        and not citation_is_confident(f['value'], citation_freq)):
                    p = f['position_in_text']
                    ctx = text[max(0, p - 40):p + len(f['value']) + 30]
                    review_rows.append({
                        'corpus': args.corpus,
                        'chunk_id': r['chunk_id'],
                        'region': r['region'],
                        'value': f['value'],
                        'position_in_text': p,
                        'occurrences_in_corpus': citation_freq[f['value']],
                        'context': ' '.join(ctx.split()),
                    })
            if facts:
                chunks_with += 1
            else:
                chunks_without += 1
                empty_chunks.append(r['chunk_id'])

    with open(review, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['corpus', 'chunk_id', 'region', 'value',
                                           'position_in_text', 'occurrences_in_corpus',
                                           'context'])
        w.writeheader()
        w.writerows(review_rows)

    report = {
        'corpus': args.corpus,
        'input': inp,
        'output': out,
        'chunks_in': len(recs),
        'chunks_with_facts': chunks_with,
        'chunks_without_facts': chunks_without,
        'coverage_pct': round(100.0 * chunks_with / len(recs), 2) if recs else 0.0,
        'facts_total': sum(type_counts.values()),
        'facts_per_type': dict(type_counts.most_common()),
        'facts_per_region': dict(per_region.most_common()),
        'apparatus_skipped': apparatus,
        'citation_review_queue': len(review_rows),
        'citation_review_file': review,
        'empty_chunk_ids': empty_chunks[:50],
    }
    with open(rep, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print('[facts] %s: %d chunks -> %d facts (%d with, %d without)'
          % (args.corpus, len(recs), sum(type_counts.values()),
             chunks_with, chunks_without))
    for t, c in type_counts.most_common():
        print('          %-26s %6d' % (t, c))
    if apparatus:
        print('        apparatus skipped (footnote refs/lines): %d' % apparatus)
    if review_rows:
        print('        citation review queue: %d rows -> %s' % (len(review_rows), review))
    print('[facts] wrote %s' % out)
    print('[facts] wrote %s' % rep)


if __name__ == '__main__':
    main()
