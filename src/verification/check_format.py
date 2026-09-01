# -*- coding: utf-8 -*-
"""Task 3 - Format conformance check.

Input : a response string + its declared format_type
Output: match / mismatch / unknown_format / empty, with the features behind the call

Fully deterministic - regex and counting only. No model, no LLM call.

This replaces the `check_format()` stub in check_dpo.py, which returned 'unavailable'
and made `wrong_formatting` the one charter rejection_type that could never be
corroborated.

The format vocabulary is the pipeline's own, not a generic one
--------------------------------------------------------------
`format_type` is written by chunk.py's `classify_format()`, and its vocabulary is:

    dictionary_entry   glossary entries (fmt=dictionary short-circuits to this)
    verse              stanzas (fmt=verse short-circuits to this)
    footnote_block     >=50% of units start with a footnote marker
    list               >=50% of units start with a list marker
    prose              one body paragraph after headings are dropped
    narrative_paragraph  more than one body paragraph

MEASURED: both corpora are 100% `dictionary_entry` (337 + 394 chunks). Both sources were
ingested with --format dictionary, so the other five values have ZERO instances in any
real data. Only dictionary_entry is validated against genuine corpus content; the rest
are implemented against the same rules chunk.py uses to assign them, and are exercised
only by synthetic cases. That is a real difference in confidence and is why the coverage
table reports per-format status rather than one number.

Reuse, not restatement
----------------------
LIST_MARKER_RE, FOOTNOTE_RE and is_heading() are imported from chunk.py rather than
re-declared, so the check cannot drift from the rule that assigned the label in the first
place. If chunk.py's heuristics change, this follows automatically.

What "conforms" means for a RESPONSE
------------------------------------
A chunk is classified from its units; a generated response has no unit boundaries, so
the test is structural rather than statistical. For dictionary_entry the question is:
does this response actually present lemma-and-gloss pairs, or is it undifferentiated
text? A bag of words carrying every correct term but no entry structure is exactly the
`wrong_formatting` weakness, and it is invisible to the fact and similarity checks -
those see the same vocabulary either way.

Deliberately conservative
-------------------------
`prose` and `narrative_paragraph` are accepted interchangeably. chunk.py separates them
by body-paragraph count, which is a property of how a chunk was packed, not of whether a
response is well-formed - insisting on the distinction would fail responses for a reason
that says nothing about quality.

An unrecognised format_type returns `unknown_format`, never a pass. Absence of a check
must never read as a clean result.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_engineering'))

import chunk as _chunk                                          # noqa: E402

MATCH = 'match'
MISMATCH = 'mismatch'
UNKNOWN_FORMAT = 'unknown_format'
EMPTY = 'empty'

KNOWN_FORMATS = ('dictionary_entry', 'list', 'footnote_block', 'prose',
                 'narrative_paragraph', 'verse')
PROSE_ALIASES = {'prose', 'narrative_paragraph'}

# A lemma the response marks as a cited word. Same convention check_facts.py uses to
# find lexical items, for the same reason: an unmarked word cannot be told apart from
# ordinary running text.
MARKED_LEMMA = re.compile(r'[\(\"«“]([^\)\"»”\n]{1,30})[\)\"»”]')

# Connectives that introduce a gloss. A lemma followed by one of these - or by a colon -
# is an entry; a lemma floating in a sentence is not.
DEFINITIONAL = re.compile(r'(?:معناه|معناها|تعني|يعني|بمعنى|أي\s|وهي\s|وهو\s|هي\s|هو\s|يقال)')
GLOSS_WINDOW = 60          # chars after the lemma in which a gloss marker must appear

SENTENCE_END = re.compile(r'[.؟!؛]')
VERSE_MAX_WORDS_PER_LINE = 12


def _lines(text):
    return [l for l in (text or '').splitlines() if l.strip()]


def features(response):
    """Structural features of a response. Pure, cheap, and reported with every verdict."""
    text = response or ''
    lines = _lines(text)
    n = len(lines) or 1
    listy = sum(1 for l in lines if _chunk.LIST_MARKER_RE.match(l))
    notes = sum(1 for l in lines if _chunk.FOOTNOTE_RE.match(l))

    entries = 0
    for m in MARKED_LEMMA.finditer(text):
        tail = text[m.end():m.end() + GLOSS_WINDOW]
        if tail.lstrip().startswith((':', '：')) or DEFINITIONAL.search(tail):
            entries += 1
    # A colon-initial gloss at line start is the classical form: "root: gloss".
    colon_entries = sum(1 for l in lines
                        if re.match(r'^\s*[^\s:]{2,20}\s*:\s*\S', l))

    words = text.split()
    return {
        'n_lines': len(lines),
        'n_words': len(words),
        'list_ratio': listy / float(n),
        'footnote_ratio': notes / float(n),
        'marked_lemmas': len(MARKED_LEMMA.findall(text)),
        'entry_units': entries,
        'colon_entries': colon_entries,
        'sentence_marks': len(SENTENCE_END.findall(text)),
        'avg_words_per_line': (len(words) / float(n)) if lines else 0.0,
    }


def check_format(response, format_type):
    """Does `response` structurally conform to `format_type`?

    Returns a dict; `status` is one of match / mismatch / unknown_format / empty.
    """
    f = features(response)
    out = {'declared': format_type, 'features': f}

    if not (response or '').strip():
        out.update(status=EMPTY, reason='response is empty')
        return out
    if format_type not in KNOWN_FORMATS:
        out.update(status=UNKNOWN_FORMAT,
                   reason='format_type %r is not one chunk.py emits' % format_type)
        return out

    if format_type == 'dictionary_entry':
        if f['entry_units'] >= 1 or f['colon_entries'] >= 1:
            out.update(status=MATCH,
                       reason='presents %d marked lemma-gloss unit(s) and %d colon '
                              'entr(ies)' % (f['entry_units'], f['colon_entries']))
        else:
            out.update(status=MISMATCH,
                       reason='no lemma-and-gloss structure: %d marked lemma(s), no '
                              'gloss marker or colon entry - undifferentiated text'
                              % f['marked_lemmas'])
        return out

    if format_type == 'list':
        ok = f['list_ratio'] >= 0.5
        out.update(status=MATCH if ok else MISMATCH,
                   reason='%.0f%% of lines carry a list marker (chunk.py threshold 50%%)'
                          % (100 * f['list_ratio']))
        return out

    if format_type == 'footnote_block':
        ok = f['footnote_ratio'] >= 0.5
        out.update(status=MATCH if ok else MISMATCH,
                   reason='%.0f%% of lines carry a footnote marker (threshold 50%%)'
                          % (100 * f['footnote_ratio']))
        return out

    if format_type == 'verse':
        ok = (f['n_lines'] >= 2
              and f['avg_words_per_line'] <= VERSE_MAX_WORDS_PER_LINE
              and f['list_ratio'] < 0.5)
        out.update(status=MATCH if ok else MISMATCH,
                   reason='%d line(s), %.1f words/line (verse expects >=2 short lines)'
                          % (f['n_lines'], f['avg_words_per_line']))
        return out

    # prose / narrative_paragraph, accepted interchangeably
    ok = (f['sentence_marks'] >= 1 and f['list_ratio'] < 0.5
          and f['footnote_ratio'] < 0.5)
    out.update(status=MATCH if ok else MISMATCH,
               reason='%d sentence mark(s), list_ratio %.2f - prose expects continuous '
                      'sentences' % (f['sentence_marks'], f['list_ratio']))
    return out


def compare_format(chosen, rejected, format_type):
    """Which side conforms better. Pure over two strings.

    Returns (relation, chosen_result, rejected_result) where relation is one of
    'chosen_better', 'rejected_better', 'both_match', 'both_mismatch', 'undecidable'.
    """
    c = check_format(chosen, format_type)
    r = check_format(rejected, format_type)
    if c['status'] in (UNKNOWN_FORMAT,) or r['status'] in (UNKNOWN_FORMAT,):
        return 'undecidable', c, r
    cm, rm = c['status'] == MATCH, r['status'] == MATCH
    if cm and not rm:
        return 'chosen_better', c, r
    if rm and not cm:
        return 'rejected_better', c, r
    return ('both_match' if cm else 'both_mismatch'), c, r


# ------------------------------------------------------------------------- self-test

SELF_TEST = [
    # (response, format_type, expected status, note)
    ('الكلمة (خب) وهي أرض مستوية بين الرمال.', 'dictionary_entry', MATCH,
     'marked lemma followed by a gloss connective'),
    ('(ططط) :تعريف تجريبي للاختبار.', 'dictionary_entry', MATCH,
     'the corpus own RTL entry form'),
    ('جذر: شرح الجذر ومعناه.', 'dictionary_entry', MATCH,
     'classical colon-entry form'),
    ('خاطر خطار الضيوف الزائرين خطر القوم نزل عليهم ضيفا خب أرض مستوية',
     'dictionary_entry', MISMATCH,
     'every term present but no entry structure - the wrong_formatting shape'),
    ('هذا كلام متصل عن موضوع ما دون أي بنية معجمية إطلاقا.', 'dictionary_entry',
     MISMATCH, 'continuous prose declared as dictionary_entry'),

    ('- بند أول\n- بند ثان\n- بند ثالث', 'list', MATCH, 'list markers on every line'),
    ('جملة أولى متصلة. جملة ثانية متصلة.', 'list', MISMATCH, 'prose declared as list'),

    ('^( حاشية أولى\n^( حاشية ثانية', 'footnote_block', MATCH, 'footnote markers'),
    ('نص عادي بلا حواشي.', 'footnote_block', MISMATCH, 'prose declared as footnotes'),

    ('بيت أول قصير\nبيت ثان قصير', 'verse', MATCH, 'short lines, no list markers'),
    ('سطر واحد طويل جدا فيه كلمات كثيرة ومتنوعة تتجاوز الحد المقرر للشطر الشعري بكثير.',
     'verse', MISMATCH, 'single long line is not verse'),

    ('هذه جملة تامة. وهذه جملة أخرى.', 'prose', MATCH, 'continuous sentences'),
    ('هذه جملة تامة. وهذه جملة أخرى.', 'narrative_paragraph', MATCH,
     'prose aliases are interchangeable'),
    ('- بند\n- بند\n- بند', 'prose', MISMATCH, 'a list declared as prose'),

    ('نص ما', 'table', UNKNOWN_FORMAT, 'a format chunk.py never emits'),
    ('', 'dictionary_entry', EMPTY, 'empty response'),
    ('   ', 'dictionary_entry', EMPTY, 'whitespace-only response'),
]


def run_self_test():
    ok = True
    for i, (resp, fmt, want, note) in enumerate(SELF_TEST, 1):
        got = check_format(resp, fmt)['status']
        if got != want:
            print('  [FAIL] case %d (%s): got %s expected %s - %s'
                  % (i, fmt, got, want, note))
            ok = False

    # compare_format relations
    good = 'الكلمة (خب) وهي أرض مستوية.'
    bad = 'خب أرض مستوية بين الرمال بلا بنية'
    cases = [
        (good, bad, 'chosen_better'),
        (bad, good, 'rejected_better'),
        (good, good, 'both_match'),
        (bad, bad, 'both_mismatch'),
    ]
    for a, b, want in cases:
        rel = compare_format(a, b, 'dictionary_entry')[0]
        if rel != want:
            print('  [FAIL] compare_format -> %s, expected %s' % (rel, want))
            ok = False
    if compare_format(good, bad, 'table')[0] != 'undecidable':
        print('  [FAIL] unknown format must be undecidable, not a comparison')
        ok = False

    # An unknown format must never be reported as a match.
    for fmt in ('table', 'structured_explanation', None, ''):
        if check_format('أي نص', fmt)['status'] == MATCH:
            print('  [FAIL] unknown format_type %r reported as match' % fmt)
            ok = False

    # Every format chunk.py can emit must be handled.
    for fmt in KNOWN_FORMATS:
        st = check_format('نص تجريبي. سطر آخر.', fmt)['status']
        if st == UNKNOWN_FORMAT:
            print('  [FAIL] %r is emitted by chunk.py but unhandled here' % fmt)
            ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Format conformance check.')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--format-type')
    ap.add_argument('--response')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    if a.response and a.format_type:
        import json
        print(json.dumps(check_format(a.response, a.format_type),
                         ensure_ascii=False, indent=2))
    else:
        ap.print_help()
