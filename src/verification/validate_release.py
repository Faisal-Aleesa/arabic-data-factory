# -*- coding: utf-8 -*-
"""Task 3 Group D item 13 - Final formatting / encoding validator.

Input : any released SFT or DPO .jsonl
Output: a finding list; exit 1 if any ERROR is present

الفحص الأخير قبل الإصدار: سلامة JSON، والترميز، والإملاء، والحقول المطلوبة. هو نفس
الانضباط الذي طبّقته مرحلة هندسة البيانات، يُعاد تطبيقه مرة أخيرة على المُخرج النهائي.

The last-mile check, run once more at the very end. Everything upstream already
validates its own stage output; this exists because a release file is assembled, copied
and rewritten after those stages finish, and each of those steps can reintroduce exactly
the damage the pipeline spent effort removing.

Reuses data_engineering rather than restating it
------------------------------------------------
  ftfy.fix_text              the same encoding repair clean.py runs (clean.py:386)
  clean.ALEF_VARIANTS        the exact fold table, so "has this been folded?" cannot
  clean.YA_VARIANTS          drift from what clean.py actually folds
  clean.normalize_arabic     used to demonstrate the fold in the self-test

If clean.py's fold table changes, this check follows automatically.

What it checks
--------------
  JSON            every line parses; the line number is reported
  SCHEMA          required fields present and non-empty for the record type
  ENCODING        mojibake (ftfy would change the text), U+FFFD replacement chars,
                  non-NFC normalisation, control characters, and invisible
                  zero-width / bidi marks
  ORTHOGRAPHY     text that has been alef/ya FOLDED when the project ships original
                  orthography

The orthography rule, and where its threshold comes from
--------------------------------------------------------
The project ships `paragraphs_original` - hamza forms (أ إ آ) and alef maksura (ى)
exactly as written - and folds only for dedup matching. A released record whose Arabic
carries none of those variants has been folded somewhere it should not have been.

MEASURED across all 731 chunks of both corpora:

  chunks containing >= 1 orthography variant   337/337 and 394/394  = 100%
  variants per word, minimum observed          0.0609  (dialect)
  variants per word, median                    0.1135 dialect / 0.1722 classical
  the same text after normalize_arabic()       0.0000

So real Arabic text in this project never drops below ~0.06 variants per word, and
folded text sits at exactly zero. The separation is total, which is why the rule is
binary: zero variants is the signal, not "few".

The only risk is a short text that legitimately contains none, so the check applies only
at MIN_WORDS_FOR_ORTHOGRAPHY or longer. At the minimum observed rate a 25-word passage
expects ~1.5 variants and at the p10 rate ~2.2, so zero in 25+ words is genuinely
anomalous rather than merely unlucky.

Findings cascade, and that is intended
--------------------------------------
One broken record often raises several findings: a mojibake round-trip that destroys the
Arabic will also trip REPLACEMENT_CHAR and FOLDED_ORTHOGRAPHY, because after the damage
there genuinely are no orthography variants left. All three are true statements about the
text. Read the first ERROR on a line as the likely root cause rather than counting
findings as if they were independent defects.

تتراكم النتائج على السطر الواحد عمدا: تلف الترميز يُسقط الإملاء أيضا، وكلها صحيحة.
اقرأ أول خطأ في السطر بوصفه السبب الجذري.

What this CANNOT do
-------------------
- It cannot tell a legitimately unfolded-but-variant-free text from a folded one below
  the word threshold. Short responses are simply not checked for orthography.
- It validates FORM, not content. A record can pass every check here and still be
  factually wrong, ungrounded or badly formatted - that is what verify_sft.py,
  check_facts.py and check_format.py are for.
- Tatweel is deliberately NOT checked. It is a per-source decision in this project -
  the dialect corpus keeps it, the classical corpus strips it - so its presence or
  absence in a release file is not evidence of anything on its own.
- Until real released output exists, it has only been exercised against hand-built
  broken records.
"""

import argparse
import json
import os
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'data_engineering'))

import ftfy                                                     # noqa: E402
import clean as _clean                                          # noqa: E402

ERROR = 'ERROR'
WARN = 'WARN'

# The exact fold table clean.py uses; imported, not restated.
ORTHOGRAPHY_VARIANTS = frozenset(set(_clean.ALEF_VARIANTS) | set(_clean.YA_VARIANTS))

# Derived from the measurement in the module docstring: 100% of 731 real chunks carry
# variants, at a minimum of 0.0609 per word. A 25-word passage expects ~1.5 at that
# minimum, so zero is anomalous rather than unlucky.
MIN_WORDS_FOR_ORTHOGRAPHY = 25

SCHEMAS = {
    'sft': ('instruction', 'response', 'source_chunk_id', 'source_region',
            'format_type', 'model_version'),
    'dpo': ('prompt', 'chosen', 'rejected', 'source_chunk_id', 'source_region',
            'rejection_type', 'model_version'),
}
# Fields whose text is checked for encoding and orthography.
TEXT_FIELDS = {
    'sft': ('instruction', 'response'),
    'dpo': ('prompt', 'chosen', 'rejected'),
}

REPLACEMENT_CHAR = '�'
# Zero-width and bidi controls. These survive copy-paste and round-trips through
# spreadsheets and editors, and are invisible in every viewer.
# محارف غير مرئية تنجو من النسخ والتحويل ولا تظهر في أي عارض.
INVISIBLE = frozenset('​‌‍‎‏‪‫‬‭‮'
                      '⁦⁧⁨⁩﻿')


def _finding(line, code, severity, detail, field=None):
    return {'line': line, 'code': code, 'severity': severity, 'detail': detail,
            'field': field}


# ------------------------------------------------------------------- per-text checks

def check_encoding(text, line, field):
    out = []
    if ftfy.fix_text(text) != text:
        out.append(_finding(line, 'MOJIBAKE', ERROR,
                            'ftfy would change this text, so it carries encoding '
                            'damage', field))
    if REPLACEMENT_CHAR in text:
        out.append(_finding(line, 'REPLACEMENT_CHAR', ERROR,
                            'contains U+FFFD, i.e. bytes were already lost', field))
    if unicodedata.normalize('NFC', text) != text:
        out.append(_finding(line, 'NOT_NFC', WARN,
                            'not in NFC form; the pipeline normalises to NFC', field))
    ctrl = sorted({c for c in text
                   if unicodedata.category(c) == 'Cc' and c not in '\n\t'})
    if ctrl:
        out.append(_finding(line, 'CONTROL_CHARS', ERROR,
                            'control characters present: %s'
                            % ' '.join('U+%04X' % ord(c) for c in ctrl), field))
    inv = sorted({c for c in text if c in INVISIBLE})
    if inv:
        out.append(_finding(line, 'INVISIBLE_CHARS', WARN,
                            'invisible/bidi marks present: %s'
                            % ' '.join('U+%04X' % ord(c) for c in inv), field))
    return out


def orthography_rate(text):
    """Alef/ya variants per whitespace word. Zero means the text has been folded."""
    words = len(text.split())
    if not words:
        return None, 0
    n = sum(1 for c in text if c in ORTHOGRAPHY_VARIANTS)
    return n / float(words), words


def check_orthography(text, line, field):
    rate, words = orthography_rate(text)
    if rate is None or words < MIN_WORDS_FOR_ORTHOGRAPHY:
        return []          # too short to judge - see the docstring
    if rate == 0.0:
        return [_finding(line, 'FOLDED_ORTHOGRAPHY', ERROR,
                         'no alef/ya variants in %d words; every real chunk in this '
                         'project carries them (min 0.0609/word), so this text looks '
                         'alef/ya folded - the folded variant is a dedup matching key '
                         'and must never ship' % words, field)]
    return []


# ------------------------------------------------------------------ per-record checks

def check_record(rec, line, schema):
    out = []
    if not isinstance(rec, dict):
        return [_finding(line, 'NOT_AN_OBJECT', ERROR,
                         'line is valid JSON but not an object')]
    for f in SCHEMAS[schema]:
        if f not in rec:
            out.append(_finding(line, 'MISSING_FIELD', ERROR,
                                'required field is absent', f))
        elif isinstance(rec[f], str) and not rec[f].strip():
            out.append(_finding(line, 'EMPTY_FIELD', ERROR,
                                'required field is empty or whitespace only', f))
    for f in TEXT_FIELDS[schema]:
        v = rec.get(f)
        if isinstance(v, str) and v:
            out.extend(check_encoding(v, line, f))
            out.extend(check_orthography(v, line, f))
    return out


def validate_lines(lines, schema):
    """Validate an iterable of raw lines. Pure - no file IO, so it self-tests directly."""
    findings = []
    for i, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError as e:
            findings.append(_finding(i, 'BAD_JSON', ERROR, 'does not parse: %s' % e))
            continue
        findings.extend(check_record(rec, i, schema))
    return findings


def validate_file(path, schema):
    with open(path, encoding='utf-8') as fh:
        return validate_lines(fh.readlines(), schema)


# ------------------------------------------------------------------------- self-test

def run_self_test():
    ok = True
    GOOD = ('اشرح معنى هذه المادة المعجمية وبيّن أصل اللفظة وما ورد فيها من شواهد، '
            'مع ذكر النظائر المستعملة في المناطق الأخرى وما أورده المؤلف من آراء '
            'في هذا الباب، وأضف إيضاحا موجزا لكل معنى مما سبق ذكره في النص أعلاه.')

    def rec(**kw):
        d = {'instruction': 'اشرح معنى هذه المادة', 'response': GOOD,
             'source_chunk_id': 'c1', 'source_region': 'najdi',
             'format_type': 'dictionary_entry', 'model_version': 'v0'}
        d.update(kw)
        return json.dumps(d, ensure_ascii=False)

    # clean baseline must produce nothing
    if validate_lines([rec(), rec(), ''], 'sft'):
        print('  [FAIL] clean records produced findings:', validate_lines([rec()], 'sft'))
        ok = False

    cases = [
        (['{"instruction": broken'], 'BAD_JSON', 'malformed JSON'),
        (['[1,2,3]'], 'NOT_AN_OBJECT', 'valid JSON that is not an object'),
        ([json.dumps({'instruction': 'x'}, ensure_ascii=False)], 'MISSING_FIELD',
         'missing required fields'),
        ([rec(response='   ')], 'EMPTY_FIELD', 'whitespace-only required field'),
        ([rec(response=GOOD.encode('utf-8').decode('cp1252', 'replace'))], 'MOJIBAKE',
         'utf-8 text decoded as cp1252'),
        ([rec(response=GOOD + '�')], 'REPLACEMENT_CHAR', 'U+FFFD present'),
        ([rec(response=GOOD + '\x07')], 'CONTROL_CHARS', 'control character'),
        ([rec(response=GOOD + '​')], 'INVISIBLE_CHARS', 'zero-width space'),
        ([rec(response=_clean.normalize_arabic(GOOD))], 'FOLDED_ORTHOGRAPHY',
         'alef/ya folded text, folded with clean.py own function'),
    ]
    for lines, want, note in cases:
        codes = {f['code'] for f in validate_lines(lines, 'sft')}
        if want not in codes:
            print('  [FAIL] %s: expected %s, got %s' % (note, want, sorted(codes) or 'none'))
            ok = False

    # A SHORT folded text must NOT be flagged - too little evidence.
    short = 'اشرح معنى الكلمة'
    if any(f['code'] == 'FOLDED_ORTHOGRAPHY'
           for f in validate_lines([rec(response=_clean.normalize_arabic(short))], 'sft')):
        print('  [FAIL] a short text was flagged for orthography'); ok = False

    # The folded long text must differ from the original - i.e. the case is real.
    if _clean.normalize_arabic(GOOD) == GOOD:
        print('  [FAIL] the self-test baseline has no orthography variants to fold')
        ok = False

    # DPO schema is validated too, and its own fields are the ones checked.
    dpo = json.dumps({'prompt': 'س', 'chosen': GOOD, 'rejected': GOOD,
                      'source_chunk_id': 'c1', 'source_region': 'najdi',
                      'rejection_type': 'verbosity', 'model_version': 'v0'},
                     ensure_ascii=False)
    if validate_lines([dpo], 'dpo'):
        print('  [FAIL] a clean DPO record produced findings'); ok = False
    bad_dpo = json.dumps({'prompt': 'س', 'chosen': GOOD, 'source_chunk_id': 'c1',
                          'source_region': 'najdi', 'rejection_type': 'verbosity',
                          'model_version': 'v0'}, ensure_ascii=False)
    if 'MISSING_FIELD' not in {f['code'] for f in validate_lines([bad_dpo], 'dpo')}:
        print('  [FAIL] a DPO record missing `rejected` was not caught'); ok = False

    # Line numbers must point at the offending line, not the record index.
    f = validate_lines([rec(), '', 'not json'], 'sft')
    if not f or f[0]['line'] != 3:
        print('  [FAIL] line number wrong: %r' % f); ok = False

    print('passed: %s' % ok)
    return ok


# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description='Final formatting/encoding validator for a release file.')
    ap.add_argument('--in', dest='inp', help='released .jsonl to validate')
    ap.add_argument('--schema', choices=sorted(SCHEMAS), default='sft')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)
    if not args.inp:
        sys.stderr.write('[validate] --in <file.jsonl> is required\n')
        sys.exit(1)
    if not os.path.exists(args.inp):
        sys.stderr.write('[validate] not found: %s\n' % args.inp)
        sys.exit(1)

    findings = validate_file(args.inp, args.schema)
    errors = [f for f in findings if f['severity'] == ERROR]

    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
    else:
        print('[validate] %s  schema=%s' % (args.inp, args.schema))
        if not findings:
            print('  no findings.')
        for f in findings:
            print('  line %-4s %-8s %-20s %s%s'
                  % (f['line'], f['severity'], f['code'],
                     ('[%s] ' % f['field']) if f['field'] else '', f['detail'][:88]))
        print()
        print('  %d finding(s): %d ERROR, %d WARN'
              % (len(findings), len(errors), len(findings) - len(errors)))
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
