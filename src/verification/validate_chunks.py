# -*- coding: utf-8 -*-
"""Validate data-engineering chunks before paid generation calls.

This check is deliberately local and conservative. It validates the hand-off from
``src/data_engineering`` to the post-training generation stage without deciding
whether a chunk is useful for a particular task.
"""

import argparse
import json
import os
import sys


REQUIRED_FIELDS = (
    'chunk_id',
    'chunk_text',
    'format_type',
    'source',
    'license',
)
KNOWN_FORMATS = frozenset(('dictionary_entry', 'prose', 'narrative_paragraph',
                           'verse', 'list', 'footnote_block'))
FATAL_REVIEW_FLAGS = frozenset(('damaged_page', 'unreadable', 'residue_tokens'))


def finding(line, code, detail, field=None):
    return {'line': line, 'code': code, 'detail': detail, 'field': field}


def validate_record(record, line, fatal_review_flags=FATAL_REVIEW_FLAGS):
    """Return findings for one chunk record; never raises on bad input."""
    if not isinstance(record, dict):
        return [finding(line, 'NOT_AN_OBJECT', 'line must contain a JSON object')]

    findings = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if field not in record:
            findings.append(finding(line, 'MISSING_FIELD',
                                    'required field is absent', field))
        elif not isinstance(value, str) or not value.strip():
            findings.append(finding(line, 'EMPTY_FIELD',
                                    'required field must be a non-empty string', field))

    chunk_id = record.get('chunk_id')
    if isinstance(chunk_id, str) and chunk_id and any(c.isspace() for c in chunk_id):
        findings.append(finding(line, 'INVALID_CHUNK_ID',
                                'chunk_id must not contain whitespace', 'chunk_id'))

    format_type = record.get('format_type')
    if isinstance(format_type, str) and format_type not in KNOWN_FORMATS:
        findings.append(finding(line, 'UNKNOWN_FORMAT',
                                'unsupported format_type: %s' % format_type,
                                'format_type'))

    text = record.get('chunk_text')
    if isinstance(text, str) and not text.strip():
        findings.append(finding(line, 'EMPTY_TEXT',
                                'chunk_text contains no usable text', 'chunk_text'))

    all_flags = []
    for field in ('review_flags', 'source_doc_flags'):
        flags = record.get(field, [])
        if flags is None:
            flags = []
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            findings.append(finding(line, 'INVALID_REVIEW_FLAGS',
                                    '%s must be a list of strings' % field, field))
        else:
            all_flags.extend(flags)
    fatal = sorted(set(all_flags) & set(fatal_review_flags))
    if fatal:
        findings.append(finding(line, 'FATAL_REVIEW_FLAG',
                                'chunk carries blocking review flags: %s' % ', '.join(fatal),
                                'review_flags'))

    return findings


def validate_lines(lines, fatal_review_flags=FATAL_REVIEW_FLAGS):
    """Validate JSONL lines and return a deterministic list of findings."""
    findings = []
    for line, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError as exc:
            findings.append(finding(line, 'BAD_JSON', 'does not parse: %s' % exc))
            continue
        findings.extend(validate_record(record, line, fatal_review_flags))
    return findings


def validate_file(path, fatal_review_flags=FATAL_REVIEW_FLAGS):
    with open(path, encoding='utf-8') as handle:
        return validate_lines(handle, fatal_review_flags)


def run_self_test():
    good = {
        'chunk_id': 'dictionary_c0001',
        'chunk_text': 'نص عربي صالح',
        'format_type': 'dictionary_entry',
        'source': 'book',
        'license': 'CC BY-SA 4.0',
        'review_flags': [],
    }
    if validate_record(good, 1):
        raise AssertionError('valid chunk produced findings')
    bad = dict(good, format_type='unknown', source_doc_flags=['damaged_page'])
    codes = {item['code'] for item in validate_record(bad, 2)}
    expected = {'UNKNOWN_FORMAT', 'FATAL_REVIEW_FLAG'}
    if not expected <= codes:
        raise AssertionError('expected %s, got %s' % (sorted(expected), sorted(codes)))
    lines = ['{bad json', json.dumps(good, ensure_ascii=False)]
    if [item['code'] for item in validate_lines(lines)] != ['BAD_JSON']:
        raise AssertionError('JSONL validation did not preserve line-level findings')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', help='chunk JSONL file to validate')
    parser.add_argument('--self-test', action='store_true', help='run local checks')
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        print('[validate_chunks] self-test passed')
        return 0
    if not args.path:
        parser.error('a JSONL path is required unless --self-test is used')

    findings = validate_file(args.path)
    for item in findings:
        location = '%s:%s' % (args.path, item['line'])
        field = ' [%s]' % item['field'] if item['field'] else ''
        print('%s %s%s: %s' % (location, item['code'], field, item['detail']))
    print('[validate_chunks] %d finding(s)' % len(findings))
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())