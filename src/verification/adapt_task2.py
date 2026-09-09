# -*- coding: utf-8 -*-
"""Task 3 - read-only adapter from Task 2's generated output to the locked schema.

Task 2's reconstruction pipeline emits records in its own shape. This module reads that
shape and yields records in the schema the verification stack expects. It NEVER writes to
`data/generated/**` and never modifies a Task 2 file - the same non-destructive pattern
the rest of this stack follows.

مُحوِّل للقراءة فقط: يقرأ صيغة المرحلة الثانية ويُخرج السجلات بالصيغة المتفق عليها.

    SFT locked schema: instruction, response, source_chunk_id, source_region,
                       format_type, model_version
    DPO locked schema: prompt, chosen, rejected, source_chunk_id, source_region,
                       rejection_type, model_version

SFT mapping - rename and flatten
---------------------------------
    question                                  -> instruction
    answer                                    -> response
    chunk_id                                  -> source_chunk_id
    source.region                             -> source_region
    source.format_type                        -> format_type
    generation.model + generation.prompt_version -> model_version

`model_version` joins two fields with '@' because neither alone identifies a run: the
model says nothing about which prompt produced the text, and the prompt version says
nothing about which model rendered it. Both change the output, so both belong in the
provenance string.

`chunk_id` and `source.chunk_id` are checked for agreement rather than assumed - measured
identical on all 4,645 records at intake, but a silent divergence would attach records to
the wrong source.

DPO mapping - collapse, then join
----------------------------------
`prompt`, `chosen` and `rejected` arrive as chat-message LISTS, not strings. Measured:
every one is a single-message list, prompt=user, chosen/rejected=assistant. They collapse
to that message's `content`.

`source_region` and `model_version` are ABSENT from every DPO record. They are recovered
by joining `source_sample_id` onto the SFT file's `sample_id` - measured: 3000/3000
resolve. A pair whose parent is missing is REPORTED, never silently defaulted.

THE SCAFFOLD ASYMMETRY - read this before trusting any DPO result
------------------------------------------------------------------
MEASURED at intake: `chosen` begins with a literal "Thinking:\\n...\\n\\nAnswer:\\n"
scaffold in 2,485 of 3,000 pairs. `rejected` carries it in ZERO of 3,000.

That is a formatting difference present in 83% of pairs and correlated perfectly with the
label. A preference model can score it without reading a word of Arabic - "prefer the one
that starts with Thinking:" - and every format, length and distinctness check in
`check_dpo.py` would be measuring the scaffold rather than the declared weakness.

So `--chosen-part` decides what a chosen response IS, and the default is deliberate:

    answer  (DEFAULT)  emit only the text after the "Answer:" marker. This makes chosen
                       and rejected structurally comparable and matches the SFT side,
                       where `response` is the answer with no scaffold.
    full               emit the raw content, scaffold included. Use ONLY to measure the
                       confound itself - never to produce numbers meant to describe
                       response quality.

This is a real transformation, not tidying, and it is reported in the summary so a run's
numbers can never be read without knowing which mode produced them. The underlying defect
belongs to the generator and should be fixed there; stripping it here keeps the confound
out of every downstream measurement in the meantime.
"""

from __future__ import unicode_literals

import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SFT_IN = os.path.join(REPO, 'data', 'generated', 'sft', 'accepted.jsonl')
DPO_IN = os.path.join(REPO, 'data', 'generated', 'dpo', 'candidates.jsonl')

SFT_FIELDS = ('instruction', 'response', 'source_chunk_id', 'source_region',
              'format_type', 'model_version')
DPO_FIELDS = ('prompt', 'chosen', 'rejected', 'source_chunk_id', 'source_region',
              'rejection_type', 'model_version')

CHOSEN_ANSWER = 'answer'
CHOSEN_FULL = 'full'

# The generator's scaffold. Anchored at the start so a mid-text mention is not stripped.
#
# TWO SHAPES, not one. The full form carries a Thinking section and then an Answer
# section; the short form is a bare "Answer:" header with no Thinking at all. The first
# version of this module matched only the full form, so bare-header records passed
# through with the header still attached.
#
# That was not a rare edge case. MEASURED on the 3,000 adapted pairs: 2,485 had the full
# scaffold and the remaining 515 (17.2%) had the bare header - the two partition the file
# exactly. So 515 `chosen` values shipped starting with "Answer:\n" while `rejected`
# carried it in ZERO records, which is exactly the asymmetric-scaffold confound this
# module exists to remove. A preference model could learn "prefer the text beginning with
# Answer:" and score well without reading any Arabic.
#
# It also suppressed a measurement of itself. The residual 7-character prefix made those
# 515 `chosen` values compare unequal to their byte-identical SFT `response` twins, so the
# cross-file overlap read 2,485 of 3,000 (82.8%) when the true figure is 3,000 of 3,000
# (100%). A stripping bug that also hides its own consequences is the worst kind, and it
# is why run_self_test() now pins BOTH shapes rather than only the one that was written.
_THINKING_RE = re.compile(r'^\s*Thinking\s*:\s*\n(.*?)\n\s*Answer\s*:\s*\n(.*)$', re.S)
_ANSWER_ONLY_RE = re.compile(r'^\s*Answer\s*:\s*\n?(.*)$', re.S)


def read_jsonl(path):
    with io.open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def split_scaffold(text):
    """(thinking, answer) if the scaffold is present, else (None, text).

    The full Thinking/Answer form is tried first; failing that, a bare "Answer:" header
    is stripped on its own and the thinking half comes back None, because there was none.
    Both patterns are anchored at the start, so a mid-text mention of either word is left
    alone - that distinction is pinned by run_self_test().
    """
    m = _THINKING_RE.match(text or '')
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _ANSWER_ONLY_RE.match(text or '')
    if m:
        return None, m.group(1).strip()
    return None, text


def collapse_messages(value):
    """A chat-message list -> one string. Non-list input passes through unchanged."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ''
    parts = []
    for m in value:
        if isinstance(m, dict):
            parts.append(m.get('content', '') or '')
        elif isinstance(m, str):
            parts.append(m)
    return '\n\n'.join(p for p in parts if p)


def _model_version(rec):
    g = rec.get('generation') or {}
    model = g.get('model') or 'unknown-model'
    pv = g.get('prompt_version') or 'unknown-prompt'
    return '%s@%s' % (model, pv)


def adapt_sft_record(rec):
    """One Task 2 SFT record -> (locked_record, list_of_problems)."""
    problems = []
    src = rec.get('source') or {}
    top_chunk = rec.get('chunk_id')
    nested_chunk = src.get('chunk_id')
    if top_chunk and nested_chunk and top_chunk != nested_chunk:
        problems.append('chunk_id %r != source.chunk_id %r' % (top_chunk, nested_chunk))
    out = {
        'instruction': rec.get('question') or '',
        'response': rec.get('answer') or '',
        'source_chunk_id': top_chunk or nested_chunk or '',
        'source_region': src.get('region') or '',
        'format_type': src.get('format_type') or '',
        'model_version': _model_version(rec),
    }
    for f in SFT_FIELDS:
        if not out[f]:
            problems.append('empty %s' % f)
    return out, problems


def adapt_dpo_record(rec, parents, chosen_part=CHOSEN_ANSWER):
    """One Task 2 DPO record -> (locked_record, problems). `parents` maps sample_id -> SFT."""
    problems = []
    parent = parents.get(rec.get('source_sample_id'))
    if parent is None:
        problems.append('source_sample_id %r does not resolve to an SFT record; '
                        'source_region and model_version cannot be recovered'
                        % rec.get('source_sample_id'))
        region, mv = '', ''
    else:
        region = (parent.get('source') or {}).get('region') or ''
        mv = _model_version(parent)

    chosen_raw = collapse_messages(rec.get('chosen'))
    rejected_raw = collapse_messages(rec.get('rejected'))
    thinking, answer = split_scaffold(chosen_raw)
    if chosen_part == CHOSEN_ANSWER:
        chosen = answer
    else:
        chosen = chosen_raw
    # rejected is measured never to carry the scaffold, but strip symmetrically anyway so
    # the two sides are always treated by the same rule rather than by an assumption.
    _, rejected_answer = split_scaffold(rejected_raw)
    rejected = rejected_answer if chosen_part == CHOSEN_ANSWER else rejected_raw

    out = {
        'prompt': collapse_messages(rec.get('prompt')),
        'chosen': chosen,
        'rejected': rejected,
        'source_chunk_id': rec.get('source_chunk_id') or '',
        'source_region': region,
        'rejection_type': rec.get('rejection_type') or '',
        'model_version': mv,
        # Optional enrichment beyond the locked minimum, recovered from the parent.
        # `format_type` is not in the locked DPO schema and check_dpo does not require
        # it, but check_dpo DOES consult it: without it the format axis reports
        # 'unknown_format'/'undecidable' on every pair and the wrong_formatting
        # corroboration silently cannot fire. Carrying it across costs nothing and
        # keeps a real check alive.
        'format_type': (parent.get('source') or {}).get('format_type') or ''
                       if parent else '',
    }
    for f in DPO_FIELDS:
        if not out[f]:
            problems.append('empty %s' % f)
    # "had a scaffold" must mean EITHER shape, not just the full Thinking/Answer one.
    # `thinking is not None` was the old test, and it under-reported by exactly the 515
    # bare-header records - the summary said 2,485 stripped when the true figure is
    # 3,000. A counter that misses the same cases the stripper missed cannot reveal the
    # bug, so it is derived from whether anything was actually removed.
    had_scaffold = (thinking is not None) or (answer != (chosen_raw or '').strip())
    return out, problems, had_scaffold


def load_parents(sft_path):
    return {r.get('sample_id'): r for _, r in read_jsonl(sft_path) if r.get('sample_id')}


def adapt_all(sft_path=SFT_IN, dpo_path=DPO_IN, chosen_part=CHOSEN_ANSWER):
    """Returns (sft_records, dpo_records, summary). Reads only; writes nothing."""
    summary = {'chosen_part': chosen_part, 'sft_problems': [], 'dpo_problems': [],
               'scaffold_in_chosen': 0, 'sft_in': 0, 'dpo_in': 0}
    sft_out = []
    parents = {}
    if os.path.exists(sft_path):
        for ln, rec in read_jsonl(sft_path):
            summary['sft_in'] += 1
            if rec.get('sample_id'):
                parents[rec['sample_id']] = rec
            out, probs = adapt_sft_record(rec)
            sft_out.append(out)
            for p in probs:
                summary['sft_problems'].append('line %d: %s' % (ln, p))
    dpo_out = []
    if os.path.exists(dpo_path):
        for ln, rec in read_jsonl(dpo_path):
            summary['dpo_in'] += 1
            out, probs, had = adapt_dpo_record(rec, parents, chosen_part)
            if had:
                summary['scaffold_in_chosen'] += 1
            dpo_out.append(out)
            for p in probs:
                summary['dpo_problems'].append('line %d: %s' % (ln, p))
    return sft_out, dpo_out, summary


def write_jsonl(path, records):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, 'w', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')


# ------------------------------------------------------------------------------ self-test

def run_self_test():
    """Offline. Uses invented records, so it needs neither Task 2's files nor a corpus."""
    ok = True

    sft_rec = {
        'sample_id': 'x_c0001_sft_001', 'chunk_id': 'x_c0001',
        'question': 'ما معنى الكلمة؟', 'answer': 'المعنى كذا.',
        'source': {'chunk_id': 'x_c0001', 'region': 'classical',
                   'format_type': 'dictionary_entry'},
        'generation': {'model': 'vendor/model-1', 'prompt_version': 'pv-9'},
    }
    out, probs = adapt_sft_record(sft_rec)
    if sorted(out) != sorted(SFT_FIELDS):
        print('  [FAIL] SFT output keys %s' % sorted(out)); ok = False
    if probs:
        print('  [FAIL] clean SFT record reported problems: %s' % probs); ok = False
    if out['model_version'] != 'vendor/model-1@pv-9':
        print('  [FAIL] model_version join: %r' % out['model_version']); ok = False
    if out['instruction'] != 'ما معنى الكلمة؟' or out['response'] != 'المعنى كذا.':
        print('  [FAIL] question/answer did not map to instruction/response'); ok = False

    # a chunk_id disagreement must be REPORTED, never silently preferred
    bad = dict(sft_rec, chunk_id='x_c0002')
    _, probs = adapt_sft_record(bad)
    if not any('!=' in p for p in probs):
        print('  [FAIL] chunk_id / source.chunk_id disagreement not reported'); ok = False

    # missing generation block must not crash and must be visible
    _, probs = adapt_sft_record({'question': 'q', 'answer': 'a', 'chunk_id': 'c',
                                 'source': {'region': 'classical',
                                            'format_type': 'dictionary_entry'}})
    if not any('unknown' in p or 'empty' in p for p in probs) and \
            adapt_sft_record({'question': 'q'})[0]['model_version'] != \
            'unknown-model@unknown-prompt':
        print('  [FAIL] missing generation block not surfaced'); ok = False

    # --- message collapsing --------------------------------------------------------
    if collapse_messages([{'role': 'user', 'content': 'hello'}]) != 'hello':
        print('  [FAIL] single-message collapse'); ok = False
    if collapse_messages([{'role': 'a', 'content': 'x'},
                          {'role': 'b', 'content': 'y'}]) != 'x\n\ny':
        print('  [FAIL] multi-message collapse'); ok = False
    if collapse_messages('already a string') != 'already a string':
        print('  [FAIL] string passthrough'); ok = False
    if collapse_messages(None) != '':
        print('  [FAIL] None collapse should be empty'); ok = False

    # --- scaffold splitting --------------------------------------------------------
    scaffolded = 'Thinking:\nراجعت المعنى.\n\nAnswer:\nالجواب هنا.'
    th, ans = split_scaffold(scaffolded)
    if th != 'راجعت المعنى.' or ans != 'الجواب هنا.':
        print('  [FAIL] scaffold split: %r / %r' % (th, ans)); ok = False
    th, ans = split_scaffold('نص عادي بلا سقالة.')
    if th is not None or ans != 'نص عادي بلا سقالة.':
        print('  [FAIL] plain text must pass through unsplit'); ok = False
    # a mid-text mention must NOT be stripped - the pattern is anchored
    mid = 'الجواب هو كذا. Thinking:\nليس سقالة\n\nAnswer:\nولا هذا'
    if split_scaffold(mid)[0] is not None:
        print('  [FAIL] mid-text "Thinking:" was treated as a scaffold'); ok = False

    # THE BARE "Answer:" HEADER, with no Thinking section. This is the shape the first
    # version of this module missed, and it was 515 of 3,000 real records (17.2%) - not
    # an edge case. It left an asymmetric scaffold token on `chosen` that `rejected`
    # never had, which is the exact confound this module removes.
    for variant in ('Answer:\nالجواب هنا.',
                    'Answer:  \nالجواب هنا.',
                    '  Answer:\n\nالجواب هنا.',
                    'Answer:الجواب هنا.'):
        th, ans = split_scaffold(variant)
        if th is not None or ans != 'الجواب هنا.':
            print('  [FAIL] bare Answer header %r -> %r / %r' % (variant, th, ans))
            ok = False
    # ... and the anchor still holds for the bare form: a mid-text "Answer:" stays put
    mid_answer = 'الجواب هو كذا. Answer:\nولا هذا'
    if split_scaffold(mid_answer)[1] != mid_answer:
        print('  [FAIL] mid-text "Answer:" was stripped'); ok = False
    # nothing that merely CONTAINS the word may trigger it
    if split_scaffold('Answers:\nكذا')[1] != 'Answers:\nكذا':
        print('  [FAIL] "Answers:" was treated as the scaffold header'); ok = False

    # --- DPO adaptation ------------------------------------------------------------
    parents = {'x_c0001_sft_001': sft_rec}
    dpo_rec = {
        'source_sample_id': 'x_c0001_sft_001', 'source_chunk_id': 'x_c0001',
        'prompt': [{'role': 'user', 'content': 'ما معنى الكلمة؟'}],
        'chosen': [{'role': 'assistant', 'content': scaffolded}],
        'rejected': [{'role': 'assistant', 'content': 'جواب خاطئ.'}],
        'rejection_type': 'partial_factual_errors',
    }
    out, probs, had = adapt_dpo_record(dpo_rec, parents)
    # Every locked field must be present. `format_type` rides along as a documented
    # enrichment - check_dpo consults it and reports 'undecidable' on every pair without
    # it - so the assertion is "locked fields are a subset", not "keys are exactly equal".
    if not set(DPO_FIELDS).issubset(out):
        print('  [FAIL] DPO output missing locked fields: %s'
              % sorted(set(DPO_FIELDS) - set(out))); ok = False
    if set(out) - set(DPO_FIELDS) != {'format_type'}:
        print('  [FAIL] unexpected extra DPO fields: %s'
              % sorted(set(out) - set(DPO_FIELDS) - {'format_type'})); ok = False
    if out.get('format_type') != 'dictionary_entry':
        print('  [FAIL] format_type not recovered from parent: %r'
              % out.get('format_type')); ok = False
    if probs:
        print('  [FAIL] clean DPO record reported problems: %s' % probs); ok = False
    if not had:
        print('  [FAIL] scaffold presence not detected'); ok = False
    if out['source_region'] != 'classical' or out['model_version'] != 'vendor/model-1@pv-9':
        print('  [FAIL] parent join did not recover region/model_version: %s' % out)
        ok = False
    # DEFAULT mode must remove the scaffold, so the two sides are comparable
    if out['chosen'] != 'الجواب هنا.':
        print('  [FAIL] default mode kept the scaffold: %r' % out['chosen']); ok = False
    if 'Thinking:' in out['chosen']:
        print('  [FAIL] scaffold leaked into chosen'); ok = False
    # full mode must keep it
    out_full, _, _ = adapt_dpo_record(dpo_rec, parents, chosen_part=CHOSEN_FULL)
    if not out_full['chosen'].startswith('Thinking:'):
        print('  [FAIL] full mode stripped the scaffold'); ok = False

    # an unresolvable parent is REPORTED, never silently defaulted
    orphan = dict(dpo_rec, source_sample_id='nope')
    out_o, probs_o, _ = adapt_dpo_record(orphan, parents)
    if not any('does not resolve' in p for p in probs_o):
        print('  [FAIL] orphan pair not reported'); ok = False
    if out_o['source_region'] != '':
        print('  [FAIL] orphan pair invented a region'); ok = False

    # --- the adapter must never modify its inputs -----------------------------------
    # Behavioural, not a source grep: write real input files, run the full adapter over
    # them, and assert the bytes and mtimes are unchanged. A grep for "open(...'w')"
    # cannot see an indirect write and fires on its own guard strings.
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        s_in = os.path.join(tmp, 'sft.jsonl')
        d_in = os.path.join(tmp, 'dpo.jsonl')
        with io.open(s_in, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(sft_rec, ensure_ascii=False) + '\n')
        with io.open(d_in, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(dpo_rec, ensure_ascii=False) + '\n')
        before = [(open(p, 'rb').read(), os.path.getmtime(p)) for p in (s_in, d_in)]
        s_rows, d_rows, summ = adapt_all(s_in, d_in)
        after = [(open(p, 'rb').read(), os.path.getmtime(p)) for p in (s_in, d_in)]
        if before != after:
            print('  [FAIL] adapt_all MODIFIED an input file'); ok = False
        if len(s_rows) != 1 or len(d_rows) != 1:
            print('  [FAIL] adapt_all row counts: %d/%d' % (len(s_rows), len(d_rows)))
            ok = False
        if summ['scaffold_in_chosen'] != 1:
            print('  [FAIL] adapt_all scaffold count: %s' % summ['scaffold_in_chosen'])
            ok = False
        # a missing input is not an error - it yields nothing
        s2, d2, _ = adapt_all(os.path.join(tmp, 'nope.jsonl'), d_in)
        if s2 or len(d2) != 1:
            print('  [FAIL] a missing SFT input should yield no SFT rows'); ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Read-only adapter for Task 2 output.")
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--sft-in', default=SFT_IN)
    ap.add_argument('--dpo-in', default=DPO_IN)
    ap.add_argument('--sft-out')
    ap.add_argument('--dpo-out')
    ap.add_argument('--chosen-part', choices=[CHOSEN_ANSWER, CHOSEN_FULL],
                    default=CHOSEN_ANSWER,
                    help='what a chosen response is; see the module docstring')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)

    sft, dpo, summary = adapt_all(a.sft_in, a.dpo_in, a.chosen_part)
    if a.sft_out:
        write_jsonl(a.sft_out, sft)
    if a.dpo_out:
        write_jsonl(a.dpo_out, dpo)
    summary['sft_out'] = len(sft)
    summary['dpo_out'] = len(dpo)
    summary['sft_problem_count'] = len(summary.pop('sft_problems'))
    summary['dpo_problem_count'] = len(summary.pop('dpo_problems'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
