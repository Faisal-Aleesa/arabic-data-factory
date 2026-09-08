# -*- coding: utf-8 -*-
"""Dialect-marker detector - a SUPPORTING / DASHBOARD-ONLY signal.

Why this file exists
---------------------
llm_judge.py's system prompt already warns the model not to judge `wrong_register`
from the presence or absence of dialect markers alone (see llm_judge.py's system
prompt text). That warning implies a reviewer will want to *see* whether dialect
markers are present in `rejected`, as context, without that presence ever being
allowed to drive - or substitute for - the Judge's or check_dpo's verdict.

Nothing before this file computed such a signal anywhere in the delivered code.
This module adds exactly one small, pure, deterministic function that does:
detect a fixed, documented list of common Arabic dialect markers (Egyptian,
Levantine, Gulf, Maghrebi - not exhaustive, not a dialect classifier) as literal
substrings of `rejected`.

Hard boundary (do not weaken this)
-----------------------------------
- This function's output MUST NOT be written into judge_contract.JudgeOutput,
  MUST NOT be passed to judge_pipeline._maybe_escalate(), and MUST NOT be added to
  DETERMINISTIC_SIGNAL_KEYS in judge_pipeline.py. It has no verdict authority.
- Its only sanctioned consumer is a dashboard/report layer (see
  generate_dashboard_data.py), always rendered under a label that says
  "supporting signal - not a verdict signal".
- It is intentionally NOT imported by llm_judge.py, judge_pipeline.py, or
  judge_contract.py, so there is no code path by which it could leak into the
  Unified Verdict or the Judge's own assessment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# A fixed, documented, non-exhaustive list of common dialectal (non-Classical/MSA)
# tokens. Literal-substring matching only - the same "no fuzzy matching, no
# invented grounding" discipline judge_contract.py applies to evidence quotes.
DIALECT_MARKERS = (
    'هالجذر', 'هاللفظ', 'هالكلمة',           # demonstrative "hā-" prefix (Levantine/Gulf)
    'يعني', 'بس', 'مش', 'إيش', 'ليش', 'شو',   # Levantine
    'اللي', 'عشان', 'كده', 'دلوقتي', 'إزاي',  # Egyptian
    'اللي', 'وايد', 'زين', 'شلون', 'شنو',      # Gulf
    'ماكاين', 'بزاف', 'واش',                  # Maghrebi
)


@dataclass(frozen=True)
class DialectMarkerReport:
    rejected_text: str
    markers_found: List[str]

    @property
    def has_dialect_markers(self) -> bool:
        return len(self.markers_found) > 0

    def as_dict(self):
        return {
            'signal': 'dialect_markers_in_rejected',
            'signal_kind': 'supporting',            # NEVER 'verdict'
            'has_dialect_markers': self.has_dialect_markers,
            'markers_found': list(self.markers_found),
        }


# Arabic word tokenizer: splits on whitespace and punctuation, keeps Arabic letters
# (incl. tashkeel-stripped forms are NOT normalized here - matching is exact/literal
# by design, same discipline as judge_contract.py's evidence-quote containment).
_WORD_RE = re.compile(r'[\u0621-\u064A\u0660-\u0669]+')


def detect_dialect_markers(rejected_text: str) -> DialectMarkerReport:
    """Pure, deterministic, whole-word check. No model call, no fuzzy match.

    Matches on whole words (tokenized, not raw substrings) so a short marker like
    'مش' cannot false-positive inside an unrelated word such as 'المشهورة'. A word
    counts as a hit only if it equals a marker exactly. Returns every marker found,
    in first-match order, de-duplicated.
    """
    if not isinstance(rejected_text, str) or not rejected_text:
        return DialectMarkerReport(rejected_text=rejected_text or '', markers_found=[])
    words = _WORD_RE.findall(rejected_text)
    marker_set = set(DIALECT_MARKERS)
    found = []
    for w in words:
        if w in marker_set and w not in found:
            found.append(w)
    return DialectMarkerReport(rejected_text=rejected_text, markers_found=found)


# ------------------------------------------------------------------------- self-test

def run_self_test() -> bool:
    ok = True

    r1 = detect_dialect_markers('يعني هالجذر أتى، لما نقول أتى عليهم الدهر يعني خلصهم كلهم.')
    if not r1.has_dialect_markers or 'يعني' not in r1.markers_found:
        print('  [FAIL] known-dialect sentence should detect markers, got %s'
              % r1.markers_found)
        ok = False

    r2 = detect_dialect_markers(
        'أتى: أتى إليه إحسانا أي فعله، وأتى عليهم الدهر بمعنى أفناهم.')
    if r2.has_dialect_markers:
        print('  [FAIL] Classical/MSA sentence with no listed markers should not '
              'flag any: got %s' % r2.markers_found)
        ok = False

    r3 = detect_dialect_markers('')
    if r3.has_dialect_markers:
        print('  [FAIL] empty text must not flag markers'); ok = False

    # --- regression: short markers must not false-positive inside unrelated words
    # (e.g. 'مش' must not match inside 'المشهورة') - this is why matching is
    # whole-word, not raw substring
    r3b = detect_dialect_markers(
        'وهذا من المعاني المشهورة المتداولة المعروفة عند أهل اللغة قاطبة.')
    if r3b.has_dialect_markers:
        print('  [FAIL] whole-word matching regressed - false-matched inside a word: %s'
              % r3b.markers_found)
        ok = False

    r4 = detect_dialect_markers(None)  # type: ignore[arg-type]
    if r4.has_dialect_markers:
        print('  [FAIL] None input must not raise and must not flag markers'); ok = False

    d = r1.as_dict()
    if d.get('signal_kind') != 'supporting' or d.get('signal') != 'dialect_markers_in_rejected':
        print('  [FAIL] as_dict() must self-label as a supporting signal'); ok = False

    # --- boundary check: this module must not be imported by verdict-authoring code
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ('judge_contract.py', 'judge_pipeline.py', 'llm_judge.py'):
        path = os.path.join(here, fname)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                src = f.read()
            if 'dialect_markers' in src:
                print('  [FAIL] %s references dialect_markers - it must stay out of '
                      'verdict-authoring code' % fname)
                ok = False

    print('passed: %s' % ok)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Dialect-marker supporting signal.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        import sys
        sys.exit(0 if run_self_test() else 1)
    ap.print_help()
