# -*- coding: utf-8 -*-
"""Static symbol check - catches edits that silently did not apply.

فحص ساكن للرموز: يكشف التعديلات التي لم تُطبَّق فعليا دون أن تُصدر خطأ.

Why this exists
---------------
Anchored string-replace edits fail SILENTLY when the anchor does not match - a stray
CRLF, a changed blank line, a reflowed comment - and the file is simply left alone. That
happened twice in one session here. Both times the self-test still reported success,
because a self-test cannot exercise a function that was never written.

So "the self-test passed" is not evidence that an edit landed. This checks the thing the
self-test structurally cannot: that the symbols actually exist.

Two modes
---------
  --assert FILE SYM [SYM...]
      Assert each symbol is defined at module level in FILE. Use immediately after an
      anchored edit, before believing it worked.

  --cross-module DIR
      Parse every .py in DIR, resolve `import x as y` aliases, and verify every `y.name`
      reference points at something `x` actually defines. This is the automatic form: it
      would have caught `vs.is_grounded` being referenced before verify_sft defined it,
      which in this session surfaced only as an AttributeError at run time.

AST-based on purpose - no imports, no side effects, no model downloads, and it works even
when a module's third-party dependencies are missing.

يعتمد على تحليل الشجرة النحوية لا على الاستيراد: بلا آثار جانبية، ويعمل حتى لو غابت
الاعتماديات الخارجية.
"""

import argparse
import ast
import io
import os
import sys

# Attribute names that legitimately exist on any module object.
DUNDER_OK = {'__file__', '__name__', '__doc__', '__path__', '__dict__', '__all__'}


def _read(path):
    return io.open(path, encoding='utf-8', newline='').read()


def line_endings(path):
    raw = _read(path)
    has_crlf, has_lf = '\r\n' in raw, '\n' in raw.replace('\r\n', '')
    if has_crlf and has_lf:
        return 'MIXED'
    return 'CRLF' if has_crlf else 'LF'


def module_symbols(path):
    """Top-level names a module defines or imports."""
    tree = ast.parse(_read(path), filename=path)
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
                elif isinstance(t, ast.Tuple):
                    out.update(e.id for e in t.elts if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
    return out


def local_import_aliases(path):
    """`import x as y` -> {y: x} for modules that live beside this file."""
    tree = ast.parse(_read(path), filename=path)
    here = os.path.dirname(os.path.abspath(path))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                base = a.name.split('.')[0]
                if os.path.exists(os.path.join(here, base + '.py')):
                    out[a.asname or base] = os.path.join(here, base + '.py')
    return out


def attribute_uses(path, alias):
    """Every `alias.name` referenced in the file, with line numbers."""
    tree = ast.parse(_read(path), filename=path)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == alias):
            hits.append((node.attr, node.lineno))
    return hits


def check_assert(path, symbols):
    have = module_symbols(path)
    missing = [s for s in symbols if s not in have]
    eol = line_endings(path)
    print('%s  [%s]' % (path, eol))
    for s in symbols:
        print('  %-40s %s' % (s, 'OK' if s in have else 'MISSING'))
    if eol == 'MIXED':
        print('  WARNING: mixed line endings - the usual cause of a silently '
              'non-matching edit anchor')
    if missing:
        print('\nFAIL: %d symbol(s) missing. The edit did not apply.' % len(missing))
        return False
    print('\nOK: all %d symbol(s) present.' % len(symbols))
    return True


def check_cross_module(directory):
    files = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                   if f.endswith('.py'))
    cache, problems, checked = {}, [], 0
    for f in files:
        try:
            aliases = local_import_aliases(f)
        except SyntaxError as e:
            problems.append((f, 0, '<syntax error>', str(e)))
            continue
        for alias, target in aliases.items():
            if target not in cache:
                try:
                    cache[target] = module_symbols(target)
                except SyntaxError as e:
                    problems.append((target, 0, '<syntax error>', str(e)))
                    cache[target] = set()
            have = cache[target]
            for attr, lineno in attribute_uses(f, alias):
                checked += 1
                if attr not in have and attr not in DUNDER_OK:
                    problems.append((f, lineno, '%s.%s' % (alias, attr),
                                     'not defined in %s' % os.path.basename(target)))
    print('cross-module symbol check: %d file(s), %d reference(s)'
          % (len(files), checked))
    for f in files:
        eol = line_endings(f)
        if eol == 'MIXED':
            print('  WARNING mixed line endings: %s' % f)
    if problems:
        print('\nFAIL:')
        for f, ln, ref, why in problems:
            print('  %s:%s  %s  %s' % (os.path.basename(f), ln or '?', ref, why))
        return False
    print('all cross-module references resolve.')
    return True


def main():
    ap = argparse.ArgumentParser(description='Static symbol check.')
    ap.add_argument('--assert', dest='assert_file', metavar='FILE')
    ap.add_argument('symbols', nargs='*')
    ap.add_argument('--cross-module', metavar='DIR')
    args = ap.parse_args()

    ok = True
    if args.assert_file:
        if not args.symbols:
            sys.stderr.write('--assert FILE needs at least one symbol name\n')
            sys.exit(2)
        ok = check_assert(args.assert_file, args.symbols) and ok
    if args.cross_module:
        ok = check_cross_module(args.cross_module) and ok
    if not args.assert_file and not args.cross_module:
        ap.print_help()
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
