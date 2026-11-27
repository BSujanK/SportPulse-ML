#!/usr/bin/env python3
"""Mechanical README checks: the rules a script can enforce, so the writer can spend judgement elsewhere.

    python check_readme.py README.md            # exit 1 on errors, 0 otherwise
    python check_readme.py README.md --strict   # warnings also fail
    python check_readme.py docs/*.md --page     # docs pages: links, placeholders and fences only

Errors: placeholders, broken relative links or images, unbalanced code fences, no H1 near the top,
no runnable code block in the first screen. Warnings: hand-typed version badges (they go stale),
length past the landing-page budget, skipped heading levels, images without alt text.
Standard library only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FIRST_SCREEN = 60          # visible lines a visitor sees before deciding (blank lines and bare HTML tags excluded)
BARE_TAG = re.compile(r"^\s*</?(p|div|br|hr|picture|center)\b[^>]*>\s*$", re.I)
LANDING_BUDGET = 300       # beyond this, move reference material to docs/
PLACEHOLDER = re.compile(
    r"<\s*(your|insert|repo|project|username|org|url|path|token|api[-_ ]?key)[^>]*>"
    r"|\bYOUR_[A-Z_]+\b|\b(TODO|TBD|FIXME|lorem ipsum)\b|github\.com/(user|username|your-?\w*)/", re.I)
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_SRC = re.compile(r"""<(?:img|source|a)\b[^>]*?(?:src|srcset|href)=["']([^"']+)["']""", re.I)
STATIC_VERSION_BADGE = re.compile(r"img\.shields\.io/badge/(version|release|v)[-_]", re.I)


def check(path: Path, page: bool = False) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # YAML front matter (Hugging Face Spaces, Jekyll) is metadata, not part of the first screen
    offset = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        offset = (end + 1) if end else 0
    errors, warnings = [], []
    in_code = False
    fences = 0
    first_code_line = None
    h1_line = None
    last_level = 0
    visible = 0
    for n, line in enumerate(lines, 1):
        if n > offset and line.strip() and not BARE_TAG.match(line):
            visible += 1
        # placeholders are checked everywhere, code blocks included: a placeholder in a command breaks copy-paste
        if PLACEHOLDER.search(line):
            errors.append(f"line {n}: placeholder left in: {line.strip()[:90]}")
        if line.lstrip().startswith("```"):
            fences += 1
            in_code = not in_code
            if in_code and first_code_line is None:
                first_code_line = visible
            continue
        if in_code:
            continue
        m = re.match(r"^(#{1,6})\s", line)
        if m or re.match(r"^\s*<h1\b", line, re.I):
            level = len(m.group(1)) if m else 1
            if level == 1 and h1_line is None:
                h1_line = visible
            if last_level and level > last_level + 1:
                warnings.append(f"line {n}: heading jumps from h{last_level} to h{level}")
            last_level = level
        for alt, target in LINK.findall(line):
            _check_target(path, target, n, errors)
            if line[line.find(f"[{alt}]") - 1:line.find(f"[{alt}]")] == "!" and not alt.strip():
                warnings.append(f"line {n}: image without alt text")
        for target in HTML_SRC.findall(line):
            _check_target(path, target, n, errors)
        if STATIC_VERSION_BADGE.search(line):
            warnings.append(f"line {n}: hand-typed version badge; use a live one (PyPI, npm, GitHub release)")
    if fences % 2:
        errors.append("unbalanced ``` code fences")
    if page:
        return errors, warnings
    if h1_line is None or h1_line > 25:
        errors.append("no H1 title in the first 25 lines")
    if first_code_line is None or first_code_line > FIRST_SCREEN:
        errors.append(f"no code block in the first {FIRST_SCREEN} lines: the quick start must be in the first screen")
    if len(lines) - offset > LANDING_BUDGET:
        warnings.append(f"{len(lines) - offset} lines: past the {LANDING_BUDGET}-line landing-page budget; move reference "
                        "material (API tables, experiment logs, architecture detail) to docs/ and link it")
    return errors, warnings


def _check_target(readme: Path, target: str, n: int, errors: list[str]) -> None:
    if re.match(r"^(https?:|mailto:|#|data:)", target):
        return
    rel = target.split("#")[0].split("?")[0]
    if rel and not (readme.parent / rel).exists():
        errors.append(f"line {n}: relative link or image not found: {target}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    strict, page = "--strict" in argv, "--page" in argv
    failed = False
    for name in [a for a in argv if not a.startswith("--")]:
        errors, warnings = check(Path(name), page)
        for e in errors:
            print(f"ERROR   {name}: {e}")
        for w in warnings:
            print(f"WARNING {name}: {w}")
        print(f"{name}: {len(errors)} error(s), {len(warnings)} warning(s)")
        failed |= bool(errors or (strict and warnings))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
