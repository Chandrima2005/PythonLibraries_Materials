"""Build the chapter notebooks from src/*.md and execute them.

Usage (from the repository root):
    python tools/build_notebooks.py            # build and execute every chapter
    python tools/build_notebooks.py 03 07      # only chapters whose file starts with 03 or 07
    python tools/build_notebooks.py --no-exec  # build without running the code

Rules for a source file:
- each fenced ```python block becomes a code cell;
- the text between code blocks becomes markdown, split again at every "## " heading.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def split_markdown(text: str) -> list[str]:
    parts, current = [], []
    for line in text.splitlines():
        if line.startswith("## ") and any(l.strip() for l in current):
            parts.append("\n".join(current))
            current = []
        current.append(line)
    if any(l.strip() for l in current):
        parts.append("\n".join(current))
    return [p.strip("\n") for p in parts if p.strip()]


def to_notebook(source: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    pos = 0
    for m in FENCE.finditer(source):
        for md in split_markdown(source[pos : m.start()]):
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        nb.cells.append(nbformat.v4.new_code_cell(m.group(1).rstrip("\n")))
        pos = m.end()
    for md in split_markdown(source[pos:]):
        nb.cells.append(nbformat.v4.new_markdown_cell(md))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main(argv: list[str]) -> int:
    execute = "--no-exec" not in argv
    wanted = [a for a in argv if not a.startswith("--")]
    files = sorted(SRC.glob("*.md"))
    if wanted:
        files = [f for f in files if any(f.name.startswith(w) for w in wanted)]
    failed = []
    for f in files:
        nb = to_notebook(f.read_text(encoding="utf-8"))
        out = ROOT / f"{f.stem}.ipynb"
        if execute:
            try:
                NotebookClient(nb, timeout=180, kernel_name="python3",
                               resources={"metadata": {"path": str(ROOT)}}).execute()
            except Exception as e:  # keep going, report at the end
                failed.append(f.name)
                print(f"FAIL {f.name}: {str(e).strip().splitlines()[-1][:300]}")
        nbformat.write(nb, out)
        print(f"wrote {out.name} ({len(nb.cells)} cells)")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
