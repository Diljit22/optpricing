"""
Generate a clean text tree of the repository.

Rules
-----
Skip any directory whose name starts with a dot (e.g. .git, .venv).
Skip any directory named '__pycache__'.
Skip the contents of directories whose names contain a dot
"""

from collections.abc import Iterator
from pathlib import Path


def walk(path: Path, prefix: str = "") -> Iterator[str]:
    entries = sorted(
        [
            p
            for p in path.iterdir()
            if not p.name.startswith(".") and p.name != "__pycache__"
        ],
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    last = len(entries) - 1

    for idx, entry in enumerate(entries):
        connector = "└── " if idx == last else "├── "
        yield f"{prefix}{connector}{entry.name}"

        if entry.is_dir() and "." not in entry.name:
            ext = "    " if idx == last else "│   "
            yield from walk(entry, prefix + ext)


if __name__ == "__main__":
    root = Path(".").resolve()
    print(root.name)
    for line in walk(root):
        print(line)
