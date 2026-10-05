"""Point the rented-host shell scripts at the Data/ Model/ Other/ layout.

The GPU hosts unpack the same tree, so their scripts must use the same paths.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent

REWRITES = [
    (r'(?<![\w/])data/processed/', 'Data/processed/'),
    (r'(?<![\w/])data/raw', 'Data/raw'),
    (r'(?<![\w/])outputs/', 'Other/outputs/'),
    (r'(?<![\w/])checkpoints/', 'Other/checkpoints/'),
    (r'(?<![\w/])config/', 'Other/config/'),
    (r'(?<![\w/])(model\d+)/', r'Model/\1/'),
    (r'\$PWD/Other/checkpoints', '$PWD/Other/checkpoints'),
]


def main() -> None:
    changed = []
    for path in sorted((ROOT / "Model").glob("model*/**/*.sh")):
        text = original = path.read_text(encoding="utf-8")
        for pattern, replacement in REWRITES:
            text = re.sub(pattern, replacement, text)
        text = (text.replace("Other/Other/", "Other/")
                    .replace("Model/Model/", "Model/")
                    .replace("Data/Data/", "Data/"))
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(str(path.relative_to(ROOT)))
    print(f"rewrote {len(changed)} shell scripts")
    for item in changed:
        print("  ", item)


if __name__ == "__main__":
    main()
