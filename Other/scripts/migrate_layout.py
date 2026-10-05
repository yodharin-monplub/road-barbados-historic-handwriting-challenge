"""Rewrite the active pipeline for the Data/ Model/ Other/ layout.

AGENTS.md now asks for exactly three top-level folders plus AGENTS.md itself.
Scripts used to sit one level below the project root; they now sit two levels
below (Model/modelN/), and the data, source and output trees moved. Rather than
hard-code a new depth, every script now walks up until it finds AGENTS.md, so
the next reorganisation cannot break them.

Run once, from anywhere:  python Other/scripts/migrate_layout.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent

ROOT_FINDER = '''ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent'''

# Scripts of the live pipeline. Legacy models 1-121 are historical and their
# saved predictions no longer exist, so they are left untouched on purpose.
ACTIVE = [f"model{n}" for n in range(122, 140)]

PATH_REWRITES = [
    (r'(["\'])data/processed/', r'\1Data/processed/'),
    (r'(["\'])data/raw/', r'\1Data/raw/'),
    (r'(["\'])outputs/', r'\1Other/outputs/'),
    (r'(["\'])checkpoints/', r'\1Other/checkpoints/'),
    (r'(["\'])config/', r'\1Other/config/'),
    (r'(["\'])submissions/', r'\1Other/submissions/'),
    (r'ROOT / "src"', r'ROOT / "Other" / "src"'),
    (r'ROOT / "model(\d+)"', r'ROOT / "Model" / "model\1"'),
    (r'ROOT / "outputs"', r'ROOT / "Other" / "outputs"'),
    (r'ROOT / "data/processed/train_folds.csv"', r'ROOT / "Data/processed/train_folds.csv"'),
]


def fix_python(text: str) -> str:
    text = re.sub(r'ROOT = Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', ROOT_FINDER, text)
    for pattern, replacement in PATH_REWRITES:
        text = re.sub(pattern, replacement, text)
    # never double-prefix
    text = text.replace("Other/Other/", "Other/").replace("Data/Data/", "Data/")
    return text


def fix_config(data):
    mapping = {
        "train_csv": ("data/processed/", "Data/processed/"),
        "test_csv": ("data/", "Data/"),
        "image_dir": ("data/raw/", "Data/raw/"),
        "exclude_ids": ("config/", "Other/config/"),
        "output_dir": ("outputs/", "Other/outputs/"),
        "base": ("checkpoints/", "Other/checkpoints/"),
        "sft_model": ("outputs/", "Other/outputs/"),
        "pseudo_csv": ("outputs/", "Other/outputs/"),
    }
    for key, (old, new) in mapping.items():
        value = data.get(key)
        if isinstance(value, str) and value.startswith(old):
            data[key] = new + value[len(old):]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    changed = []
    for name in ACTIVE:
        folder = ROOT / "Model" / name
        if not folder.is_dir():
            continue
        for path in folder.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            new = fix_python(text)
            if new != text:
                changed.append(str(path.relative_to(ROOT)))
                if args.apply:
                    path.write_text(new, encoding="utf-8")
        for path in folder.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            before = json.dumps(data, sort_keys=True)
            data = fix_config(data)
            if json.dumps(data, sort_keys=True) != before:
                changed.append(str(path.relative_to(ROOT)))
                if args.apply:
                    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"{'rewrote' if args.apply else 'would rewrite'} {len(changed)} files")
    for item in changed:
        print("  ", item)


if __name__ == "__main__":
    main()
