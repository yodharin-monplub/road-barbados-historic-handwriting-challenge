"""Final path touch-ups in Other/REPRODUCE.md after the layout change."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent

path = ROOT / "Other" / "REPRODUCE.md"
text = path.read_text(encoding="utf-8")

replacements = [
    ("       Other/outputs/ Other/checkpoints/          junctions",
     "       outputs/ checkpoints/          junctions"),
    ("       Other/submissions/ .env email.txt    submission",
     "       submissions/ .env email.txt    submission"),
    ("uv venv .venv --python 3.11", "uv venv Other/.venv --python 3.11"),
    ("uv venv .venv-kraken --python 3.12", "uv venv Other/.venv-kraken --python 3.12"),
    ("--python .venv-kraken/bin/python", "--python Other/.venv-kraken/bin/python"),
    ("registry .venv-kraken/bin/kraken", "registry Other/.venv-kraken/bin/kraken"),
    ("discover -s tests", "discover -s Other/tests"),
    ("`config/suspect_row_audit.txt`", "`Other/config/suspect_row_audit.txt`"),
    ("see model137/readme.txt", "see Model/model137/readme.txt"),
    ("(model126)", "(Model/model126)"),
    ("(model127)", "(Model/model127)"),
]
for old, new in replacements:
    text = text.replace(old, new)

for number in (122, 123, 125, 129, 134, 136, 138):
    text = text.replace(f"--config model{number}/", f"--config Model/model{number}/")

path.write_text(text, encoding="utf-8")
print("REPRODUCE.md finalised")
