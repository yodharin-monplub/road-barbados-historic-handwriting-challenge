# Pretrained model eligibility ledger

Checked on 2026-09-03. Recheck the exact pinned revisions before final code
submission and preserve their licence files in the reproducibility archive.

| Candidate | Pinned revision | Licence/status |
|---|---|---|
| Qwen/Qwen2-VL-2B-Instruct | 895c3a49bc3fa70a340399125c650a463535e71c | Apache-2.0; organizer VLM starter base used by Model27 |
| lightonai/LightOnOCR-2-1B | c97bd377f04481830395218fa8951df9deaba756 | Apache-2.0 |
| lightonai/LightOnOCR-2-1B-base | c1bc8eb6625be7b34178ff53481b242f7b8ffcb8 | Apache-2.0 |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec70b05090aa6165b016eac8ec08e71b17 | Apache-2.0 |
| Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 | Apache-2.0; public ungated local text corrector for Model93 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c | Apache-2.0; public ungated direct-image challenger for Model95 |
| Qwen/Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 | Apache-2.0 |
| microsoft/trocr-large-handwritten | e68501f437cd2587ae5d68ee457964cac824ddee | MIT: no licence tag on the HF card, but microsoft/unilm trocr/README.md states the project is under the repository root LICENSE, which is MIT (checked 2026-09-24). Used by Model154 |
| PP-OCRv6 medium historical HTR | Zenodo 21788410, SHA256 `15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9` | Apache-2.0; explicitly approved by Zindi on 2026-08-28 |

Do not use `stanford-oval/churro-3B` or `Qwen/Qwen2.5-VL-3B-Instruct`.
Their Qwen Research licences restrict commercial use. Zindi explicitly ruled
CHURRO-3B ineligible because a winning solution may be commercially deployed.
