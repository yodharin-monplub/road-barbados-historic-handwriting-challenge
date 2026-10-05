#!/usr/bin/env bash
# Model154 evaluation: pruned twelve (deployed) vs thirteen (+TrOCR), bootstrap, an
# honest member-selection re-run, and the thirteen-member submission.
# Run on the Model154 host after all five folds are DONE; needs /workspace/p12.tgz.
cd /workspace/project
tar -xzf /workspace/p12.tgz 2>/dev/null
PY=/workspace/venv/bin/python
P12="model134_pseudo3 model140_pseudo4 model122_prod_lora model130_kraken model136_qwen25vl7b model143_qwen3vl4b_vision model144_qwen25vl7b_vision model145_qwen3vl4b_vision_pseudo4 model146_qwen3vl8b_vision model147_qwen3vl8b_pseudo5 model149_kraken_pseudo5 model150_kraken_pseudo4"
N=model154_trocr
for f in 0 1 2 3 4; do printf "m154 fold$f: "; ls Other/outputs/$N/fold$f/ | tr '\n' ' '; echo; done
oof() { printf "%-40s" "$1"; $PY Model/model124/ensemble.py oof --members $3 --min-votes $2 | grep -E '"ensemble"'; cp Other/outputs/model124/oof_ensemble.csv $4; }
printf "%-40s" "model154 alone"; $PY Model/model124/ensemble.py oof --members $N --min-votes 2 | grep -E '"ensemble"'
oof "12 pruned (deployed, public .919377)" 2 "$P12"     /tmp/w12.csv
oof "13 (+model154)"                       2 "$P12 $N"  /tmp/w13.csv
oof "13, min_votes 3"                      3 "$P12 $N"  /tmp/w13m3.csv
for c in w13 w13m3; do echo "== bootstrap $c vs 12"; $PY Model/model124/bootstrap.py --base /tmp/w12.csv --candidate /tmp/$c.csv | tail -n 2; done
echo "== member selection with TrOCR available"
$PY -u Model/model154/member_select154.py
mkdir -p Other/submissions
$PY Model/model124/ensemble.py test --members $P12 $N --min-votes 2 --fold-min-votes 3 >/dev/null
$PY Model/model124/build_submission.py --anchor Other/outputs/model124/test_ensemble.csv \
    --output Other/submissions/model154_ensemble13_candidate.csv | grep -E 'sha256|rows|blank'
$PY - <<'EOF'
import csv
def load(p):
    with open(p, encoding="utf-8", newline="") as h:
        return {r[0]: r[1] for r in csv.reader(h)}
a = load("Other/submissions/model152_ensemble12_pruned.csv")
b = load("Other/submissions/model154_ensemble13_candidate.csv")
print("test lines changed vs the deployed twelve:", sum(a[k] != b.get(k) for k in a), "of", len(a))
EOF
echo FINAL154_DONE
