#!/usr/bin/env bash
# Verify the 7B host is ready before spending GPU time on the gate.
cd /workspace/project
/workspace/venv/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("Qwen2_5_VL available:", hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"))
EOF
echo "pseudo3 files: $(ls Other/outputs/pseudo3/*.csv 2>/dev/null | wc -l)"
echo "images: $(ls Data/raw/images 2>/dev/null | wc -l)"
echo "weights: $(du -sh Other/checkpoints/model136 | cut -f1)"
