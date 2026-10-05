#!/usr/bin/env bash
# Unpack uploaded code/images into /workspace/project (safe to re-run for code updates).
set -euo pipefail
mkdir -p /workspace/project/Data/raw
cd /workspace/project
tar -xzf /workspace/code.tgz
# images_small.tar holds the pre-resized crops (height<=176, longest edge<=2800,
# JPEG q92) that scripts/shrink_images.py produces; every Model122 config
# downscales below that cap anyway, and the slow home uplink cannot send 442MB.
if [ ! -d Data/raw/images ]; then
  if [ -f /workspace/images_small.tar ]; then
    tar -xf /workspace/images_small.tar -C Data/raw
    mv Data/raw/images_small Data/raw/images
  elif [ -f /workspace/images.tar ]; then
    tar -xf /workspace/images.tar -C Data/raw
  fi
fi
ls Data/raw/images | wc -l
echo UNPACK_OK

