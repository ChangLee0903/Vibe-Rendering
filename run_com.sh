#!/usr/bin/env bash
set -euo pipefail

SCRIPT="ours.py"
mkdir -p outputs

# teapot.obj
VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_con.png" \
VIBE_PROMPT="Glossy cyan teapot on solid yellow background, strong left light, sharp highlights." \
python "$SCRIPT"

VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_abs.png" \
VIBE_PROMPT="A vibrant, cinematic teapot scene with bold contrast, clean minimal backdrop, energetic mood." \
python "$SCRIPT"