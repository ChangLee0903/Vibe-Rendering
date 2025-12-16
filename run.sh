#!/usr/bin/env bash
set -euo pipefail

SCRIPT="ours.py"
mkdir -p outputs

# dolphin.obj
VIBE_OBJ_PATH="inputs/dolphin.obj" \
VIBE_OUT_IMAGE="outputs/dolphin_ours.png" \
VIBE_PROMPT="Glossy teal dolphin, white studio background." \
python "$SCRIPT"

# teapot.obj
VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_ours.png" \
VIBE_PROMPT="Matte black teapot, warm soft light, no fog." \
python "$SCRIPT"

# Tree.obj
VIBE_OBJ_PATH="inputs/Tree.obj" \
VIBE_OUT_IMAGE="outputs/tree_ours.png" \
VIBE_PROMPT="Misty tree silhouette, cool blue fog, vignette." \
python "$SCRIPT"
