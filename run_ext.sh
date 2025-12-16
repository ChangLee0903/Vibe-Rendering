#!/usr/bin/env bash
set -euo pipefail

SCRIPT="ours.py"
mkdir -p outputs

# teapot.obj
VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_ext1.png" \
VIBE_PROMPT="Teapot made of glowing emerald glass, razor-sharp chrome highlights, floating in a pure magenta studio void." \
python "$SCRIPT"

VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_ext2.png" \
VIBE_PROMPT="Teapot made of glowing emerald glass, razor-sharp chrome highlights, floating in a pure magenta studio void. Hard white key light from upper-left, subtle rim light.
" \
python "$SCRIPT"

VIBE_OBJ_PATH="inputs/teapot.obj" \
VIBE_OUT_IMAGE="outputs/teapot_ext3.png" \
VIBE_PROMPT="Teapot made of glowing emerald glass, razor-sharp chrome highlights, floating in a pure magenta studio void. Hard white key light from upper-left, subtle rim light.
High contrast, slight vignette, no fog." \
python "$SCRIPT"
