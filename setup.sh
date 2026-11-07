#!/bin/zsh
# Rebuild ComfyUI, custom nodes, and models exactly as this project uses them.
# Safe to re-run: skips anything already present. Needs git, uv, curl (~11GB of downloads).
set -e
cd "$(dirname "$0")"

# repo url, pinned commit, destination
clone() {
  [ -d "$3/.git" ] && { echo "✓ $3"; return; }
  git init -q "$3" && git -C "$3" remote add origin "$1"
  git -C "$3" fetch -q --depth 1 origin "$2" && git -C "$3" checkout -q FETCH_HEAD
  echo "✓ cloned $3"
}

clone https://github.com/comfyanonymous/ComfyUI.git 1332fabfcc507aa7c548cd5152071ed494c207e8 ComfyUI
clone https://github.com/ltdrdata/ComfyUI-Manager.git 4c50a2bc8b9e684d8a85930567957470b952bebe ComfyUI/custom_nodes/comfyui-manager
clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git 9257651221002dcba0a12f9cff37e1944e58fb60 ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 4d907bee61e92c2e65af3bd6383a4e4d356126d1 ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git a0f451a5113cf9becb0847b92884cb10cbdec0ef ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus

cd ComfyUI
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python torch torchvision torchaudio
uv pip install -q --python .venv/bin/python -r requirements.txt -r manager_requirements.txt \
  -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
echo "✓ python deps"

# destination, url
fetch() {
  [ -s "models/$1" ] && { echo "✓ $1"; return; }
  mkdir -p "models/$(dirname "$1")"
  echo "↓ $1" && curl -sSL --fail -o "models/$1.part" "$2" && mv "models/$1.part" "models/$1"
}

HF=https://huggingface.co
fetch checkpoints/juggernautXL_v9.safetensors $HF/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors
fetch loras/dmd2_sdxl_4step_lora_fp16.safetensors $HF/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora_fp16.safetensors
fetch checkpoints/dreamshaper_8.safetensors $HF/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors
fetch animatediff_models/AnimateLCM_sd15_t2v.ckpt $HF/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt
fetch loras/AnimateLCM_sd15_t2v_lora.safetensors $HF/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors
fetch ipadapter/ip-adapter-plus_sd15.safetensors $HF/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors
fetch clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors $HF/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors

echo "Done. Run ./start.sh"
