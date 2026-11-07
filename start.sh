#!/bin/zsh
# Start ComfyUI at http://127.0.0.1:8188
cd "$(dirname "$0")/ComfyUI" && exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 "$@"
