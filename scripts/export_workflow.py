#!/usr/bin/env python3
"""Pull the ComfyUI graph that made an image out of its PNG metadata and save it as JSON.

ComfyUI's SaveImage embeds the exact (API-format) graph in every PNG. The saved JSON can be
dropped onto the ComfyUI canvas, opened via Workflow > Open, or POSTed to /prompt as-is.

Usage:
  python scripts/export_workflow.py outputs/<image>.png workflows/<name>.json
"""
import json
import sys
from pathlib import Path

from PIL import Image

src, dst = sys.argv[1], Path(sys.argv[2])
graph = json.loads(Image.open(src).info["prompt"])
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(graph, indent=2) + "\n")
print(f"{dst}  ({len(graph)} nodes: {', '.join(sorted({n['class_type'] for n in graph.values()}))})")
