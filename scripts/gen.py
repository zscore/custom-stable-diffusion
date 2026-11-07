#!/usr/bin/env python3
"""Queue SDXL text-to-image jobs on a running ComfyUI server and save the results.

Usage:
  python scripts/gen.py "a red fox in fresh snow, golden hour" -n 2
  python scripts/gen.py "portrait of an old fisherman" --fast --seed 42
"""
import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_CKPT = "juggernautXL_v9.safetensors"
FAST_LORA = "dmd2_sdxl_4step_lora_fp16.safetensors"
DEFAULT_NEG = "lowres, blurry, jpeg artifacts, watermark, text, signature, deformed, bad anatomy, extra fingers"


def api(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(SERVER + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def build_graph(a, seed):
    g = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.ckpt}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["ckpt", 1]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": a.width, "height": a.height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["ckpt", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": a.steps, "cfg": a.cfg, "sampler_name": a.sampler,
            "scheduler": a.scheduler, "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "gen"}},
    }
    if a.fast:
        g["lora"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["ckpt", 0], "clip": ["ckpt", 1], "lora_name": FAST_LORA,
            "strength_model": 1.0, "strength_clip": 1.0}}
        g["sample"]["inputs"]["model"] = ["lora", 0]
        g["pos"]["inputs"]["clip"] = ["lora", 1]
        g["neg"]["inputs"]["clip"] = ["lora", 1]
    return g


def run(graph):
    pid = str(uuid.uuid4())
    api("/prompt", {"prompt": graph, "client_id": "gen.py", "prompt_id": pid})
    while True:
        hist = json.loads(api(f"/history/{pid}"))
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("status_str") == "error":
                sys.exit(f"ComfyUI error: {json.dumps(status.get('messages'), indent=2)}")
            return [img for out in hist[pid]["outputs"].values() for img in out.get("images", [])]
        time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--negative", default=DEFAULT_NEG)
    p.add_argument("-n", "--count", type=int, default=1)
    p.add_argument("--seed", type=int, help="first seed; increments per image")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--fast", action="store_true", help="DMD2 LoRA: ~6 steps, cfg 1")
    p.add_argument("--name", default="", help="label added to output filenames")
    a = p.parse_args()

    # DMD2 wants lcm/sgm_uniform at cfg 1; Juggernaut likes dpmpp_2m/karras around cfg 4.5
    defaults = dict(steps=6, cfg=1.0, sampler="lcm", scheduler="sgm_uniform") if a.fast else \
               dict(steps=30, cfg=4.5, sampler="dpmpp_2m", scheduler="karras")
    for k, v in defaults.items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    OUT_DIR.mkdir(exist_ok=True)
    seed0 = a.seed if a.seed is not None else random.randint(0, 2**32 - 1)
    for i in range(a.count):
        seed = seed0 + i
        t0 = time.time()
        images = run(build_graph(a, seed))
        for img in images:
            q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
            stem = f"{datetime.now():%Y%m%d-%H%M%S}{'-' + a.name if a.name else ''}-{seed}"
            path = OUT_DIR / f"{stem}.png"
            path.write_bytes(api(f"/view?{q}"))
            meta = {k: getattr(a, k) for k in ("prompt", "negative", "width", "height", "steps", "cfg",
                                               "sampler", "scheduler", "ckpt", "fast")}
            (OUT_DIR / f"{stem}.json").write_text(json.dumps({**meta, "seed": seed}, indent=2))
            print(f"{path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
