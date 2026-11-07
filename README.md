# custom-stable-diffusion 🧪

A local Stable Diffusion setup built around **ComfyUI**, run headless and driven by Claude via ComfyUI's HTTP API. You describe what you want; Claude writes the prompt, queues the job, looks at the result, and iterates. The ComfyUI node editor is still there at http://127.0.0.1:8188 if you want to poke at it yourself.

Also home to a small, blue, deeply anxious homunculus.

![homunculus poses](outputs/sheets/poses.png)

## Quick start

```sh
./setup.sh        # one time: clones ComfyUI + nodes at pinned commits, installs deps, downloads ~11GB of models
./start.sh        # starts ComfyUI on http://127.0.0.1:8188
```

Then, in another terminal:

```sh
PY=ComfyUI/.venv/bin/python

# SDXL (Juggernaut XL), photoreal-leaning, 1024x1024
$PY scripts/gen.py "a red fox in fresh snow at golden hour" -n 2
$PY scripts/gen.py "a cozy cabin at night" --fast          # DMD2 speed LoRA

# SD1.5 (DreamShaper 8), better for cartoony stuff, and what AnimateDiff uses
$PY scripts/gen.py "cute cartoon blue homunculus..." --ckpt dreamshaper_8.safetensors \
    --width 512 --height 512 --steps 25 --cfg 7 --seed 692341844

# tile a bunch of images into one labeled sheet
$PY scripts/contact.py outputs/sheets/whatever.png outputs/*-homunc3-*.png
```

Every image lands in `outputs/` next to a `.json` with its full recipe (prompt, negative, seed, sampler, model…), so anything can be reproduced or riffed on.

## What's in here

| Path | What |
|---|---|
| `setup.sh` | Rebuilds the whole ComfyUI install from scratch, pinned to exact commits |
| `start.sh` | Launches the ComfyUI server |
| `scripts/gen.py` | Text-to-image: builds a ComfyUI graph, queues it, saves PNG + JSON. Stdlib only |
| `scripts/contact.py` | Makes contact sheets for reviewing batches |
| `outputs/` | Everything generated so far, with recipes |
| `ComfyUI/` | *(gitignored, ~13GB)* ComfyUI, its venv, custom nodes, and models |

### Models

| Model | Base | Used for |
|---|---|---|
| Juggernaut XL v9 | SDXL | General / photoreal images |
| DMD2 4-step LoRA | SDXL | `--fast` mode (6 steps, cfg 1) |
| DreamShaper 8 | SD1.5 | Cartoony / stylized; the homunculus |
| AnimateLCM (motion module + LoRA) | SD1.5 | Fast AnimateDiff animation *(installed, not yet used)* |

### Custom nodes

ComfyUI-Manager, ComfyUI-AnimateDiff-Evolved, ComfyUI-VideoHelperSuite.

## Hardware reality check

Running on an **M1 Pro, 16GB** via PyTorch's MPS backend. Measured times:

| Job | Time |
|---|---|
| SDXL, 30 steps, 1024² | ~3.5 min (first run, includes model load) |
| SDXL `--fast`, 6 steps | ~55s (about 20s sampling; the rest is memory shuffling, since 16GB is tight) |
| SD1.5, 25 steps, 512² | ~21s |

SDXL is about the ceiling here. Flux, Wan 14B and friends technically run but would take many minutes per image. For serious video, rent a cloud GPU.

## The homunculus saga (so far)

The goal: an original nervous little blue homunculus. He's *inspired by* the vibe of a certain anxious MTG character but deliberately his own design, to stay clear of anyone's IP.

1. **Round 1:** four different style takes. One great result (peeking over books), one that drew a sad human boy instead, and one that looked like a screaming shark.
2. **Round 2:** locked the character description (bald round head, big pointy ears, huge frightened eyes, thin tail). Way more consistent. The **design sheet, seed `692341844`**, became the favorite.
3. **Round 3:** dialed up the fear and put him in situations (teacup, potion accident, storm). Adorably scared, but SD1.5 mostly ignored the props: no cat, no lantern, and the spellbook became two orbs.
4. **Pose variations:** same seed + prompt with only the pose swapped. Style holds; proportions and expression drift. Good, but not truly the *same* guy every time.

**Lesson:** prompt tricks get you "same species," not "same character."

## Next up

- [ ] **IPAdapter**: use the favorite image as a visual reference so every generation is *him*
- [ ] **AnimateDiff**: make him scurry around nervously (models already downloaded; needs an `animate.py`)
- [ ] Maybe train a character LoRA on the best ~15–20 images for real consistency
- [ ] Upscaling, img2img, inpainting as needed
