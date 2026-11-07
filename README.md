# custom-stable-diffusion 🧪

A local Stable Diffusion setup built around **ComfyUI**, run headless and driven by Claude via ComfyUI's HTTP API. You describe what you want; Claude writes the prompt, queues the job, looks at the result, and iterates. The ComfyUI node editor is still there at http://127.0.0.1:8188 if you want to poke at it yourself.

Also home to a small, blue, deeply anxious (and lately very grumpy) homunculus, who is currently being DeepDreamed.

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

# IPAdapter: keep him on-model by passing reference images
$PY scripts/gen.py "cute cartoon blue homunculus creature, side view walking, very grumpy..." \
    --ckpt dreamshaper_8.safetensors --width 512 --height 512 --steps 25 --cfg 7 \
    --ref refs/homunculus-crouch.png refs/homunculus-armscrossed.png --ref-weight 0.7

# tile a bunch of images into one labeled sheet
$PY scripts/contact.py outputs/sheets/whatever.png outputs/*-homunc3-*.png

# animation tools (see "Grumpy dream walk" below)
$PY scripts/patterns.py sheets                                   # background pattern option sheets
$PY scripts/patterns.py fill outputs/<background>.png            # animated vortex flow in its dark void
$PY scripts/dreamwalk.py outputs/*grumpy-walk-*.png --hold 5     # DeepDreamed walk cycle (first pass)
```

Every image lands in `outputs/` next to a `.json` with its full recipe (prompt, negative, seed, sampler, model…), so anything can be reproduced or riffed on.

## What's in here

| Path | What |
|---|---|
| `setup.sh` | Rebuilds the whole ComfyUI install from scratch, pinned to exact commits |
| `start.sh` | Launches the ComfyUI server |
| `scripts/gen.py` | Text-to-image: builds a ComfyUI graph, queues it, saves PNG + JSON. Stdlib only |
| `scripts/contact.py` | Makes contact sheets for reviewing batches |
| `scripts/patterns.py` | Blue procedural backgrounds; animated vortex-lattice loop; fills a background's void with it |
| `scripts/dreamwalk.py` | Cuts the homunculus out of renders, walks him through a scene, shatters him, DeepDreams every frame |
| `refs/` | IPAdapter reference images of the homunculus |
| `outputs/` | Everything generated so far, with recipes |
| `outputs/animations/` | Rendered loops (mp4/gif) and `wip/` sheets from the animation experiments. Frame dumps are gitignored |
| `ComfyUI/` | *(gitignored, ~13GB)* ComfyUI, its venv, custom nodes, and models |

### Models

| Model | Base | Used for |
|---|---|---|
| Juggernaut XL v9 | SDXL | General / photoreal images |
| DMD2 4-step LoRA | SDXL | `--fast` mode (6 steps, cfg 1) |
| DreamShaper 8 | SD1.5 | Cartoony / stylized; the homunculus |
| AnimateLCM (motion module + LoRA) | SD1.5 | Fast AnimateDiff animation *(installed, not yet used)* |
| IP-Adapter Plus + CLIP ViT-H image encoder | SD1.5 | `--ref`: character consistency from reference images |
| GoogLeNet / Inception v1 (torchvision, auto-downloads ~50MB) | n/a | DeepDream, the original "everything is dogs" network |

### Custom nodes

ComfyUI-Manager, ComfyUI-AnimateDiff-Evolved, ComfyUI-VideoHelperSuite, ComfyUI_IPAdapter_plus.

## Hardware reality check

Running on an **M1 Pro, 16GB** via PyTorch's MPS backend. Measured times:

| Job | Time |
|---|---|
| SDXL, 30 steps, 1024² | ~3.5 min (first run, includes model load) |
| SDXL `--fast`, 6 steps | ~55s (about 20s sampling; the rest is memory shuffling, since 16GB is tight) |
| SD1.5, 25 steps, 512² | ~21s (~25s with IPAdapter) |
| SDXL `--fast` at 1344×768, after SD1.5 was loaded | 3+ min: model swapping under memory pressure |
| DeepDream, 768×432, 5 octaves × 15 iters | ~4–5s per frame on MPS |

SDXL is about the ceiling here. Flux, Wan 14B and friends technically run but would take many minutes per image. For serious video, rent a cloud GPU.

## The homunculus saga (so far)

The goal: an original nervous little blue homunculus. He's *inspired by* the vibe of a certain anxious MTG character but deliberately his own design, to stay clear of anyone's IP.

1. **Round 1:** four different style takes. One great result (peeking over books), one that drew a sad human boy instead, and one that looked like a screaming shark.
2. **Round 2:** locked the character description (bald round head, big pointy ears, huge frightened eyes, thin tail). Way more consistent. The **design sheet, seed `692341844`**, became the favorite.
3. **Round 3:** dialed up the fear and put him in situations (teacup, potion accident, storm). Adorably scared, but SD1.5 mostly ignored the props: no cat, no lantern, and the spellbook became two orbs.
4. **Pose variations:** same seed + prompt with only the pose swapped. Style holds; proportions and expression drift. Good, but not truly the *same* guy every time.

**Lesson:** prompt tricks get you "same species," not "same character."

5. **IPAdapter:** added `--ref` to `gen.py`. Feeding the crouch + arms-crossed refs (`refs/`) at weight 0.5 vs 0.75 across five poses ([sheet](outputs/sheets/ipadapter.png)): now it's recognizably *him* every time. 0.75 is more on-model; 0.5 lets the pose prompt win more often.
6. **Grumpy walk cycle:** six side-view "grumpy, scowling, stomping" walks at ref weight 0.7 (`*-grumpy-walk-*`). All face right with very furrowed brows: a usable stomp cycle.

## Grumpy dream walk (animation, in progress)

The idea: the grumpy homunculus stomps along looping, arching roads through a void with a wild procedural background, while his body comes apart into drifting pieces, and the whole thing is DeepDreamed (the 2015 Google thing that turned everything into dogs). Work-in-progress sheets live in [`outputs/animations/wip/`](outputs/animations/wip/).

### Pass 1: procedural everything (`scripts/dreamwalk.py`)

- **Cutout:** flood-fill the white background from the borders (so eye whites survive), drop big enclosed white pockets (the gap between his legs), keep the largest blob.
- **Discombobulation:** each sprite is Voronoi-split into ~9 pieces that fly apart and wobble on a pulsing envelope, then snap back together.
- **Scene:** Julia-set void, glowing trochoid loop-de-loop roads, and an aqueduct road he walks on. Everything is periodic in the loop length, so the video loops seamlessly.
- **DeepDream:** gradient ascent on GoogLeNet `inception4c` + `inception4d` over 5 octaves. Each frame starts from the previous dreamed frame, warped by optical flow, so the hallucinations ride the motion instead of flickering. Feedback is reduced on the character so his face stays readable. 30 warm-up frames are dreamed before frame 0 so the loop wraps cleanly.
- **Gotcha worth remembering:** torchvision's `googlenet` expects `_transform_input()` to run before `conv1`. Calling the layers directly skips it, and you get mushy crosshatch instead of eyes and dogs. With the fix you get the real thing: eyeballs, lizard-dogs, pagodas ([sweep](outputs/animations/wip/4-dream-settings-sweep.png)).

The procedural roads were too flat and garland-like, so the scene is being reworked (below).

### Pass 2: bone halls + procedural void

- **Background art:** SDXL backgrounds of bone-white ribbon walkways, lattice webbing and teal liquid, curving from the far distance into the foreground. Inspired by the mood of *Slay the Spire*'s Winding Halls but described in our own words, not copied. The first try drew an asphalt highway with lane lines (`*-bonehalls-*`), so the prompt now insists the walkway itself is bone and negative-prompts asphalt and road markings (`*-bonepath-*`).
- **Void pattern:** explored [six blue procedural patterns](outputs/animations/wip/5-background-patterns.png), then [four "structured" versions](outputs/animations/wip/6-structured-flow.png) where flow and ink follow a hidden lattice. Picked **C, the vortex lattice**: a hex grid of alternating-spin vortices steering streamlines.
- **Animation trick:** streamlines are traced once. Each segment also writes a phase into a per-pixel phasor buffer, so per frame a single rotation slides light pulses down every line. Add slowly cycling marbled ink and a periodic breathing warp, and it loops seamlessly in under 0.1s per frame.
- **Fill:** `patterns.py fill` finds the background's void and animates the vortex flow into it ([loop](outputs/animations/bonehalls1-vortex.mp4)).
- **Masking, round 1:** brightness threshold, blurred. The blur smeared the mask across edges and painted swirls onto the road ([v1](outputs/animations/wip/7-bonehalls-vortex-fill.png)).
- **Masking, round 2:** hard threshold, drop specks, then fade *inward* only with a distance transform, so nothing bleeds onto the structure. That still caught the far end of the road, which is as black and smooth as the void itself (brightness ~0.02 vs ~0.01), so no pixel statistic can separate them. `--protect "x,y x,y …"` polygons (0–1 coords) mark regions that must never be filled ([v2: outline + composite](outputs/animations/wip/8-void-mask-v2.png)):
  ```sh
  $PY scripts/patterns.py fill outputs/20261107-193514-bonehalls-2292325229.png --protect "0,1 0,.74 .69,.515 .80,.515 .79,1"
  ```

### Still to do

- Pick a bone-walkway background and trace the walking path on it, so he comes from the distance into the foreground and scales up as he approaches
- Composite the shattering walk cycle onto it, then run DeepDream over the whole thing

## Next up

- [x] **IPAdapter**: use the favorite image as a visual reference so every generation is *him*
- [ ] Finish the grumpy dream walk (see above)
- [ ] **AnimateDiff**: make him scurry around nervously (models already downloaded; needs an `animate.py`)
- [ ] Maybe train a character LoRA on the best ~15–20 images for real consistency
- [ ] Upscaling, img2img, inpainting as needed
