#!/usr/bin/env python3
"""A procession of homunculi down the bone ribbon, each more discombobulated than the last, while
afterimage trails and DeepDream ramp up over the clip. Linear timeline (not a loop).

Reuses dreamwalk.py's cutout/scene/water/dream pieces. Run with the ComfyUI venv:
  G=(outputs/*grumpy-walk-{1,2,4,5,6,3}-*.png); G=(${G/%/:flip})   # zsh
  $PY scripts/procession.py $G --background outputs/<bg>.png --path "..." --water 0.555 --protect "..."
  ... --no-dream          # choreography preview in ~1 min
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import dreamwalk as dw
import patterns

ROOT = Path(__file__).resolve().parent.parent


def ramp(t, t0, t1, power=1.0):
    """0 before t0, 1 after t1, smoothstep in between (optionally eased further by power)."""
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return float((x * x * (3 - 2 * x)) ** power)


def walker_pose(scene, u, f, level, hold, fps, finale=0.0, blast=13.0, vanish=False):
    """Pose for a walker at path position u in [0,1]; level scales how scrambled he is.
    finale in [0,1] blows him apart for good: pieces fly, spin and grow past the frame edges."""
    x, y, w = (np.interp(u, scene.ground, scene.pts[:, i]) for i in range(3))
    sh = w * scene.size
    step = f / hold  # one bob per pose change
    bob = abs(math.sin(np.pi * step)) * sh * 0.05
    wobble = (0.5 - 0.5 * math.cos(2 * np.pi * f / (fps * 1.3))) ** 2  # ~1.3s pulse
    fade_in = np.clip(u / 0.08, 0, 1)
    burst = np.clip((u - 0.88) / 0.12, 0, 1) ** 1.5  # end of the ribbon: scrambled ones blow apart
    env = level * (0.25 + 0.75 * wobble) + burst * (0.6 + level)
    fade = fade_in * (1 - burst)
    if finale > 0:
        env = level + blast * finale ** 2
        fade = fade_in * (1 - ramp(finale, 0.75, 0.97) if vanish else 1)
    return dict(foot=np.array([x, y - bob]), ground=(x, y), sh=sh, angle=0.0, env=env,
                fade=float(fade), grow=1 + 7 * finale ** 2.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sprites", nargs="+", help="character pngs on white; append :flip to mirror")
    p.add_argument("--background", required=True)
    p.add_argument("--path", required=True, help='"x,y,width ..." in 0..1 coords, far to near')
    p.add_argument("--protect", action="append", default=[])
    p.add_argument("--water", type=float)
    p.add_argument("--name", default="procession")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seconds", type=float, default=14)
    p.add_argument("--intro", type=float, default=1.0, help="seconds of empty scene before the first walker")
    p.add_argument("--spacing", type=float, default=1.6, help="seconds between the first two walkers")
    p.add_argument("--spacing-decay", type=float, default=0.85, help="each gap is this times the previous one")
    p.add_argument("--min-spacing", type=float, default=0.35)
    p.add_argument("--trip", type=float, default=6.0, help="seconds for the first walker to walk the path")
    p.add_argument("--min-trip", type=float, default=3.2, help="later walkers get faster, down to this")
    p.add_argument("--max-level", type=float, default=6.0, help="discombobulation of the very last walker")
    p.add_argument("--flicker", type=float, default=0.5, help="share of walkers hidden at any moment by the end")
    p.add_argument("--flicker-start", type=float, default=4.0)
    p.add_argument("--loop", action="store_true", help="everyone explodes off-screen at the end and it all "
                   "drains back to the empty opening frame, so the clip loops")
    p.add_argument("--finale", type=float, default=1.7, help="seconds over which the last walker explodes off-frame")
    p.add_argument("--hold", type=int, default=4)
    p.add_argument("--pieces", type=int, default=9)
    p.add_argument("--size", type=float, default=1.5)
    p.add_argument("--no-dream", action="store_true")
    p.add_argument("--dream-start", type=float, default=4.0, help="second at which DeepDream starts ramping in")
    p.add_argument("--dream-full", type=float, default=10.0, help="second at which it reaches full strength")
    p.add_argument("--dream-ease", type=float, default=0.7, help="<1 strengthens early, >1 holds back until late")
    p.add_argument("--layers", default="inception4a:1.0,guide:inception4b:6.0",
                   help='"layer:w" amplifies a layer (4a ~ architecture; 4c/4d ~ dogs), "guide:layer:w" pulls '
                        'toward --guide, "class:N:w" toward ImageNet class N')
    p.add_argument("--guide", default="outputs/animations/wip/guide-strip.png",
                   help="image the guided layers steer toward (a strip of grumpy homunculi)")
    p.add_argument("--octaves", type=int, default=5)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--step", type=float, default=0.025, help="DeepDream step at full ramp")
    p.add_argument("--feedback", type=float, default=0.45, help="dream carry-over at full ramp")
    p.add_argument("--char-protect", type=float, default=0.75)
    a = p.parse_args()

    rng = np.random.default_rng(1)
    sprites = []
    for spec in a.sprites:
        path, _, mod = spec.partition(":")
        rgba = dw.cutout(ROOT / path, flip=(mod == "flip"))
        sprites.append((rgba, *dw.shatter(rgba, a.pieces, rng)))

    T = int(a.seconds * a.fps)
    scene = dw.BackdropScene(a.width, a.height, T, ROOT / a.background, dw.patterns_poly(a.path),
                             [dw.patterns_poly(t) for t in a.protect], size=a.size, water=a.water)
    # (start, trip, level): an accelerating procession; gaps shrink, later walkers are faster and more scrambled
    starts, s, gap = [], a.intro, a.spacing
    while s <= a.seconds - a.finale - 1.0:
        starts.append(s)
        s += gap
        gap = max(a.min_spacing, gap * a.spacing_decay)
    n = len(starts)
    walkers = []
    for i, s in enumerate(starts):
        k = i / max(1, n - 1)
        walkers.append((s, a.trip + (a.min_trip - a.trip) * k, a.max_level * k ** 1.3))
    s, trip, level = walkers[-1]
    walkers[-1] = (s, max(trip, a.seconds - s + 0.01), level)  # the finale walker stays on until the end
    print(f"{n} walkers, starting at " + ", ".join(f"{w[0]:.1f}" for w in walkers))

    dreamer = None
    if not a.no_dream:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        guide = None
        if a.guide:
            guide = cv2.imread(str(ROOT / a.guide))[..., ::-1].astype(np.float32) / 255
        dreamer = dw.Dreamer({k: float(v) for k, v in (kv.rsplit(":", 1) for kv in a.layers.split(","))}, dev, guide)

    frames_dir = dw.OUT_DIR / f"{a.name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    H, W = a.height, a.width
    trail = np.zeros((H, W, 4), np.float32)  # premultiplied afterimage buffer
    prev_raw = prev_dream = None
    for f in range(T):
        t = f / a.fps
        img = scene.render(f)
        layer = np.zeros((H, W, 4), np.float32)
        # draw farthest first so nearer walkers overlap the ones behind them
        boom = a.seconds - a.finale
        # loop mode: everyone still on the ribbon when the finale starts stays for it and blows up
        live = [(min(1.0, (t - s) / trip), i) for i, (s, trip, _) in enumerate(walkers)
                if s <= t and (t < s + trip + 1e-6 or (a.loop and s + trip >= boom and t >= boom))]
        hide = a.flicker * ramp(t, a.flicker_start, boom)
        for u, i in sorted(live):
            if a.loop:
                lag = 0.35 * np.random.default_rng(i).random()  # staggered, so it ripples through the crowd
                finale = ramp(t, boom + lag, a.seconds - 0.2)
            else:
                finale = ramp(t, boom, a.seconds) if i == len(walkers) - 1 else 0.0
            # flicker in and out of existence in 2-frame blips; their trails keep fading meanwhile
            if finale == 0 and np.random.default_rng(i * 100003 + f // 2).random() < hide:
                continue
            spr, labels, seeds = sprites[(f // a.hold + 2 * i) % len(sprites)]
            pose = walker_pose(scene, u, f + 7 * i, walkers[i][2], a.hold, a.fps, finale,
                               blast=30.0 if a.loop else 13.0, vanish=a.loop)
            img, _ = dw.place_character(img, spr, labels, seeds, pose, t / 3)
            rgb, alpha = dw.place_character(np.zeros_like(img), spr, labels, seeds, pose, t / 3)
            layer = np.dstack([rgb, alpha]) + layer * (1 - alpha[..., None])

        # afterimages: decaying buffer of past character layers, tinted and shown behind the live ones
        decay = 0.6 + 0.3 * ramp(t, 3, a.seconds)
        trail = layer + trail * decay * (1 - layer[..., 3:4])
        ghost = trail * (0.85 * ramp(t, 3, a.seconds * 0.8))
        if a.loop:  # trails drain away so the last frame matches the empty opening
            clear = 1 - ramp(t, a.seconds - 0.7, a.seconds - 0.15)
            trail *= clear
            ghost *= clear
        tint = np.array([0.6, 0.85, 1.25], np.float32)
        live_a = layer[..., 3:4]
        frame = img + np.clip(ghost[..., :3] * tint, 0, 1) * (1 - live_a) * (1 - 0.5 * ghost[..., 3:4])
        frame = np.clip(frame, 0, 1).astype(np.float32)

        r = ramp(t, a.dream_start, a.dream_full, power=a.dream_ease)  # dream strength
        if a.loop:  # the blast blows the dream away too, back to the clean opening frame
            r *= 1 - ramp(t, a.seconds - 0.8, a.seconds - 0.1)
        if dreamer is not None and r > 0.02:
            fb = a.feedback * r * (1 - a.char_protect * cv2.GaussianBlur(live_a[..., 0], (0, 0), 6))[..., None]
            init = frame if prev_dream is None else (1 - fb) * frame + fb * dw.flow_warp(prev_dream, prev_raw, frame)
            out = dreamer.dream(init.astype(np.float32), octaves=a.octaves, scale=1.4, iters=a.iters,
                                step=a.step * r, seed=f)
            prev_raw, prev_dream = frame, out
        else:
            out = frame
        cv2.imwrite(str(frames_dir / f"{f:04d}.png"), (np.clip(out, 0, 1)[..., ::-1] * 255).astype(np.uint8))
        print(f"\rframe {f + 1}/{T}  dream {r:.2f}", end="", flush=True)
    print()
    patterns.encode(frames_dir, a.name, a.fps)


if __name__ == "__main__":
    main()
