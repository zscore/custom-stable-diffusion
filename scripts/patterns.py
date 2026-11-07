#!/usr/bin/env python3
"""Blue procedural background patterns, plus an animated vortex-lattice loop that can fill
the dark void areas of a generated background.

Run with the ComfyUI venv (needs numpy/scipy/cv2):
  PY=ComfyUI/.venv/bin/python
  $PY scripts/patterns.py sheets                    # option sheets -> outputs/animations/wip/
  $PY scripts/patterns.py loop --name vortex        # seamless vortex-lattice loop (mp4 + gif)
  $PY scripts/patterns.py fill outputs/bg.png       # vortex flow animated into bg's dark areas
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "animations"
BLUES = np.array([[.01, .01, .05], [.04, .08, .30], [.10, .35, .85], [.35, .75, 1.0], [.85, .95, 1.0]])


def blue(t):
    """Scalar in [0,1] -> navy / cobalt / cyan / ice, matching the roads and the homunculus."""
    t = np.clip(t, 0, 1)
    pos = np.linspace(0, 1, len(BLUES))
    return np.stack([np.interp(t, pos, BLUES[:, c]) for c in range(3)], -1).astype(np.float32)


def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


class Field:
    """Shared pixel grid + noise helpers at a fixed resolution."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def fbm(self, octaves=6, base=4, seed=0):
        r = np.random.default_rng(seed)
        out = np.zeros((self.H, self.W), np.float32)
        amp = 1
        for o in range(octaves):
            g = base * 2 ** o
            grid = r.random((int(g * self.H / self.W) + 2, g + 2)).astype(np.float32)
            out += amp * cv2.resize(grid, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            amp *= 0.5
        return norm(out)

    def warp(self, img, dx, dy, k):
        mx, my = (self.xx + k * dx).astype(np.float32), (self.yy + k * dy).astype(np.float32)
        return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def marble(self, seed=1, freq=18):
        """Domain-warped noise -> marbled ink bands."""
        a, b, c, d = (self.fbm(seed=seed + i) for i in range(4))
        m = self.warp(d, self.warp(c, a - .5, b - .5, 180) - .5, self.warp(a, c - .5, b - .5, 180) - .5, 260)
        return norm(np.sin(m * freq) * .5 + .5)

    def trace(self, vx, vy, n=1400, steps=70, step=2.0, seed=0):
        """Advect particles through a vector field; returns (n, steps+1, 2) trails."""
        r = np.random.default_rng(seed)
        mag = np.hypot(vx, vy) + 1e-6
        ux, uy = vx / mag, vy / mag
        p = r.random((n, 2)) * [self.W, self.H]
        trail = [p.copy()]
        for _ in range(steps):
            j = np.clip(p[:, 0], 0, self.W - 1).astype(int)
            i = np.clip(p[:, 1], 0, self.H - 1).astype(int)
            p = p + step * np.stack([ux[i, j], uy[i, j]], 1)
            trail.append(p.copy())
        return np.stack(trail, 1)

    def streamlines(self, vx, vy, **kw):
        acc = np.zeros((self.H, self.W), np.float32)
        cv2.polylines(acc, list(np.round(self.trace(vx, vy, **kw) * 4).astype(np.int32)), False, 0.25, 1,
                      cv2.LINE_AA, shift=2)
        return 0.75 * norm(np.log1p(acc * 3)) ** 1.3

    def vortex_field(self, spacing=110, drift=0.3, strength=40, seed=0):
        """Hex grid of alternating-spin vortices on top of a gentle rightward drift."""
        vx = np.full((self.H, self.W), drift, np.float32)
        vy = np.zeros((self.H, self.W), np.float32)
        for row, y0 in enumerate(np.arange(-spacing, self.H + spacing, spacing * 0.866)):
            for x0 in np.arange(-spacing, self.W + spacing, spacing) + (row % 2) * spacing / 2:
                dx, dy = self.xx - x0, self.yy - y0
                r2 = dx * dx + dy * dy + 300
                spin = 1 if (int(x0 // spacing) + row) % 2 else -1
                vx += spin * -dy / r2 * strength
                vy += spin * dx / r2 * strength
        return vx, vy


# ---------- option sheets (how we picked the look) ----------

def label_grid(tiles, cols=2):
    out = []
    for label, t in tiles:
        im = (t[..., ::-1] * 255).astype(np.uint8).copy()
        cv2.putText(im, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(im, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.append(im)
    return np.vstack([np.hstack(out[i:i + cols]) for i in range(0, len(out), cols)])


def sheets(W=768, H=432):
    F, rng = Field(W, H), np.random.default_rng(3)
    xx, yy = F.xx, F.yy

    # round 1: pure patterns
    marble = blue(F.marble() ** 1.5 * 0.9)
    U, V = np.ones((H // 2, W // 2), np.float32), np.zeros((H // 2, W // 2), np.float32)
    m = rng.random(U.shape) > 0.985
    V[m], U[m] = 1, 0.5
    lap = lambda Z: np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4 * Z
    for _ in range(6000):  # Gray-Scott, coral/labyrinth regime
        uvv = U * V * V
        U += 0.16 * lap(U) - uvv + 0.0545 * (1 - U)
        V += 0.08 * lap(V) + uvv - (0.0545 + 0.062) * V
    coral = blue(norm(cv2.resize(V, (W, H), interpolation=cv2.INTER_CUBIC)) * 1.05)
    pts = rng.random((140, 2)) * [W, H]
    P = np.stack([xx + 60 * (F.fbm(seed=5) - .5), yy + 60 * (F.fbm(seed=6) - .5)], -1).reshape(-1, 2)
    dist, _ = cKDTree(pts).query(P, k=2)
    cells = blue(np.exp(-(dist[:, 1] - dist[:, 0]).reshape(H, W) / 5) * 0.95 + 0.25 * norm(dist[:, 0].reshape(H, W)))
    mo = sum(np.cos(np.hypot(xx - cx, yy - cy) * (0.08 + 0.02 * i)) for i, (cx, cy) in enumerate(rng.random((5, 2)) * [W, H]))
    ang = F.fbm(octaves=4, base=3, seed=7) * 4 * np.pi
    flow = blue(F.streamlines(np.cos(ang), np.sin(ang), n=2500, steps=60, step=2.5, seed=7) / 0.75)
    r = np.hypot(xx - W / 2, yy - H / 2)
    th = np.abs(((np.arctan2(yy - H / 2, xx - W / 2) + np.pi) % (np.pi / 4)) - np.pi / 8)
    kal = cv2.remap(norm(np.sin(marble.mean(-1) * 12)).astype(np.float32),
                    np.clip(r * np.cos(th) * 0.9, 0, W - 1).astype(np.float32),
                    np.clip(r * np.sin(th) * 0.9 + H * 0.1, 0, H - 1).astype(np.float32), cv2.INTER_LINEAR)
    s1 = label_grid([("1 marbled ink (domain warp)", marble), ("2 reaction-diffusion coral", coral),
                     ("3 warped voronoi lattice", cells), ("4 moire interference", blue(norm(mo) ** 2.2)),
                     ("5 flow-field streamlines", flow), ("6 kaleidoscope mandala", blue(norm(kal) * (1 - 0.6 * norm(r))))])

    # round 2: flow / ink that trace a hidden lattice
    pts = rng.random((70, 2)) * [W, H]
    P = np.stack([xx + 50 * (F.fbm(seed=5) - .5), yy + 50 * (F.fbm(seed=6) - .5)], -1).reshape(-1, 2)
    dist, idx = cKDTree(pts).query(P, k=2)
    edge = (dist[:, 1] - dist[:, 0]).reshape(H, W)
    membrane = np.exp(-edge / 6)
    gy, gx = np.gradient(cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 3))
    na = (F.fbm(octaves=4, base=3, seed=9) - .5) * 1.2
    vx, vy = -gy, gx  # tangent to the lattice's iso-lines
    A = F.streamlines(vx * np.cos(na) - vy * np.sin(na), vx * np.sin(na) + vy * np.cos(na), seed=1)
    cell_id = idx[:, 0].reshape(H, W)
    mb = np.zeros((H, W), np.float32)
    for k, sd in enumerate([1, 11, 21]):
        mb = np.where(cell_id % 3 == k, F.marble(seed=sd, freq=14 + 6 * k), mb)
    C = F.streamlines(*F.vortex_field(), n=1600, seed=2)
    holes = []
    for row, y0 in enumerate(np.arange(30, H, 95)):
        for x0 in np.arange(20, W, 120) + (row % 2) * 60:
            holes.append((x0 + rng.normal(0, 12), y0 + rng.normal(0, 10), rng.uniform(22, 40), rng.uniform(0.6, 1.0)))
    psi = yy.copy()  # stream function: uniform flow past elliptical "bone holes"
    inside = np.zeros((H, W), np.float32)
    for x0, y0, rad, sq in holes:
        dx, dy = xx - x0, (yy - y0) / sq
        psi -= rad * rad * dy / (dx * dx + dy * dy + 1)
        inside = np.maximum(inside, (dx * dx + dy * dy < rad * rad).astype(np.float32))
    psi += 30 * (F.fbm(octaves=3, base=2, seed=12) - .5)
    gy, gx = np.gradient(psi)
    D = F.streamlines(gy, -gx, n=1600, steps=90, seed=3)
    rim = cv2.GaussianBlur(cv2.morphologyEx(inside, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)), (0, 0), 2)
    s2 = label_grid([
        ("A flow traces voronoi lattice", blue(0.25 * F.marble(seed=20) * (1 - membrane) + 0.9 * A + 0.25 * membrane)),
        ("B marbled ink in lattice cells", blue(np.maximum(mb * 0.75 * (1 - membrane) ** 2, membrane * 0.95))),
        ("C vortex lattice flow", blue(0.9 * C + 0.2 * F.marble(seed=30))),
        ("D flow around porous holes", blue(np.clip(0.9 * D * (1 - inside) + 0.15 * F.marble(seed=40) * (1 - inside) + rim * 1.2, 0, 1)))])

    wip = OUT_DIR / "wip"
    wip.mkdir(parents=True, exist_ok=True)
    for name, img in [("5-background-patterns.png", s1), ("6-structured-flow.png", s2)]:
        cv2.imwrite(str(wip / name), img)
        print(wip / name)


# ---------- animated vortex lattice ----------

class VortexLoop:
    """Vortex-lattice streamlines with light pulses travelling along them; seamless over phase 0..1.

    Each streamline segment deposits a phasor e^(i*2pi*position_along_line) into the pixels it covers.
    Rotating that phasor per frame slides bright pulses down every line, at the cost of one
    multiply per frame, so the expensive tracing happens only once.
    """

    def __init__(self, W, H, pulses=3, seed=2):
        F = Field(W, H)
        self.F = F
        trails = F.trace(*F.vortex_field(), n=1600, steps=70, step=2.0, seed=seed)
        offsets = np.random.default_rng(seed).random(len(trails))
        dens = np.zeros((H, W), np.float32)
        re, im = np.zeros_like(dens), np.zeros_like(dens)
        pts = np.round(trails * 4).astype(np.int32)
        steps = trails.shape[1] - 1
        for n, line in enumerate(pts):
            for k in range(steps):
                ph = 2 * np.pi * (pulses * k / steps + offsets[n])
                a, b = tuple(line[k]), tuple(line[k + 1])
                cv2.line(dens, a, b, 0.25, 1, cv2.LINE_AA, shift=2)
                cv2.line(re, a, b, 0.25 * np.cos(ph), 1, cv2.LINE_AA, shift=2)
                cv2.line(im, a, b, 0.25 * np.sin(ph), 1, cv2.LINE_AA, shift=2)
        self.dens = 0.75 * norm(np.log1p(dens * 3)) ** 1.3
        mag = np.hypot(re, im) + 1e-6
        self.re, self.im = re / mag, im / mag
        self.ink = F.marble(seed=30)
        self.wx, self.wy = F.fbm(octaves=3, base=2, seed=40) - .5, F.fbm(octaves=3, base=2, seed=41) - .5
        self.pulses = pulses

    def frame(self, phase):
        th = 2 * np.pi * phase * self.pulses  # integer pulse cycles per loop -> seamless
        pulse = 0.5 + 0.5 * (self.re * np.cos(th) + self.im * np.sin(th))
        lines = self.dens * (0.35 + 0.9 * pulse ** 2)
        ink = 0.2 * (0.5 + 0.5 * np.sin(2 * np.pi * (self.ink + phase)))  # marble bands cycle through
        v = lines + ink
        # slow breathing warp, periodic in phase
        k = 14 * np.sin(2 * np.pi * phase)
        v = self.F.warp(v.astype(np.float32), self.wx, self.wy, k)
        return blue(v)


def void_mask(bg, protect=(), thresh=0.11, min_area=0.0005, fade=10):
    """Dark areas of a background -> 1 (where the pattern shows through), 0 on the structure.

    Hard threshold first, then fade *inward* from the edges with a distance transform, so the
    pattern never bleeds onto the structure the way blurring the mask across edges did.
    protect: polygons in 0..1 image coords that are never filled, for dark parts of the
    structure that are as black as the void (shadowed walkways).
    """
    lum = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY), (0, 0), 1)
    dark = (lum < thresh).astype(np.uint8)
    h, w = dark.shape
    for poly in protect:
        cv2.fillPoly(dark, [np.round(np.array(poly) * [w, h]).astype(np.int32)], 0)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > min_area * dark.size  # drop little crevices/shadows
    dark = keep[labels].astype(np.uint8)
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    return np.clip(dist / fade, 0, 1).astype(np.float32)[..., None]


def encode(frames_dir, name, fps):
    mp4, gif = OUT_DIR / f"{name}.mp4", OUT_DIR / f"{name}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%04d.png"),
                    "-vf", "scale=512:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a",
                    "-loop", "0", str(gif)], check=True)
    print(mp4)
    print(gif)


def render_loop(a, bg=None):
    W, H = (bg.shape[1], bg.shape[0]) if bg is not None else (a.width, a.height)
    loop = VortexLoop(W, H, pulses=a.pulses)
    mask = void_mask(bg, a.protect) if bg is not None else None
    frames_dir = OUT_DIR / f"{a.name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    for f in range(a.frames):
        img = loop.frame(f / a.frames)
        if bg is not None:
            # pattern fills the void; screen-blend a little into the rest so it feels lit by it
            img = bg * (1 - mask) + np.maximum(bg, img * a.strength) * mask
        cv2.imwrite(str(frames_dir / f"{f:04d}.png"), (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))
        print(f"\rframe {f + 1}/{a.frames}", end="", flush=True)
    print()
    encode(frames_dir, a.name, a.fps)


def parse_poly(text):
    return [tuple(map(float, pt.split(","))) for pt in text.split()]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheets", help="render the pattern option sheets")
    for name in ("loop", "fill"):
        s = sub.add_parser(name)
        if name == "fill":
            s.add_argument("background", help="image whose dark void areas get the pattern")
            s.add_argument("--strength", type=float, default=0.8)
            s.add_argument("--protect", action="append", default=[], type=parse_poly,
                           help='never fill this polygon, "x,y x,y ..." in 0..1 image coords; repeatable')
        s.add_argument("--name", default="vortex" if name == "loop" else "vortex-fill")
        s.add_argument("--width", type=int, default=768)
        s.add_argument("--height", type=int, default=432)
        s.add_argument("--frames", type=int, default=120)
        s.add_argument("--fps", type=int, default=15)
        s.add_argument("--pulses", type=int, default=3, help="light pulses per streamline per loop")
    a = p.parse_args()

    if a.cmd == "sheets":
        sheets()
    elif a.cmd == "loop":
        render_loop(a)
    else:
        bg = cv2.imread(a.background)[..., ::-1].astype(np.float32) / 255
        if bg.shape[1] > a.width:  # keep frame size manageable
            bg = cv2.resize(bg, (a.width, int(bg.shape[0] * a.width / bg.shape[1])), interpolation=cv2.INTER_AREA)
        render_loop(a, bg)


if __name__ == "__main__":
    main()
