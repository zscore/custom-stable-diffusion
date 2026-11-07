#!/usr/bin/env python3
"""Grumpy homunculus walks looping blue roads through a fractal void, then gets DeepDreamed.

Cuts the character out of white-background renders, composites it onto a procedurally drawn
scene (Julia-set void + arched roads), shatters it into drifting pieces, and runs classic
GoogLeNet DeepDream over every frame with optical-flow feedback so the hallucinations flow.
Everything is periodic in --frames, so the result loops seamlessly.

Run with the ComfyUI venv (needs torch/torchvision/cv2):
  ComfyUI/.venv/bin/python scripts/dreamwalk.py outputs/a.png outputs/b.png:flip --name walk
  ... --no-dream          # just the raw composite, fast preview
"""
import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "animations"


# ---------- character cutout ----------

def cutout(path, flip=False):
    """White background -> RGBA, cropped. Flood-fills from the border so white eyes stay opaque."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    nearwhite = ((img.min(axis=2) > 215) & (np.ptp(img, axis=2) < 25)).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1)]:
        if nearwhite[seed[1], seed[0]]:
            cv2.floodFill(nearwhite, mask, seed, 2)
    bg = (nearwhite == 2).astype(np.uint8)
    fg = 1 - bg
    # keep only the largest blob (drops stray splotches/signatures), fill its holes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        fg = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # big enclosed white pockets (gap between legs) are background too; small ones are eye whites
    n, labels, stats, _ = cv2.connectedComponentsWithStats(nearwhite * fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.002 * h * w:
            fg[labels == i] = 0
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 0)
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = img[..., ::-1].astype(np.float32) / 255 * alpha[..., None]  # premultiplied
    rgba = np.dstack([rgb, alpha])[y0:y1, x0:x1]
    return rgba[:, ::-1].copy() if flip else rgba


def shatter(rgba, n_pieces, rng):
    """Voronoi-partition the sprite into pieces; returns label map and per-piece seed points."""
    h, w = rgba.shape[:2]
    ys, xs = np.nonzero(rgba[..., 3] > 0.5)
    pick = rng.choice(len(xs), n_pieces, replace=False)
    seeds = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (gx[..., None] - seeds[:, 0]) ** 2 + (gy[..., None] - seeds[:, 1]) ** 2
    return d.argmin(-1), seeds


# ---------- scene ----------

def cosine_palette(t, a, b, c, d):
    return a + b * np.cos(2 * np.pi * (c * t[..., None] + d))


def fractal_void(W, H, phase, scale=0.75):
    """Julia set whose c breathes on a closed path, so it loops with phase in [0,1)."""
    w, h = int(W * scale), int(H * scale)
    zoom = 0.85 + 0.1 * math.sin(2 * np.pi * phase)
    rot = 0.25 * math.sin(2 * np.pi * phase)
    y, x = np.mgrid[-1:1:h * 1j, -W / H:W / H:w * 1j] * zoom
    z = (x + 1j * y) * np.exp(1j * rot)
    c = 0.7885 * np.exp(1j * (2.6 + 0.12 * math.sin(2 * np.pi * phase) + 0.05 * math.cos(4 * np.pi * phase)))
    smooth = np.zeros(z.shape)
    alive = np.ones(z.shape, bool)
    for i in range(48):
        z[alive] = z[alive] ** 2 + c
        esc = alive & (np.abs(z) > 4)
        smooth[esc] = i + 1 - np.log2(np.log(np.abs(z[esc])))
        alive &= ~esc
    t = np.log1p(smooth) / np.log(49) * 2.2 + phase
    col = cosine_palette(t, np.array([.25, .15, .35]), np.array([.25, .2, .3]),
                         np.array([1.0, 1.0, 1.0]), np.array([.55, .7, .85]))
    col *= 0.35 + 0.65 * (np.log1p(smooth) / np.log(49))[..., None]  # fast-escape regions fade toward void
    col[alive] = (0.02, 0.0, 0.05)
    col = cv2.resize(col.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    vy, vx = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    return np.clip(col, 0, 1) * (1 - 0.55 * (vx ** 2 + vy ** 2))[..., None].clip(0, 1)


def draw_road(canvas, pts, width, core, edge, glow=0.6):
    """Thick glowing ribbon along pts (Nx2)."""
    H, W = canvas.shape[:2]
    p = np.round(pts * 4).astype(np.int32).reshape(-1, 1, 2)  # 2-bit subpixel for smooth AA
    layer = np.zeros((H, W, 3), np.float32)
    cv2.polylines(layer, [p], False, edge, width + 6, cv2.LINE_AA, shift=2)
    cv2.polylines(layer, [p], False, core, width, cv2.LINE_AA, shift=2)
    cv2.polylines(layer, [p], False, [min(1, v * 1.6) for v in core], max(1, width // 8), cv2.LINE_AA, shift=2)
    mask = np.zeros((H, W), np.float32)
    cv2.polylines(mask, [p], False, 1.0, width + 6, cv2.LINE_AA, shift=2)
    halo = cv2.GaussianBlur(layer, (0, 0), width * 0.9) * glow
    canvas[:] = canvas * (1 - mask[..., None]) + layer * mask[..., None] + halo
    np.clip(canvas, 0, 1, out=canvas)


class Scene:
    def __init__(self, W, H, frames):
        self.W, self.H, self.T = W, H, frames
        self.P = W * 0.55              # main road arch spacing (px)
        self.D = 2 * self.P            # scroll distance over one loop; every layer period divides it

    def main_road_y(self, x, s):
        u = (x + s) / self.P
        return self.H * 0.80 - self.H * 0.15 * np.sin(np.pi * u) ** 2

    def render(self, f):
        W, H, ph = self.W, self.H, f / self.T
        s = ph * self.D
        img = fractal_void(W, H, ph)

        # far layer: looping trochoid roads (loop-de-loops), slow parallax
        for k, (yc, r, sp, col, wid) in enumerate([(0.30, 1.9, 0.25, (.25, .45, .95), 7),
                                                    (0.50, 1.5, 0.5, (.2, .55, 1.0), 11)]):
            period = self.D * sp / (k + 1)  # (k+1) loops per (sp*D) of travel -> seamless
            a = period / (2 * np.pi)
            t = np.linspace(-2 * np.pi, (W / a) + 4 * np.pi, 1400)
            x = a * (t - r * np.sin(t)) - (s * sp) % period - period
            y = H * yc - a * r * 0.55 * np.cos(t)
            draw_road(img, np.stack([x, y], 1), wid, col, (.05, .1, .3), glow=0.35)

        # main road: aqueduct of arches, character walks on its humps
        xs = np.linspace(-20, W + 20, 900)
        ys = self.main_road_y(xs, s)
        pier = self.P / 2
        under = np.zeros((H, W), np.float32)
        for k in range(-2, int(W / pier) + 3):
            px = k * pier - (s % pier)
            top = int(self.main_road_y(np.array([px]), s)[0])
            cv2.line(under, (int(px), top), (int(px), H), 1.0, 10, cv2.LINE_AA)
            cv2.ellipse(under, (int(px + pier / 2), int(self.main_road_y(np.array([px + pier / 2]), s)[0] + 34)),
                        (int(pier / 2) - 6, 38), 0, 180, 360, 1.0, 6, cv2.LINE_AA)
        fade = np.linspace(0, 1, H)[:, None] ** 0.5
        under *= np.clip(1.4 - 1.2 * fade, 0, 1) * (np.linspace(0, 1, H)[:, None] > 0.55)
        under = cv2.GaussianBlur(under, (0, 0), 1.2)
        img = img * (1 - 0.8 * under[..., None]) + 0.8 * under[..., None] * np.array([.12, .28, .7])
        draw_road(img, np.stack([xs, ys], 1), 22, (.25, .55, 1.0), (.05, .12, .45), glow=0.7)
        return img

    def pose(self, f, n_steps, x_frac=0.36, height=0.36):
        ph = f / self.T
        s = ph * self.D
        sh = self.H * height
        cx = self.W * x_frac
        y_road = self.main_road_y(np.array([cx - 3, cx, cx + 3]), s)
        bob = abs(math.sin(np.pi * n_steps * ph)) * sh * 0.05
        return dict(foot=np.array([cx, y_road[1] - bob + sh * 0.02]), ground=(cx, y_road[1]), sh=sh,
                    angle=math.degrees(math.atan2(y_road[2] - y_road[0], 6)),
                    env=(0.5 - 0.5 * math.cos(2 * np.pi * 3 * ph)) ** 2, fade=1.0)


class BackdropScene:
    """A generated background with the vortex lattice animated into its void, and a hand-traced
    path the character walks from the distance into the foreground."""

    def __init__(self, W, H, frames, background, path, protect=(), size=1.5, strength=0.8, explode=0.15, water=None):
        import patterns
        self.W, self.H, self.T = W, H, frames
        bg = cv2.imread(str(background))[..., ::-1].astype(np.float32) / 255
        self.bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        self.mask = patterns.void_mask(self.bg, protect)
        self.vortex = patterns.VortexLoop(W, H)
        self.strength = strength
        # path points carry (x, y, walkway width) in 0..1 coords; width sets the character's scale
        pts = np.array(path, np.float32) * [W, H, W]
        # constant speed over the ground: screen distance divided by local scale
        seg = np.hypot(*np.diff(pts[:, :2], axis=0).T) / ((pts[1:, 2] + pts[:-1, 2]) / 2)
        self.ground = np.r_[0, np.cumsum(seg)] / seg.sum()
        self.pts, self.size, self.explode = pts, size, explode
        self.water = Water(self.bg, water) if water is not None else None

    def render(self, f):
        v = self.vortex.frame(f / self.T)
        img = self.bg * (1 - self.mask) + np.maximum(self.bg, v * self.strength) * self.mask
        if self.water:
            img = self.water.apply(img, v * self.mask, f / self.T)
        return img


    def pose(self, f, n_steps):
        ph = f / self.T
        u = ph  # one trip down the path per loop
        x, y, w = (np.interp(u, self.ground, self.pts[:, i]) for i in range(3))
        sh = w * self.size
        bob = abs(math.sin(np.pi * n_steps * ph)) * sh * 0.05
        # at the end of the path he blows apart and fades, then reassembles at the far end
        tail = max(0.0, (ph - (1 - self.explode)) / self.explode)
        head = max(0.0, 1 - ph / (self.explode * 0.6))
        burst = max(tail, head) ** 1.5
        wobble = 0.35 * (0.5 - 0.5 * math.cos(2 * np.pi * 3 * ph)) ** 2
        return dict(foot=np.array([x, y - bob]), ground=(x, y), sh=sh, angle=0.0,
                    env=wobble + 2.5 * burst, fade=1 - burst)


class Water:
    """Everything below the horizon that isn't bright bone is water: it reflects the swirling void
    and gets looping perspective waves, expanding ring ripples, and glints on the crests."""

    def __init__(self, bg, horizon, n_rings=7, seed=5):
        H, W = bg.shape[:2]
        self.H, self.W, self.yh = H, W, horizon * H
        hsv = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        below = yy > self.yh
        val, sat = hsv[..., 2], hsv[..., 1]
        # bone: bright and colorless. Shadowed bone is dimmer, so also take dim colorless pixels that
        # are connected to bright bone close by (hysteresis); arch reflections further out stay water
        strong = (below & (val > 0.58)).astype(np.uint8)  # bone + bright coral
        # close the speckle so coral clumps (light/dark blue flecks) become solid
        strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) & below
        near = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
        weak = (below & (val > 0.15) & (sat < 0.3)).astype(np.uint8) & near | strong
        n, labels, stats, _ = cv2.connectedComponentsWithStats(weak)
        hit = np.zeros(n, bool)
        hit[np.unique(labels[strong > 0])] = True
        hit[0] = False
        bone = hit[labels].astype(np.uint8)
        # small enclosed pockets (perforation holes, gaps in the ribbon) count as bone too
        n, labels, stats, _ = cv2.connectedComponentsWithStats((below & (bone == 0)).astype(np.uint8))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.002 * H * W:
                bone[labels == i] = 1
        bone = cv2.dilate(bone, np.ones((5, 5), np.uint8))
        water = (below & (bone == 0)).astype(np.uint8)
        water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.mask = np.clip(cv2.distanceTransform(water, cv2.DIST_L2, 3) / 2.5, 0, 1)[..., None].astype(np.float32)
        # water source with the bone painted out, so ripples don't drag ribbon pixels into the pool
        self.src = cv2.inpaint((bg * 255).astype(np.uint8), bone, 6, cv2.INPAINT_TELEA).astype(np.float32) / 255
        # perspective: depth grows toward the horizon; world X/Z drive the wave phases
        dyh = np.maximum(yy - self.yh, 1.0)
        self.Z = H * 0.6 / (dyh + H * 0.04)  # capped so far-off waves don't alias into stripes
        self.X = (xx - W / 2) / W * self.Z * 2.5
        self.near = np.clip(dyh / (H * 0.18), 0, 1) ** 2  # calm toward the horizon
        self.mirror_y = np.clip(2 * self.yh - yy, 0, H - 1)
        self.fade = np.exp(-dyh / (H * 0.35))[..., None]
        self.xx, self.yy = xx, yy
        rng = np.random.default_rng(seed)
        self.waves = [(rng.uniform(-1.5, 1.5), rng.uniform(2, 5), rng.uniform(0.5, 1), rng.choice([1, 2, 3]), rng.uniform(0, 2 * np.pi))
                      for _ in range(5)]
        self.rings = [(rng.uniform(-3, 3), rng.uniform(1.2, 6), rng.random()) for _ in range(n_rings)]

    def height(self, ph):
        h = np.zeros_like(self.Z)
        for kx, kz, a, n, p0 in self.waves:
            h += a * np.sin(kx * self.X + kz * self.Z - 2 * np.pi * n * ph + p0)
        for x0, z0, off in self.rings:
            age = (ph + off) % 1.0  # each ring is born, spreads and dies once per loop
            r = np.hypot(self.X - x0, (self.Z - z0) * 1.0)
            R = 0.2 + age * 2.2
            h += 2.0 * np.sin(np.pi * age) * np.sin(9 * (r - R)) * np.exp(-((r - R) / 0.35) ** 2)
        return cv2.GaussianBlur((h * self.near).astype(np.float32), (0, 0), 1.2)

    def apply(self, img, void_pattern, ph):
        h = self.height(ph)
        gy, gx = np.gradient(h)
        px = np.clip(self.yy - self.yh, 0, None) / self.H  # waves look smaller (in px) far away
        dx, dy = gx * 40 * px, gy * 60 * px
        refl = cv2.remap(void_pattern.astype(np.float32), self.xx, self.mirror_y.astype(np.float32), cv2.INTER_LINEAR)
        dark = np.clip(1 - cv2.cvtColor(self.src, cv2.COLOR_RGB2HSV)[..., 2:3] / 0.5, 0, 1)
        surface = self.src + 0.7 * refl * self.fade * (0.3 + 0.7 * dark)
        warped = cv2.remap(surface.astype(np.float32), (self.xx + dx).astype(np.float32),
                           (self.yy + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        glint = np.clip(-gy * 4, 0, 1)[..., None] ** 2 * np.array([.55, .8, 1.0], np.float32) * 0.45
        return img * (1 - self.mask) + np.clip(warped + glint, 0, 1) * self.mask


# ---------- character placement ----------

def place_character(img, sprite, labels, seeds, pose, ph):
    H, W = img.shape[:2]
    sh, env, angle, foot = int(pose["sh"]), pose["env"], pose["angle"], pose["foot"]
    k = sh / sprite.shape[0]
    spr = cv2.resize(sprite, (max(1, int(sprite.shape[1] * k)), sh), interpolation=cv2.INTER_AREA) * pose["fade"]
    lab = cv2.resize(labels.astype(np.float32), (spr.shape[1], spr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(int)
    seeds = seeds * k

    # discombobulate: pieces fly out along the envelope, each with its own wobble
    center = np.array([spr.shape[1] / 2, spr.shape[0] / 2])
    rng = np.random.default_rng(7)
    out = np.zeros((H, W, 4), np.float32)
    for i in range(len(seeds)):
        piece = spr * (lab == i)[..., None]
        if piece[..., 3].max() == 0:
            continue
        d = seeds[i] - center
        d = d / (np.linalg.norm(d) + 1e-6)
        wob = rng.uniform(0, 2 * np.pi, 3)
        off = d * env * sh * 0.2 + sh * 0.06 * env * np.array(
            [math.sin(2 * np.pi * 2 * ph + wob[0]), math.cos(2 * np.pi * 3 * ph + wob[1])])
        rot = angle + env * 40 * math.sin(2 * np.pi * ph + wob[2])
        # sprite coords -> screen: anchor sprite bottom-center at the foot, rotate about the piece seed
        M = cv2.getRotationMatrix2D(tuple(map(float, seeds[i])), -rot, 1.0)
        M[:, 2] += foot - np.array([spr.shape[1] / 2, spr.shape[0]]) + off
        warped = cv2.warpAffine(piece, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        a = warped[..., 3:4]
        out[..., :3] = warped[..., :3] + out[..., :3] * (1 - a)
        out[..., 3:4] = a + out[..., 3:4] * (1 - a)
    # soft shadow on the ground
    shadow = np.zeros((H, W), np.float32)
    gx, gy = pose["ground"]
    cv2.ellipse(shadow, (int(gx), int(gy)), (max(1, int(sh * 0.28 * (1 - 0.5 * min(env, 1)))), max(1, int(sh * 0.04))),
                angle, 0, 360, 0.6 * pose["fade"], -1, cv2.LINE_AA)
    img = img * (1 - cv2.GaussianBlur(shadow, (0, 0), 4)[..., None])
    a = out[..., 3:4]
    return out[..., :3] + img * (1 - a), a[..., 0]


# ---------- deepdream ----------

class Dreamer:
    def __init__(self, layers, device):
        import torch
        import torchvision
        self.torch = torch
        self.dev = device
        net = torchvision.models.googlenet(weights="IMAGENET1K_V1").eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        self.acts = {}
        for name, w in layers.items():
            getattr(net, name).register_forward_hook(lambda m, i, o, name=name: self.acts.__setitem__(name, o))
        self.net, self.layers = net, layers
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _run_to_last_layer(self, x):
        n = self.net
        # stop after the deepest requested layer instead of running the classifier
        order = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3",
                 "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4",
                 "inception5a", "inception5b"]
        last = max(order.index(l) for l in self.layers)
        x = n._transform_input(x)  # torchvision's googlenet expects this before conv1
        for name in order[:last + 1]:
            x = getattr(n, name)(x)

    def dream(self, rgb, octaves=3, scale=1.4, iters=8, step=0.015, jitter=24, seed=0):
        torch, F = self.torch, self.torch.nn.functional
        g = torch.Generator().manual_seed(seed)
        base = (torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.dev) - self.mean) / self.std
        lo, hi = (0 - self.mean) / self.std, (1 - self.mean) / self.std
        pyramid = [base]
        for _ in range(octaves - 1):
            h, w = pyramid[-1].shape[-2:]
            pyramid.append(F.interpolate(pyramid[-1], size=(int(h / scale), int(w / scale)), mode="bilinear", align_corners=False))
        detail = torch.zeros_like(pyramid[-1])
        for oct_img in reversed(pyramid):
            detail = F.interpolate(detail, size=oct_img.shape[-2:], mode="bilinear", align_corners=False)
            x = (oct_img + detail).detach()
            for _ in range(iters):
                ox, oy = torch.randint(-jitter, jitter + 1, (2,), generator=g).tolist()
                x = torch.roll(x, (oy, ox), (2, 3)).requires_grad_(True)
                self.acts.clear()
                self._run_to_last_layer(x)
                loss = sum(w * self.acts[n].pow(2).mean() for n, w in self.layers.items())
                loss.backward()
                gr = x.grad / (x.grad.abs().mean() + 1e-8)
                x = torch.roll((x + step * gr).detach(), (-oy, -ox), (2, 3))
                x = torch.max(torch.min(x, hi), lo)
            detail = x - oct_img
        out = (x * self.std + self.mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return out


def flow_warp(prev_dream, prev_raw, cur_raw):
    """Warp last dreamed frame onto the current frame's geometry so hallucinations ride the motion."""
    g0 = cv2.cvtColor((prev_raw * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor((cur_raw * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 25, 3, 7, 1.5, 0)
    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(prev_dream, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def patterns_poly(text):
    return [tuple(map(float, pt.split(","))) for pt in text.split()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sprites", nargs="+", help="character pngs on white; append :flip to mirror")
    p.add_argument("--name", default="dreamwalk")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--frames", type=int, default=180, help="loop length")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--hold", type=int, default=6, help="frames per pose before switching")
    p.add_argument("--pieces", type=int, default=9, help="how many chunks the body breaks into")
    p.add_argument("--background", help="image to walk through instead of the procedural scene")
    p.add_argument("--path", help='walk path for --background: "x,y,width ..." in 0..1 coords, far to near')
    p.add_argument("--protect", action="append", default=[], help="polygon kept free of the void pattern (patterns.py fill)")
    p.add_argument("--water", type=float, help="horizon height (0..1): below it, non-bone pixels become rippling water")
    p.add_argument("--size", type=float, default=1.5, help="character height relative to the walkway width")
    p.add_argument("--no-dream", action="store_true")
    p.add_argument("--no-character", action="store_true", help="backdrop only (for stage-by-stage renders)")
    p.add_argument("--no-shatter", action="store_true", help="walk without discombobulating")
    p.add_argument("--layers", default="inception4c:1.0,inception4d:1.0", help="4c-4e is where the dogs live")
    p.add_argument("--octaves", type=int, default=5)
    p.add_argument("--octave-scale", type=float, default=1.4)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--feedback", type=float, default=0.45, help="how much of the previous dream carries over")
    p.add_argument("--char-protect", type=float, default=0.6, help="cut feedback on the character by this fraction")
    p.add_argument("--warmup", type=int, default=30, help="frames dreamed before the loop start so it wraps cleanly")
    p.add_argument("--limit", type=int, help="only render the first N frames (testing)")
    a = p.parse_args()

    rng = np.random.default_rng(1)
    sprites = []
    for spec in a.sprites:
        path, _, mod = spec.partition(":")
        rgba = cutout(ROOT / path if not Path(path).is_absolute() else path, flip=(mod == "flip"))
        sprites.append((rgba, *shatter(rgba, a.pieces, rng)))

    if a.background:
        protect = [patterns_poly(t) for t in a.protect]
        scene = BackdropScene(a.width, a.height, a.frames, a.background, patterns_poly(a.path), protect,
                              size=a.size, water=a.water)
    else:
        scene = Scene(a.width, a.height, a.frames)
    n_steps = a.frames // a.hold  # one bob per pose change

    def raw_frame(f):
        f %= a.frames
        img = scene.render(f)
        if a.no_character:
            return img.astype(np.float32), np.zeros(img.shape[:2], np.float32)
        spr, labels, seeds = sprites[(f // a.hold) % len(sprites)]
        pose = scene.pose(f, n_steps)
        if a.no_shatter:
            pose["env"] = 0.0  # stays in one piece; still fades in/out at the ends of the path
        rgb, alpha = place_character(img, spr, labels, seeds, pose, f / a.frames)
        return rgb.astype(np.float32), alpha

    frames_dir = OUT_DIR / f"{a.name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    total = a.limit or a.frames
    if a.no_dream:
        for f in range(total):
            cv2.imwrite(str(frames_dir / f"{f:04d}.png"), (raw_frame(f)[0][..., ::-1] * 255).astype(np.uint8))
            print(f"\rframe {f + 1}/{total}", end="", flush=True)
    else:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        layers = {k: float(v) for k, v in (kv.split(":") for kv in a.layers.split(","))}
        dreamer = Dreamer(layers, dev)
        prev_raw = prev_dream = None
        start = -a.warmup if a.limit is None else 0
        for f in range(start, total):
            raw, alpha = raw_frame(f)
            # less carry-over on the character so he stays readable instead of smearing into a ghost
            fb = a.feedback * (1 - a.char_protect * cv2.GaussianBlur(alpha, (0, 0), 6))[..., None]
            init = raw if prev_dream is None else (1 - fb) * raw + fb * flow_warp(prev_dream, prev_raw, raw)
            out = dreamer.dream(init.astype(np.float32), octaves=a.octaves, scale=a.octave_scale,
                                iters=a.iters, step=a.step, seed=f % a.frames)
            prev_raw, prev_dream = raw, out
            if f >= 0:
                cv2.imwrite(str(frames_dir / f"{f:04d}.png"), (out[..., ::-1] * 255).astype(np.uint8))
            print(f"\rframe {f + 1}/{total}", end="", flush=True)
    print()

    mp4 = OUT_DIR / f"{a.name}.mp4"
    gif = OUT_DIR / f"{a.name}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(frames_dir / "%04d.png"),
                    "-vf", "scale=512:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a",
                    "-loop", "0", str(gif)], check=True)
    print(mp4)
    print(gif)


if __name__ == "__main__":
    main()
