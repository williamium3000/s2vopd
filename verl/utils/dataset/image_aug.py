"""Configurable student-view image augmentation for S2VOPD.

This is the augmentation space that manufactures the teacher-student asymmetry: the
STUDENT sees an augmented image while the TEACHER sees the clean image, and the
self-distillation loss enforces consistency between the
two views. Only the augmentation is added here — everything else uses Vision-OPD config.

Usage (in the dataset):
    aug = StudentImageAugmenter(aug_config="aug_configs/swa_default.json", prob=0.5)
    student_pil = aug(clean_pil)   # strong-augmented with probability `prob`, else clean

`aug_config` is a JSON file (path), a list of op dicts, or a dict
`{"policy": "sequential"|"random_pick_n", "n": int?, "ops": [...]}`. Each op dict has a
"type" and a per-op "prob" plus its parameters (see _DEFAULT_AUG_OPS). This makes it easy
to try different augmentation recipes without touching code.
"""

import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# DDPM forward-process schedule for the (optional) gaussian_noise op.
_DIFFUSION_NUM_STEPS = 1000
try:
    import torch

    _betas = torch.sigmoid(torch.linspace(-6, 6, _DIFFUSION_NUM_STEPS)) * (0.5e-2 - 1e-5) + 1e-5
    _alphas_bar = torch.cumprod(1.0 - _betas, dim=0)
    _DIFFUSION_SIGNAL_COEF = torch.sqrt(_alphas_bar)  # sqrt(ᾱ_t)
    _DIFFUSION_NOISE_COEF = torch.sqrt(1 - _alphas_bar)  # sqrt(1 - ᾱ_t)
except Exception:  # torch optional; only needed for gaussian_noise
    torch = None

# Fallback pipeline used when no recipe file is given. Keys are flat so the JSON is
# easy to hand-edit. Ops with prob 0.0 are disabled placeholders — bump prob to enable.
_DEFAULT_AUG_OPS = [
    {"type": "rotate", "prob": 0.50, "angle_min": -35.0, "angle_max": 35.0, "fillcolor": [0, 0, 0]},
    {"type": "brightness", "prob": 0.30, "factor_min": 0.5, "factor_max": 1.5},
    {"type": "contrast", "prob": 0.30, "factor_min": 0.5, "factor_max": 1.8},
    {"type": "color", "prob": 0.30, "factor_min": 0.2, "factor_max": 1.8},
    {"type": "gaussian_blur", "prob": 0.35, "radius_min": 0.6, "radius_max": 2.0},
    # disabled by default — set prob > 0 to activate:
    {"type": "sharpness", "prob": 0.00, "factor_min": 0.2, "factor_max": 2.5},
    {"type": "crop", "prob": 0.00, "min_scale": 0.55},
    {"type": "mirror", "prob": 0.00},
    {"type": "posterize", "prob": 0.00, "bits": [3, 4]},
    {"type": "solarize", "prob": 0.00, "threshold_min": 64, "threshold_max": 192},
    {"type": "gaussian_noise", "prob": 0.00, "noise_step": 200},
    {"type": "crop_fill", "prob": 0.00, "crop_size": 0.40},
    {"type": "rotate_expand", "prob": 0.00, "angle_min": -35.0, "angle_max": 35.0},
    {"type": "downscale", "prob": 0.00, "scale_min": 0.3, "scale_max": 0.7, "resize_back": False},
    {"type": "translate", "prob": 0.00, "max_frac": 0.125, "fillcolor": [0, 0, 0]},
    {"type": "random_erasing", "prob": 0.00, "num_patches": 1, "area_min": 0.02, "area_max": 0.2,
     "ratio_min": 0.3, "ratio_max": 3.3, "fill": "random"},
    {"type": "hue_jitter", "prob": 0.00, "hue_min": -0.1, "hue_max": 0.1},
    {"type": "gamma", "prob": 0.00, "gamma_min": 0.7, "gamma_max": 1.5},
    {"type": "clahe", "prob": 0.00, "clip_limit": 2.0, "tile_grid": 8},
]


def load_aug_config(aug_config):
    """Return (policy, pick_n, ops). See module docstring for accepted forms."""
    if aug_config is None:
        return ("sequential", None, _DEFAULT_AUG_OPS)
    if isinstance(aug_config, (str, Path)):
        with open(aug_config) as f:
            cfg = json.load(f)
    else:
        cfg = aug_config
    if isinstance(cfg, list):
        return ("sequential", None, cfg)
    if isinstance(cfg, dict):
        policy = cfg.get("policy", "sequential")
        ops = cfg.get("ops", [])
        if not isinstance(ops, list):
            raise ValueError("aug_config dict must include an 'ops' list.")
        if policy == "sequential":
            return ("sequential", None, ops)
        if policy == "random_pick_n":
            n = cfg.get("n")
            if not isinstance(n, int) or n <= 0:
                raise ValueError("random_pick_n policy requires a positive integer 'n'.")
            return ("random_pick_n", n, ops)
        if policy == "exclusive":
            # Exactly one operator per sample, drawn by its "weight". Sequential stacking pushes a
            # single image past the point where the teacher's targets stay reachable; this keeps
            # every image at one operator's strength while still varying the operator across the
            # batch. Weights need not sum to one: the remainder leaves the image unaugmented.
            for op in ops:
                w = float(op.get("weight", 0.0))
                if w < 0.0:
                    raise ValueError("exclusive policy requires non-negative 'weight' per op.")
            if sum(float(op.get("weight", 0.0)) for op in ops) <= 0.0:
                raise ValueError("exclusive policy requires at least one positive 'weight'.")
            return ("exclusive", None, ops)
        raise ValueError(f"Unknown aug policy: {policy!r}")
    raise ValueError("aug_config must be a JSON array of ops or a dict with 'policy'/'ops'.")


class StudentImageAugmenter:
    """Applies a configurable strong augmentation to the student image with probability `prob`."""

    def __init__(self, aug_config=None, prob=0.5):
        self.prob = max(0.0, min(1.0, float(prob)))
        self.policy, self.pick_n, self.ops = load_aug_config(aug_config)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return an augmented copy with probability `prob`, otherwise a plain copy."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.prob <= 0.0 or random.random() >= self.prob:
            return image.copy()
        return self._augment(image.copy())

    def _select_ops(self):
        if self.policy == "random_pick_n":
            n = min(self.pick_n, len(self.ops))
            return random.sample(self.ops, n) if n > 0 else []
        if self.policy == "exclusive":
            weights = [float(op.get("weight", 0.0)) for op in self.ops]
            r = random.random() * max(1.0, sum(weights))
            acc = 0.0
            for op, w in zip(self.ops, weights):
                acc += w
                if r < acc:
                    chosen = dict(op)
                    chosen["prob"] = 1.0   # the weight already decided that this one applies
                    return [chosen]
            return []                      # residual mass: the image is left unaugmented
        return self.ops

    def _augment(self, image: Image.Image) -> Image.Image:
        queue = list(self._select_ops())
        while queue:
            op = queue.pop(0)
            prob = op.get("prob", 0.0)
            if prob <= 0 or random.random() >= prob:
                continue
            t = op["type"]
            if t == "compose":
                # Composite branch: run its sub-ops in order, each rolling its own prob.
                # Lets one "exclusive" weight select a multi-op recipe (e.g. downscale+cjitter).
                queue = list(op.get("ops", [])) + queue
                continue
            if t == "rotate":
                angle = random.uniform(op.get("angle_min", -35.0), op.get("angle_max", 35.0))
                image = image.rotate(angle, fillcolor=tuple(op.get("fillcolor", [0, 0, 0])))
            elif t == "brightness":
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(op.get("factor_min", 0.5), op.get("factor_max", 1.5)))
            elif t == "contrast":
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(op.get("factor_min", 0.5), op.get("factor_max", 1.8)))
            elif t == "color":
                image = ImageEnhance.Color(image).enhance(
                    random.uniform(op.get("factor_min", 0.2), op.get("factor_max", 1.8)))
            elif t == "sharpness":
                image = ImageEnhance.Sharpness(image).enhance(
                    random.uniform(op.get("factor_min", 0.2), op.get("factor_max", 2.5)))
            elif t == "gaussian_blur":
                r = random.uniform(op.get("radius_min", 0.6), op.get("radius_max", 2.0))
                image = image.filter(ImageFilter.GaussianBlur(radius=r))
            elif t == "crop":
                w, h = image.size
                min_scale = op.get("min_scale", 0.55)
                cw = random.randint(max(1, int(w * min_scale)), w)
                ch = random.randint(max(1, int(h * min_scale)), h)
                left = random.randint(0, max(0, w - cw))
                top = random.randint(0, max(0, h - ch))
                image = image.crop((left, top, left + cw, top + ch)).resize((w, h))
            elif t == "mirror":
                image = ImageOps.mirror(image)
            elif t == "posterize":
                image = ImageOps.posterize(image, bits=random.choice(op.get("bits", [3, 4])))
            elif t == "solarize":
                image = ImageOps.solarize(
                    image, threshold=random.randint(op.get("threshold_min", 64), op.get("threshold_max", 192)))
            elif t == "gaussian_noise":
                if torch is None:
                    continue
                step = max(0, min(int(op.get("noise_step", 200)), _DIFFUSION_NUM_STEPS - 1))
                img_t = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
                noisy = torch.clamp(
                    _DIFFUSION_SIGNAL_COEF[step] * img_t + _DIFFUSION_NOISE_COEF[step] * torch.randn_like(img_t),
                    0.0, 1.0)
                image = Image.fromarray((noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            elif t == "crop_fill":
                w, h = image.size
                cs = op.get("crop_size", 0.40)
                cw, ch = int(w * cs), int(h * cs)
                if w > cw and h > ch:
                    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
                else:
                    x, y, cw, ch = 0, 0, min(cw, w), min(ch, h)
                arr = np.array(image)
                arr[y:y + ch, x:x + cw] = 0
                image = Image.fromarray(arr)
            elif t == "rotate_expand":
                angle = random.uniform(op.get("angle_min", -35.0), op.get("angle_max", 35.0))
                image = image.rotate(angle, resample=Image.BILINEAR, expand=True)
            elif t == "downscale":
                w, h = image.size
                scale = random.uniform(op.get("scale_min", 0.3), op.get("scale_max", 0.7))
                image = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
                if op.get("resize_back", False):
                    image = image.resize((w, h), Image.BILINEAR)
            elif t == "mixed_resolution":
                # Non-uniform information allocation (H7): the canvas is a moderate downscale
                # (so the token budget matches a plain downscale of the same size), but the
                # detail is split — the background carries only a deep-downscale's worth of
                # information while one random block keeps the canvas's full fidelity. Total
                # information sits between the two scales; only its spatial structure differs
                # from a uniform downscale, which is exactly the variable under test.
                w, h = image.size
                canvas = random.uniform(op.get("canvas_min", 0.5), op.get("canvas_max", 0.7))
                bg = random.uniform(op.get("bg_min", 0.2), op.get("bg_max", 0.35))
                cw, ch = max(1, int(w * canvas)), max(1, int(h * canvas))
                out = image.resize((max(1, int(w * bg)), max(1, int(h * bg))), Image.BILINEAR)
                out = out.resize((cw, ch), Image.BILINEAR)
                for _ in range(op.get("num_patches", 1)):
                    fa = random.uniform(op.get("area_min", 0.15), op.get("area_max", 0.35))
                    pw, ph = max(1, int(cw * fa ** 0.5)), max(1, int(ch * fa ** 0.5))
                    if pw >= cw or ph >= ch:
                        continue
                    x, y = random.randint(0, cw - pw), random.randint(0, ch - ph)
                    # the same region of the original, brought down to canvas fidelity only
                    src = image.crop((int(x / canvas), int(y / canvas),
                                      min(w, int((x + pw) / canvas)), min(h, int((y + ph) / canvas))))
                    out.paste(src.resize((pw, ph), Image.BILINEAR), (x, y))
                image = out
            elif t == "translate":
                w, h = image.size
                mf = op.get("max_frac", 0.125)
                dx = int(round(random.uniform(-mf, mf) * w))
                dy = int(round(random.uniform(-mf, mf) * h))
                image = image.transform(
                    (w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                    resample=Image.BILINEAR, fillcolor=tuple(op.get("fillcolor", [0, 0, 0])))
            elif t == "random_erasing":
                arr = np.array(image)
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                h, w, c = arr.shape
                fill = op.get("fill", "random")
                for _ in range(op.get("num_patches", 1)):
                    area = random.uniform(op.get("area_min", 0.02), op.get("area_max", 0.2)) * w * h
                    ratio = random.uniform(op.get("ratio_min", 0.3), op.get("ratio_max", 3.3))
                    eh, ew = int(round((area * ratio) ** 0.5)), int(round((area / ratio) ** 0.5))
                    if 0 < eh < h and 0 < ew < w:
                        y, x = random.randint(0, h - eh), random.randint(0, w - ew)
                        if fill == "random":
                            arr[y:y + eh, x:x + ew] = np.random.randint(0, 256, (eh, ew, c), dtype=arr.dtype)
                        else:
                            arr[y:y + eh, x:x + ew] = fill
                image = Image.fromarray(arr.squeeze())
            elif t == "grid_mask":
                # Regular lattice occlusion (GridMask). Unlike scattered rectangles, a periodic
                # grid can never swallow a whole object, so every region keeps partial evidence
                # while total coverage stays uniform across the image.
                arr = np.array(image)
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                h, w, c = arr.shape
                fill = op.get("fill", 128)
                d = int(round(random.uniform(op.get("d_min", 0.1), op.get("d_max", 0.3)) * min(h, w)))
                if d > 1:
                    keep = random.uniform(op.get("keep_min", 0.5), op.get("keep_max", 0.7))
                    m = max(1, int(round(d * (1 - keep))))
                    oy, ox = random.randint(0, d - 1), random.randint(0, d - 1)
                    for y in range(oy - d, h, d):
                        for x in range(ox - d, w, d):
                            y0, x0 = max(y, 0), max(x, 0)
                            y1, x1 = min(y + m, h), min(x + m, w)
                            if y1 > y0 and x1 > x0:
                                if fill == "random":
                                    arr[y0:y1, x0:x1] = np.random.randint(0, 256, (y1 - y0, x1 - x0, c), dtype=arr.dtype)
                                else:
                                    arr[y0:y1, x0:x1] = fill
                image = Image.fromarray(arr.squeeze())
            elif t == "spectral_mask":
                # Frequency-band removal: the general form of the low-pass family (downscale /
                # blur / pixelate are all special cases). Zeroes an annulus of the centered 2D
                # spectrum, so no pixel survives untouched — this deletes an information band
                # globally rather than a region locally.
                arr = np.array(image).astype(np.float32)
                squeeze = arr.ndim == 2
                if squeeze:
                    arr = arr[:, :, None]
                h, w, c = arr.shape
                yy = (np.arange(h) - h / 2.0)[:, None] / (h / 2.0)
                xx = (np.arange(w) - w / 2.0)[None, :] / (w / 2.0)
                rad = np.sqrt(yy ** 2 + xx ** 2) / np.sqrt(2.0)
                lo = random.uniform(op.get("band_min", 0.0), op.get("band_max", 0.5))
                width = random.uniform(op.get("width_min", 0.1), op.get("width_max", 0.3))
                mode = op.get("mode", "band_stop")
                if mode == "low_pass":
                    keep = rad <= lo + width
                elif mode == "high_pass":
                    keep = rad >= lo
                else:
                    keep = (rad < lo) | (rad > lo + width)
                out = np.empty_like(arr)
                for ch in range(c):
                    f = np.fft.fftshift(np.fft.fft2(arr[:, :, ch]))
                    out[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f * keep)))
                arr = np.clip(out, 0, 255).astype(np.uint8)
                image = Image.fromarray(arr.squeeze() if squeeze else arr)
            elif t == "local_pixelate":
                # Mosaic random rectangles: softer than erasing — the "something is there"
                # cue survives but fine detail is destroyed, pushing the student to match
                # the clean-image teacher's detail distribution.
                w, h = image.size
                for _ in range(op.get("num_patches", 3)):
                    area = random.uniform(op.get("area_min", 0.05), op.get("area_max", 0.25)) * w * h
                    ratio = random.uniform(op.get("ratio_min", 0.3), op.get("ratio_max", 3.3))
                    ph, pw = int(round((area * ratio) ** 0.5)), int(round((area / ratio) ** 0.5))
                    if 0 < ph < h and 0 < pw < w:
                        y, x = random.randint(0, h - ph), random.randint(0, w - pw)
                        block = random.randint(int(op.get("pixel_min", 20)), int(op.get("pixel_max", 48)))
                        region = image.crop((x, y, x + pw, y + ph))
                        small = region.resize((max(1, pw // block), max(1, ph // block)), Image.BILINEAR)
                        image.paste(small.resize((pw, ph), Image.NEAREST), (x, y))
            elif t == "zoom_out":
                # Shrink the whole image onto a same-size padded canvas (random placement):
                # scale perturbation — objects get smaller but nothing is hidden, targeting
                # scale-robust fine perception (vstar-style small objects).
                w, h = image.size
                s = random.uniform(op.get("scale_min", 0.4), op.get("scale_max", 0.8))
                nw, nh = max(1, int(w * s)), max(1, int(h * s))
                small = image.resize((nw, nh), Image.BILINEAR)
                canvas = Image.new("RGB", (w, h), tuple(op.get("fillcolor", [0, 0, 0])))
                canvas.paste(small, (random.randint(0, w - nw), random.randint(0, h - nh)))
                image = canvas
            elif t == "hue_jitter":
                delta = random.uniform(op.get("hue_min", -0.1), op.get("hue_max", 0.1))
                hsv = np.array(image.convert("HSV")).astype(np.int16)
                hsv[..., 0] = (hsv[..., 0] + int(round(delta * 255))) % 256
                image = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
            elif t == "gamma":
                g = random.uniform(op.get("gamma_min", 0.7), op.get("gamma_max", 1.5))
                lut = [int(round(255 * ((i / 255.0) ** g))) for i in range(256)]
                image = image.point(lut * len(image.getbands()))
            elif t == "clahe":
                import cv2  # optional dep, only for clahe

                clip = op.get("clip_limit", 2.0)
                grid = int(op.get("tile_grid", 8))
                lab = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
                lab[..., 0] = clahe.apply(lab[..., 0])
                image = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
            else:
                raise ValueError(f"Unknown augmentation op type: {t!r}")
        return image.convert("RGB")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to PNG bytes (lossless) for the teacher passthrough column."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
