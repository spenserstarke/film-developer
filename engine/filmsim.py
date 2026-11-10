"""
filmsim v2 — photon-counting colour negative film simulation.

Pipeline (each stage is a physical step of the real process):

  scene light (RAW or sRGB)
    -> layer exposure     camera RGB -> red/green/blue-sensitive layers (spectral crosstalk matrix)
    -> irradiation        light scatters more the deeper it goes (blue top, red bottom)
    -> halation           a fraction of ALL light reflects off the film base back into mostly the red layer
    -> grains             each layer = fast (big, sensitive) + slow (small) emulsion; photons are
                          Poisson-sampled per grain; a grain develops if it catches >= k photons
    -> dye + DIR          developed grains form dye clouds; DIR couplers release inhibitor that diffuses
                          and suppresses development nearby, in the same layer (edge effects/acutance)
                          and in other layers (interlayer effect -> colour saturation)
    -> negative           cyan/magenta/yellow dyes with realistic unwanted absorptions, compensated by
                          coloured couplers (that's where the orange mask comes from)
    -> scan / print       enlarger/scanner blur, printer lights auto-balanced on mid-grey (like a lab),
                          print paper characteristic curve, then display.

Nothing here hand-draws a tone curve: curves and grain come from the grain physics, and the print
paper curve's contrast is solved so the whole system hits a target contrast.

PARAMETERS ARE PLAUSIBLE PLACEHOLDERS, NOT MEASURED FILM DATA. Use calibrate.py to fit them to
measured step-wedge densities of a real stock.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage, stats

RAW_EXTS = {"nef", "cr2", "cr3", "arw", "dng", "raf", "orf", "rw2", "pef", "srw", "3fr", "iiq"}


# ------------------------------------------------------------------ colour helpers
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# ------------------------------------------------------------------ input
def _load_with_pillow(path):
    """HEIC/HEIF/AVIF (iPhone) and JPEG: honours the rotation flag and converts the embedded colour
    profile (iPhones shoot Display P3) to sRGB, so colours aren't shifted. Returns sRGB float."""
    import io
    from PIL import Image, ImageOps, ImageCms
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        try:
            pillow_heif.register_avif_opener()
        except Exception:
            pass
    except ImportError:
        pass
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    if im.mode not in ("RGB", "L", "RGBA"):
        im = im.convert("RGB")
    if im.mode != "RGB":
        im = im.convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if "srgb" not in (ImageCms.getProfileDescription(src) or "").lower():
                im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    return np.asarray(im, np.float32) / 255.0


def load_image(path, exposure_stops=0.0, meter=None):
    """Return scene-linear RGB (float32). RAW files keep linear sensor data;
    JPEG/PNG are display-referred, so highlights above white are already lost."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in RAW_EXTS:
        import rawpy
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                                  use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
                                  highlight_mode=rawpy.HighlightMode.Blend)
        lin = rgb.astype(np.float32) / 65535.0
        meter = "average" if meter is None else meter
    elif ext in ("heic", "heif", "avif", "hif", "jpg", "jpeg"):
        img = _load_with_pillow(path)
        lin = srgb_to_linear(img).astype(np.float32)
    else:
        from skimage import io, img_as_float
        img = img_as_float(io.imread(path))
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        lin = srgb_to_linear(img[..., :3]).astype(np.float32)
    if meter == "average":   # averaging meter: geometric mean of luminance -> 18% grey
        lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lin *= 0.18 / np.exp(np.mean(np.log(np.clip(lum, 1e-4, None))))
    return lin * np.float32(2.0 ** exposure_stops)


# ------------------------------------------------------------------ film description
@dataclass
class Emulsion:
    """One emulsion sub-layer (real colour films stack a fast and a slow emulsion per colour)."""
    grains_per_px: int        # grains per output pixel, per side (fewer = coarser)
    median_sens: float        # relative grain sensitivity (bigger crystals catch more photons)
    sigma: float              # log-normal spread of grain sizes -> how much exposure range it covers
    dye_cloud: float          # dye cloud radius in grain units
    weight: float             # share of the layer's maximum density


@dataclass
class Layer:
    name: str
    emulsions: list
    dmax: float               # max dye density this layer can form
    scatter_px: float         # irradiation blur (deeper layers scatter more)


@dataclass
class FilmStock:
    name: str
    layers: list                          # [red-sensitive(cyan), green(magenta), blue(yellow)]
    k: int = 4                            # photons for a developable latent image
    photons_per_unit: float = 3.3         # photons per sens-1 grain for linear scene value 1.0
    spectral: np.ndarray = field(default_factory=lambda: np.array(
        [[0.90, 0.10, 0.00],              # rows: layer, cols: camera R,G,B   (placeholder)
         [0.08, 0.84, 0.08],
         [0.00, 0.12, 0.88]], np.float32))
    halation: tuple = (0.045, 0.010, 0.0) # fraction of light returned to each layer
    halation_radius: float = 0.012        # as fraction of image's short side
    dir_matrix: np.ndarray = field(default_factory=lambda: np.array(
        [[0.22, 0.08, 0.04],              # dye i is inhibited by development in layer j
         [0.08, 0.22, 0.08],
         [0.04, 0.08, 0.22]], np.float32))
    dir_radius: float = 1.5               # inhibitor diffusion, px
    base: tuple = (0.12, 0.16, 0.22)      # film base + fog, Status-M R,G,B density (pre-mask)
    # unwanted dye absorptions: rows scan channel R,G,B ; cols cyan, magenta, yellow dye
    dye_absorption: np.ndarray = field(default_factory=lambda: np.array(
        [[1.00, 0.10, 0.02],
         [0.28, 1.00, 0.10],
         [0.10, 0.38, 1.00]], np.float32))
    masked: tuple = (True, True, False)   # cyan & magenta use coloured (masking) couplers
    grain_correlation: float = 0.6        # shared grain structure between layers (MEASURABLE: cross-
                                          # correlate R,G,B grain in a scan of a flat exposed patch)
    seed: int = 0

    def copy(self, **kw):
        c = copy.deepcopy(self)
        for k_, v in kw.items():
            setattr(c, k_, v)
        return c


def make_stock(name="Generic 400 (placeholder)", grain_scale=1.0, seed=0):
    """grain_scale > 1 = fewer, bigger grains (a faster, grainier film)."""
    g = lambda n: max(1, int(round(n / grain_scale)))
    layers = []
    for nm, dmax, scat in [("red/cyan", 1.9, 0.9), ("green/magenta", 2.0, 0.6), ("blue/yellow", 2.1, 0.35)]:
        layers.append(Layer(nm, [
            Emulsion(grains_per_px=g(3), median_sens=8.0, sigma=1.3, dye_cloud=1.1, weight=0.35),
            Emulsion(grains_per_px=g(8), median_sens=1.0, sigma=1.5, dye_cloud=0.8, weight=0.65),
        ], dmax, scat))
    return FilmStock(name, layers, seed=seed)


# ------------------------------------------------------------------ analytic (noise-free) model
_Q = stats.norm.ppf((np.arange(256) + 0.5) / 256).astype(np.float64)   # quantiles for expectations


def layer_curve(layer: Layer, lam, k):
    """Expected developed fraction for mean photons-per-unit-sensitivity lam (array)."""
    lam = np.atleast_1d(np.asarray(lam, np.float64))
    out = np.zeros_like(lam)
    for em in layer.emulsions:
        s = em.median_sens * np.exp(em.sigma * _Q)
        out += em.weight * stats.poisson.sf(k - 1, lam[:, None] * s[None, :]).mean(1)
    return out


def expected_dye(stock: FilmStock, lam_layers):
    """Flat-field dye amounts (3,N) for layer exposures (3,N), including DIR inhibition."""
    dev = np.stack([layer_curve(L, lam_layers[i], stock.k) for i, L in enumerate(stock.layers)])
    return dev * np.clip(1 - stock.dir_matrix.astype(np.float64) @ dev, 0, 1)


def dye_to_density(stock: FilmStock, dye):
    """Negative Status-M densities (3,...) from dye amounts (3,...)."""
    dmax = np.array([L.dmax for L in stock.layers])
    A = stock.dye_absorption.astype(np.float64)
    D = np.tensordot(A * dmax[None, :], dye, axes=(1, 0))
    for d in range(3):                     # coloured couplers: remaining coupler carries the
        if stock.masked[d]:                # unwanted absorption, so it becomes a constant (orange)
            for c in range(3):
                if c != d:
                    D[c] = D[c] + A[c, d] * dmax[d] * (1 - dye[d])
    return D + np.array(stock.base).reshape(3, *([1] * (D.ndim - 1)))


def neutral_negative(stock: FilmStock, log2_stops):
    """Negative densities for a neutral grey patch at given stops relative to 18% grey."""
    e = 0.18 * 2.0 ** np.asarray(log2_stops, np.float64)
    lam = stock.spectral.sum(1)[:, None].astype(np.float64) * e[None, :] * stock.photons_per_unit
    return dye_to_density(stock, expected_dye(stock, lam))


# ------------------------------------------------------------------ stochastic exposure + development
def _develop(light, em: Emulsion, k, rng, tile_rows=64, pad=3, shared_seed=None, rho=0.0):
    """Develop one emulsion. Processed in row tiles so any image size fits in memory
    (a 24 MP frame at 8 grains/px per side is ~1.5 billion grains per emulsion)."""
    os_ = em.grains_per_px
    h, w = light.shape
    lp = np.pad(light, pad, mode="edge")
    out = np.empty((h, w), np.float32)
    ln_med, sig = np.float32(np.log(em.median_sens)), np.float32(em.sigma)
    # inter-layer correlation: overlying grains shadow/scatter light into the layers below, so
    # part of each grain-scale exposure fluctuation is shared between layers at the same spot.
    shared = np.random.default_rng(shared_seed) if (rho > 0 and shared_seed is not None) else None
    a, b = np.float32(np.sqrt(rho)), np.float32(np.sqrt(1 - rho))
    for y0 in range(0, h, tile_rows):
        y1 = min(h, y0 + tile_rows)
        L = ndimage.zoom(lp[y0:y1 + 2 * pad], os_, order=1, grid_mode=True, mode="nearest")
        L = L.astype(np.float32)
        z = rng.standard_normal(L.shape, dtype=np.float32)
        if shared is not None:
            z = a * shared.standard_normal(L.shape, dtype=np.float32) + b * z
        s = np.exp(ln_med + sig * z)
        del z
        dev = (rng.poisson(L * s) >= k).astype(np.float32)
        del L, s
        if em.dye_cloud > 0:
            dev = ndimage.gaussian_filter(dev, em.dye_cloud)
        dev = dev[pad * os_:(pad + y1 - y0) * os_, pad * os_:(pad + w) * os_]
        out[y0:y1] = dev.reshape(y1 - y0, os_, w, os_).mean((1, 3))
    return out


def expose_and_develop(stock: FilmStock, lin):
    """Scene-linear RGB (H,W,3) -> negative densities (3,H,W) and dye (3,H,W)."""
    rng = np.random.default_rng(stock.seed)
    h, w, _ = lin.shape
    E = np.einsum("lc,hwc->lhw", stock.spectral, np.clip(lin, 0, None)).astype(np.float32)
    # halation: linear in light — a fraction of the total transmitted light comes back blurred
    r = stock.halation_radius * min(h, w)
    glow = ndimage.gaussian_filter(E.sum(0) / 3, r)
    E = E + np.array(stock.halation, np.float32)[:, None, None] * glow[None]
    dev = np.zeros_like(E)
    for i, layer in enumerate(stock.layers):
        light = ndimage.gaussian_filter(E[i], layer.scatter_px) * stock.photons_per_unit
        for e, em in enumerate(layer.emulsions):
            dev[i] += em.weight * _develop(light, em, stock.k, rng,
                                           shared_seed=[stock.seed, 1000 + e], rho=stock.grain_correlation)
    inhibitor = np.stack([ndimage.gaussian_filter(dev[j], stock.dir_radius) for j in range(3)])
    dye = dev * np.clip(1 - np.einsum("ij,jhw->ihw", stock.dir_matrix, inhibitor), 0, 1)
    return dye_to_density(stock, dye), dye


# ------------------------------------------------------------------ print / scan stage
@dataclass
class PrintPaper:
    dmin: float = 0.08
    dmax: float = 2.25
    system_gamma: float = 1.4     # target overall contrast (scene -> print) at mid-grey
    blur_px: float = 0.5          # enlarger lens / scanner optics


def print_negative(stock: FilmStock, D, paper: PrintPaper = PrintPaper()):
    """Negative densities (3,H,W) -> display-linear RGB (H,W,3), lab-style auto grey balance."""
    # paper sees light transmitted through the negative
    logT = -np.stack([ndimage.gaussian_filter(D[c], paper.blur_px) for c in range(3)])
    # grey-balance and contrast from the noise-free neutral curve of this stock
    Dn = neutral_negative(stock, np.array([-0.25, 0.0, 0.25]))
    neg_gamma = (Dn[:, 2] - Dn[:, 0]) / (0.5 * np.log10(2))           # density per log10 exposure
    rng_ = paper.dmax - paper.dmin
    target_D = paper.dmin - np.log10(0.18)                              # 18% reflectance
    p = (target_D - paper.dmin) / rng_
    a = paper.system_gamma / (neg_gamma * rng_ * p * (1 - p))           # sigmoid steepness per layer
    out = np.empty_like(logT)
    for c in range(3):
        # paper logH = logT + printer light; choose printer light so mid-grey -> target_D
        x = logT[c] - (-Dn[c, 1])                                       # 0 at mid-grey
        Dp = paper.dmin + rng_ / (1 + np.exp(-a[c] * x) * (1 / p - 1))
        out[c] = 10.0 ** -(Dp - paper.dmin)                             # relative to paper white
    return np.moveaxis(out, 0, -1)


def scan_negative_raw(D):
    """What an un-inverted scan of the negative looks like (orange mask and all)."""
    return np.moveaxis(10.0 ** -D, 0, -1) * 2.0


def simulate(stock: FilmStock, lin, paper: PrintPaper = PrintPaper()):
    D, dye = expose_and_develop(stock, lin)
    return linear_to_srgb(print_negative(stock, D, paper)), D
