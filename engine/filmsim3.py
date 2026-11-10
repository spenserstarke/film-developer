"""
filmsim v3 — physically based grain on top of published film data.

What's new vs v2
  * Grains are real objects: random positions (a Poisson/Boolean process, like Newson et al. 2017),
    log-normal crystal sizes in MICRONS, sensitivity proportional to crystal area, and dye clouds
    whose size and dye amount scale with the crystal. Grain is defined on the film, not on pixels,
    so it looks right at any output size and changes correctly with format (35mm vs 120).
  * Density is measured the way a scanner/densitometer sees it: dye density is computed on a fine
    sub-pixel grid, converted to transmittance, THEN averaged into pixels. (Averaging density
    instead is a common simulation shortcut and makes grain look too smooth and too "noisy".)
  * Three emulsion sub-layers per colour (fast / mid / slow), matching how real stocks are built.
  * Stocks can be built from published data (spectral sensitivities, dye spectra, base density,
    characteristic curves) — loaded from spektrafilm profiles (CC BY-SA 4.0, Andrea Volpato,
    https://github.com/andreavolpato/spektrafilm), digitised from Kodak/Fuji datasheets.
  * A virtual microdensitometer (48 µm aperture, like datasheets use) measures RMS granularity, and
    grain count is fitted so the simulated film hits a published RMS granularity number.
"""
from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage, stats
from scipy.optimize import least_squares

from filmsim import srgb_to_linear, linear_to_srgb, load_image  # noqa: F401  (re-exported)

LOG2 = np.log10(2.0)


# ====================================================================== description
@dataclass
class SubLayer:
    median_sens: float        # relative sensitivity of the median crystal
    sigma: float              # total log-normal spread of sensitivity (ln units)
    amp: float                # density this sub-layer adds when fully developed
    r_um: float = 0.4         # median crystal radius, µm
    size_share: float = 0.5   # fraction of sensitivity variance due to crystal size (rest: efficiency)
    cloud_um: float = 1.5     # dye-cloud sigma for the median crystal, µm
    n_per_um2: float = 0.1    # crystals per µm² (set by granularity fit)

    @property
    def sig_size(self):       # spread of ln(area); capped: real emulsions have ln-radius spread ~0.2-0.35
        return min(self.sigma * np.sqrt(self.size_share), 0.7)

    @property
    def sig_eff(self):        # remaining spread = crystal-to-crystal efficiency (sensitization, defects)
        return np.sqrt(max(self.sigma ** 2 - self.sig_size ** 2, 1e-6))


@dataclass
class Stock:
    name: str
    sub: list                       # sub[layer][j] -> SubLayer;  layers = R(cyan), G(magenta), B(yellow)
    spectral: np.ndarray            # 3x3: linear sRGB -> layer exposure (grey -> 1,1,1)
    dye_absorption: np.ndarray      # rows scan channel R,G,B ; cols cyan, magenta, yellow
    base: np.ndarray                # Status-M-ish base+fog density R,G,B
    k: int = 4                      # photons per developable latent image
    photons_per_unit: float = 3.0   # photons at sens 1 for exposure 1 (mid-grey = 0.184)
    masked: tuple = (True, True, False)
    dir_matrix: np.ndarray = field(default_factory=lambda: np.array(
        [[0.10, 0.05, 0.02], [0.05, 0.10, 0.05], [0.02, 0.05, 0.10]]))
    dir_um: float = 12.0            # inhibitor diffusion distance
    scatter_um: tuple = (6.0, 4.0, 2.5)          # irradiation per layer (red layer is deepest)
    halation: tuple = (0.02, 0.004, 0.0)         # fraction of light returned off the base
    halation_um: float = 180.0
    grain_correlation: float = 0.5  # shared exposure fluctuation between layers (light shadowing)
    seed: int = 0
    source: str = ""
    reversal: bool = False          # slide film: dye forms where crystals were NOT exposed
    uniformity: float = 0.6         # crystals are packed more evenly than random (0 = Poisson)
    spectral_data: dict = None      # wavelengths, dye spectra, base spectrum (for spectral print/view)
    paper: dict = None              # print paper data (negatives)

    def copy(self):
        return copy.deepcopy(self)


# ====================================================================== analytic (noise-free) model
_Z = stats.norm.ppf((np.arange(24) + 0.5) / 24)


def sub_curve(sl: SubLayer, lam, k, reversal=False):
    """Expected density of one sub-layer vs mean photons/unit-sensitivity lam. Dye per grain is
    proportional to crystal area, so bigger (more sensitive) crystals count for more."""
    lam = np.atleast_1d(np.asarray(lam, float))
    zs, ze = np.meshgrid(_Z, _Z, indexing="ij")
    a = np.exp(sl.sig_size * zs - 0.5 * sl.sig_size ** 2)      # area relative to mean (E[a]=1)
    s = sl.median_sens * np.exp(sl.sig_size * zs + sl.sig_eff * ze)
    p = stats.poisson.sf(k - 1, lam[:, None, None] * s[None])
    if reversal:   # first developer turns exposed crystals to silver (bleached away); the rest form dye
        p = 1 - p
    return sl.amp * (p * a[None]).mean((1, 2))


def layer_density(st: Stock, i, lam):
    return sum(sub_curve(sl, lam, st.k, st.reversal) for sl in st.sub[i])


def layer_dmax(st: Stock, i):
    return sum(sl.amp for sl in st.sub[i])


def dye_to_density(st: Stock, dye):
    dmax = np.array([layer_dmax(st, i) for i in range(3)])
    A = st.dye_absorption
    D = np.tensordot(A * dmax[None, :], dye, axes=(1, 0))
    for d in range(3):
        if st.masked[d]:
            for c in range(3):
                if c != d:
                    D[c] = D[c] + A[c, d] * dmax[d] * (1 - dye[d])
    return D + st.base.reshape(3, *([1] * (D.ndim - 1)))


def neutral_negative(st: Stock, stops):
    """Channel densities (3,N) of a neutral patch, stops relative to mid-grey (0.184)."""
    e = 0.184 * 2.0 ** np.asarray(stops, float)
    dmax = np.array([layer_dmax(st, i) for i in range(3)])
    f = np.stack([layer_density(st, i, st.spectral[i].sum() * e * st.photons_per_unit)
                  for i in range(3)]) / dmax[:, None]
    return dye_to_density(st, f * np.clip(1 - st.dir_matrix @ f, 0, 1))


# ====================================================================== build from published data
def _status_m_like(wl, kind="status_M"):
    """Approximate densitometer responses: Status M (negatives, ~650/545/445 nm) or
    Status A (slides/prints, ~615/540/440 nm)."""
    g = lambda c, w: np.exp(-0.5 * ((wl - c) / w) ** 2)
    if str(kind).lower() == "status_a":
        return np.stack([g(615, 11), g(540, 14), g(440, 11)])
    return np.stack([g(650, 14), g(545, 18), g(445, 14)])


def from_spektrafilm(path, fit=True, verbose=True):
    import colour
    raw = json.load(open(path))
    d = raw["data"]
    F = lambda a: np.array(a, dtype=float)
    wl = F(d["wavelengths"])
    sens = np.nan_to_num(10 ** F(d["log_sensitivity"]))            # (81,3) layers R,G,B
    dyes = np.nan_to_num(F(d["channel_density"]))                   # (81,3) cyan, magenta, yellow
    base_sp = F(d["base_density"]); ok = ~np.isnan(base_sp)
    base_sp = np.interp(wl, wl[ok], base_sp[ok])

    # camera (linear sRGB) -> film layers, via smooth spectral basis under the film's illuminant
    basis = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(
        colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])).values          # (81,3)
    shape = colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])
    ill_name = raw["info"].get("reference_illuminant", "D55")
    if ill_name in colour.SDS_ILLUMINANTS:
        illum = colour.SDS_ILLUMINANTS[ill_name].copy().align(shape).values
    else:   # "T" = tungsten-balanced (3200 K) stocks
        illum = colour.sd_blackbody(3200, shape).values
    M = np.einsum("wl,w,wc->lc", sens, illum, basis)
    M /= M.sum(1, keepdims=True)                                     # grey -> equal exposure (lab balance)

    resp = _status_m_like(wl, raw["info"].get("densitometer", "status_M"))
    resp /= resp.sum(1, keepdims=True)
    dens = resp @ dyes                                               # [channel, dye]
    A = dens / np.diag(dens)[None, :]
    base = resp @ base_sp

    own = np.array([resp[k_] @ dyes[:, k_] for k_ in range(3)])
    st_spec = dict(wl=wl, spec=(dyes / np.maximum(own, 1e-6)).T, base=base_sp,
                   mid_ref=np.nan_to_num(F(d.get("midscale_neutral_density", [np.nan] * len(wl)))))
    model = d["density_curves_model"]
    C, Amp, S = F(model["centers"]), F(model["amplitudes"]), F(model["sigmas"])   # [layer, sub]
    st0 = Stock(raw["info"]["name"], [], M, A, base, source=raw["metadata"]["license"])
    st0.spectral_data = st_spec
    if raw["info"].get("type") == "positive":
        st0.reversal = True
        st0.masked = (False, False, False)        # slide films have no orange mask
        st0.dir_matrix = np.zeros((3, 3))         # (interimage effects not modelled for reversal yet)
    ppu, k = st0.photons_per_unit, st0.k
    for i in range(3):
        subs = []
        for j in np.argsort(C[i]):              # fastest (lowest centre) first
            # a sub-layer whose 50% point sits at log10 exposure C needs  lam*s_med ≈ k - 0.33
            s_med = (k - 0.33) / (0.184 * 10 ** C[i, j] * ppu)
            subs.append(SubLayer(median_sens=s_med, sigma=max(0.3, S[i, j] * np.log(10) * 0.9),
                                 amp=Amp[i, j]))
        st0.sub.append(subs)
    _assign_physical_sizes(st0)
    if fit:
        le, dc = F(d["log_exposure"]), F(d["density_curves"])
        sel = (le > -2.6) & (le < 3.4)
        sel &= np.arange(len(le)) % 2 == 0        # every other point is plenty
        target = dc[sel] + base[None, :]          # datasheet curves are over D-min
        st0 = fit_curves(st0, le[sel] / LOG2, target, verbose=verbose)
        st0.profile = raw["info"]
    if not st0.reversal:
        st0.paper = load_paper(path, raw["info"].get("target_print"))
    return st0


def _assign_physical_sizes(st: Stock, r_slow=0.30, r_max=1.1):
    """Crystal size follows sensitivity (area ∝ sensitivity, tempered); clouds grow with crystals."""
    smin = min(sl.median_sens for L in st.sub for sl in L)
    for L in st.sub:
        for sl in L:
            sl.r_um = min(r_max, r_slow * (sl.median_sens / smin) ** 0.25)
            sl.cloud_um = 1.5 + 3.0 * sl.r_um     # dye clouds ~5-13 µm across
    set_counts(st, getattr(st, "n_scale", 0.02))


def set_counts(st: Stock, n_scale):
    """Crystals per µm²: equal silver per unit density. Modern stocks use thin tabular crystals
    (Kodak T-Grain / Fuji Sigma, ~0.12 µm thick), so silver per crystal ∝ area: n ∝ amp / r²."""
    st.n_scale = n_scale
    for L in st.sub:
        for sl in L:
            sl.n_per_um2 = n_scale * sl.amp / sl.r_um ** 2


def silver_volume(st: Stock, thickness_um=0.12):
    """Silver halide volume per µm² of film implied by the crystal counts (real films: ~1-2)."""
    return sum(sl.n_per_um2 * np.pi * sl.r_um ** 2 * thickness_um for L in st.sub for sl in L)


def _pack(st):
    return np.array([[np.log(sl.median_sens), sl.sigma, sl.amp] for L in st.sub for sl in L]).ravel()


def _unpack(st, v):
    s = st.copy()
    v = v.reshape(-1, 3)
    for n, sl in enumerate([sl for L in s.sub for sl in L]):
        sl.median_sens, sl.sigma, sl.amp = np.exp(v[n, 0]), v[n, 1], v[n, 2]
    return s


def fit_curves(st: Stock, stops, D_target, verbose=True):
    """Fit sub-layer sensitivities/spreads/amplitudes so the grain physics reproduces measured
    (or datasheet) channel density curves. D_target: (N,3) including base."""
    x0 = _pack(st)
    n = len(x0) // 3
    lo = np.tile([-12, 0.2, 0.01], n); hi = np.tile([8, 4.0, 2.5], n)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    res = least_squares(lambda v: (neutral_negative(_unpack(st, v), stops).T - D_target).ravel(),
                        x0, bounds=(lo, hi), x_scale="jac", max_nfev=80, ftol=1e-6, xtol=1e-6)
    out = _unpack(st, res.x)
    _assign_physical_sizes(out)
    rms = np.sqrt(np.mean((neutral_negative(out, stops).T - D_target) ** 2))
    if verbose:
        print(f"[fit] {st.name}: curve RMS error {rms:.4f} D over {len(stops)} points")
    out.fit_rms = rms
    return out


# ====================================================================== stochastic grain engine
class Frame:
    """Maps an output image onto a physical piece of film."""
    def __init__(self, h, w, width_mm=36.0):
        self.h, self.w = h, w
        self.px_um = width_mm * 1000.0 / w


def _layer_light(st: Stock, lin, frame: Frame):
    E = rgb_to_layers(st, lin) * np.float32(2.0 ** getattr(st, "exposure_offset", 0.0))
    glow = ndimage.gaussian_filter(E.mean(0), st.halation_um / frame.px_um)
    E = E + np.array(st.halation, np.float32)[:, None, None] * glow[None]
    for i in range(3):
        E[i] = ndimage.gaussian_filter(E[i], st.scatter_um[i] / frame.px_um)
    return E * np.float32(st.photons_per_unit)


def _strip_grains(st, i, j, sl, strip, light, frame, strip_rows):
    """EXPLICIT engine: every crystal in a strip (deterministic per strip). Only feasible for
    small patches / low counts; used to validate the cell engine."""
    h, w, px = frame.h, frame.w, frame.px_um
    y0 = strip * strip_rows
    rows = min(strip_rows, h - y0)
    if rows <= 0:
        return None
    rng = np.random.default_rng([st.seed, i, j, strip])
    n = rng.poisson(sl.n_per_um2 * rows * w * px * px)
    if n == 0:
        return None
    k = int(round(n * st.grain_correlation))
    gy = np.random.default_rng([st.seed, 99, j, strip, 0])
    gx = np.random.default_rng([st.seed, 99, j, strip, 1])
    ys = np.concatenate([gy.random(k, dtype=np.float32), rng.random(n - k, dtype=np.float32)]) * rows + y0
    xs = np.concatenate([gx.random(k, dtype=np.float32), rng.random(n - k, dtype=np.float32)]) * w
    zs = rng.standard_normal(n, dtype=np.float32)
    ze = rng.standard_normal(n, dtype=np.float32)
    a = np.exp(np.float32(sl.sig_size) * zs - np.float32(0.5 * sl.sig_size ** 2))
    s = np.float32(sl.median_sens) * np.exp(np.float32(sl.sig_size) * zs + np.float32(sl.sig_eff) * ze)
    lam = ndimage.map_coordinates(light, [ys - 0.5, xs - 0.5], order=1, mode="nearest")
    dev = rng.poisson(lam * s) >= st.k
    return ys[dev], xs[dev], a[dev]


def develop_layer_explicit(st: Stock, i, light, frame: Frame, band_rows=48, strip_rows=8, n_bins=4):
    """Crystal-by-crystal development (exact, slow). Density measured through transmittance."""
    h, w, px = frame.h, frame.w, frame.px_um
    subs = st.sub[i]
    cloud_max = max(sl.cloud_um * np.exp(sl.sig_size) for sl in subs)
    ss = int(np.clip(np.ceil(px / (0.9 * min(sl.cloud_um for sl in subs))), 1, 6))
    sub_um = px / ss
    margin = int(np.ceil((4 * cloud_max / px + 1) / strip_rows) * strip_rows)
    out = np.empty((h, w), np.float32)
    for b0 in range(0, h, band_rows):
        b1 = min(h, b0 + band_rows)
        g0, g1 = b0 - margin, b1 + margin
        H = (g1 - g0) * ss
        Dsub = np.zeros((H, w * ss), np.float32)
        for j, sl in enumerate(subs):
            c = sl.amp / sl.n_per_um2
            got = [r for r in (_strip_grains(st, i, j, sl, s_, light, frame, strip_rows)
                               for s_ in range(max(0, g0 // strip_rows), (min(h, g1) + strip_rows - 1) // strip_rows))
                   if r is not None]
            if not got:
                continue
            ys, xs, aa = (np.concatenate(z) for z in zip(*got))
            if ys.size == 0:
                continue
            yi = np.clip(((ys - g0) * ss).astype(np.int64), 0, H - 1)
            xi = np.clip((xs * ss).astype(np.int64), 0, w * ss - 1)
            edges = np.quantile(aa, np.linspace(0, 1, n_bins + 1)) if aa.size > n_bins else [0, np.inf]
            bins = np.clip(np.searchsorted(edges, aa, side="right") - 1, 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                sel = bins == b
                if not sel.any():
                    continue
                acc = np.bincount(yi[sel] * (w * ss) + xi[sel], weights=aa[sel] * c,
                                  minlength=H * w * ss).reshape(H, w * ss).astype(np.float32)
                acc /= sub_um * sub_um
                sig = np.sqrt(max((sl.cloud_um ** 2) * np.mean(aa[sel]) - sub_um ** 2 / 12, 0)) / sub_um
                Dsub += ndimage.gaussian_filter(acc, sig, mode="constant")
        T = (10.0 ** -Dsub).reshape(H // ss, ss, w, ss).mean((1, 3))
        out[b0:b1] = -np.log10(T[margin:margin + (b1 - b0)])
    return out


# ---------------------------------------------------------------------- fast cell engine
_TABLES = {}


def _bin_tables(sl: SubLayer, k, n_bins=3, nz=8, reversal=False):
    """For each crystal-size bin: P(forms dye | lambda) and area stats of those crystals."""
    key = (sl.median_sens, sl.sigma, sl.size_share, k, n_bins, reversal)
    if key in _TABLES:
        return _TABLES[key]
    lg = np.linspace(-6, 6, 481)
    lam = 10 ** lg
    out = []
    for b in range(n_bins):
        zs = stats.norm.ppf((b + (np.arange(nz) + 0.5) / nz) / n_bins)
        a = np.exp(sl.sig_size * zs - 0.5 * sl.sig_size ** 2)
        s = sl.median_sens * np.exp(sl.sig_size * zs[:, None] + sl.sig_eff * _Z[None, :])
        p = stats.poisson.sf(k - 1, lam[:, None, None] * s[None]).mean(2)        # (L, nz)
        if reversal:
            p = 1 - p
        P = p.mean(1)
        Pn = np.maximum(P, 1e-300)
        a_dev = (p * a[None]).mean(1) / Pn
        a2_dev = (p * a[None] ** 2).mean(1) / Pn
        a_dev[P < 1e-300] = a.mean(); a2_dev[P < 1e-300] = (a ** 2).mean()
        out.append((lg, P.astype(np.float32), a_dev.astype(np.float32),
                    np.sqrt(np.maximum(a2_dev - a_dev ** 2, 0)).astype(np.float32), float(a.mean())))
    _TABLES[key] = out
    return out


PROGRESS = None        # optional callable(fraction 0..1) for the development stage
_band_tick = None      # set by develop(): called once per finished band of rows


def develop_layer(st: Stock, i, light, frame: Frame, band_rows=64, strip_rows=8, n_bins=3):
    """Cell engine. Developed crystals of the crystal field are sampled per small cell: count and
    total area of dye-forming crystals of each size class, with the exact mean and variance of
    the underlying crystal process (Poisson thinning, plus sub-Poisson packing via `uniformity`).
    Cost scales with pixels, not crystals."""
    h, w, px = frame.h, frame.w, frame.px_um
    subs = st.sub[i]
    min_cloud = min(sl.cloud_um for sl in subs)
    # sub-pixel cells only matter when dye clouds are a sizeable fraction of a pixel
    ss = 1 if min_cloud < 0.3 * px else int(np.clip(np.ceil(px / min_cloud), 1, 3))
    cell = px / ss
    cloud_max = max(sl.cloud_um * np.exp(sl.sig_size) for sl in subs)
    margin = int(np.ceil((4 * cloud_max / px + 1) / strip_rows) * strip_rows)
    rho = np.float32(st.grain_correlation)
    a_, b_ = np.sqrt(rho), np.sqrt(1 - rho)
    u = np.float32(st.uniformity)
    lp = np.pad(light, ((margin + 1, margin + 1), (1, 1)), mode="edge")
    tables = [_bin_tables(sl, st.k, n_bins, reversal=st.reversal) for sl in subs]
    lg0, dlg, nl = tables[0][0][0][0], tables[0][0][0][1] - tables[0][0][0][0], len(tables[0][0][0])
    out = np.empty((h, w), np.float32)
    for b0 in range(0, h, band_rows):
        b1 = min(h, b0 + band_rows)
        g0, g1 = b0 - margin, b1 + margin
        Lb = lp[g0 + margin:g1 + margin + 2]
        if ss > 1:
            Lb = ndimage.zoom(Lb, ss, order=1, grid_mode=True, mode="nearest")[ss:-ss, ss:-ss]
        else:
            Lb = Lb[1:-1, 1:-1]
        x = (np.log10(np.maximum(Lb, 1e-30)) - lg0) / dlg
        x = np.clip(x, 0, nl - 1.001).astype(np.float32)
        idx = x.astype(np.int32); fr = x - idx
        look = lambda T: T[idx] * (1 - fr) + T[idx + 1] * fr      # fast table lookup
        Dsub = np.zeros(Lb.shape, np.float32)
        strips = range(g0 // strip_rows, -(-g1 // strip_rows))
        rows_c = strip_rows * ss
        for j, sl in enumerate(subs):
            c = sl.amp / sl.n_per_um2
            n_cell = np.float32(sl.n_per_um2 * cell * cell / n_bins)
            for bi, (lg, P, adev, asd, amean) in enumerate(tables[j]):
                Pc = look(P)
                mu = n_cell * Pc
                ad = look(adev); sd = look(asd)
                dye = np.empty_like(mu)
                for t, s_ in enumerate(strips):
                    sl_ = slice(t * rows_c, (t + 1) * rows_c)
                    m = mu[sl_]
                    own = np.random.default_rng([st.seed, i, j, bi, s_ + 10_000])
                    shr = np.random.default_rng([st.seed, 99, j, bi, s_ + 10_000])
                    z = a_ * shr.standard_normal(m.shape, dtype=np.float32) + \
                        b_ * own.standard_normal(m.shape, dtype=np.float32)
                    A_, S_ = ad[sl_], sd[sl_]
                    # count variance (1 - u*p)*mean: evenly packed crystals fluctuate less than random
                    var = m * ((1 - u * Pc[sl_]) * A_ * A_ + S_ * S_)
                    tot = np.maximum(m * A_ + np.sqrt(var) * z, 0)
                    small = m < 12
                    if small.any():
                        ms = m[small]
                        Ns = own.poisson(ms).astype(np.float32)
                        Ns = np.maximum(ms + (Ns - ms) * np.sqrt(1 - u * Pc[sl_][small]), 0)   # same packing
                        tot[small] = np.maximum(Ns * A_[small] + np.sqrt(Ns) * S_[small] *
                                                own.standard_normal(Ns.shape, dtype=np.float32), 0)
                    dye[sl_] = tot
                acc = dye * np.float32(c / (cell * cell))
                sig = np.sqrt(max(sl.cloud_um ** 2 * amean - cell ** 2 / 12, 0)) / cell
                Dsub += ndimage.gaussian_filter(acc, sig, mode="nearest") if sig > 0.15 else acc
        if ss > 1:
            T = (10.0 ** -Dsub).reshape(-1, ss, w, ss).mean((1, 3))
            out[b0:b1] = -np.log10(T[margin:margin + (b1 - b0)])
        else:
            out[b0:b1] = Dsub[margin:margin + (b1 - b0)]
        if _band_tick is not None:
            _band_tick()
    return out


def develop(st: Stock, lin, width_mm=36.0, engine="cell", return_dye=False):
    """Scene-linear RGB -> negative/slide channel densities (3,H,W) [and dye amounts]."""
    from concurrent.futures import ThreadPoolExecutor
    h, w, _ = lin.shape
    frame = Frame(h, w, width_mm)
    light = _layer_light(st, lin, frame)
    dev = develop_layer if engine == "cell" else develop_layer_explicit
    global _band_tick
    if PROGRESS is not None and engine == "cell":
        import threading
        total, done, lk = 3 * (-(-h // 64)), [0], threading.Lock()

        def _tick():
            with lk:
                done[0] += 1
                PROGRESS(min(done[0] / total, 1.0))
        _band_tick = _tick
    with ThreadPoolExecutor(3) as ex:
        f = np.stack(list(ex.map(lambda i: dev(st, i, light[i], frame) / layer_dmax(st, i), range(3))))
    _band_tick = None
    inh = np.stack([ndimage.gaussian_filter(f[j], st.dir_um / frame.px_um) for j in range(3)])
    dye = f * np.clip(1 - np.einsum("ij,jhw->ihw", st.dir_matrix, inh), 0, 1)
    D = dye_to_density(st, dye)
    return (D, dye) if return_dye else D


# ====================================================================== measurement: RMS granularity
def measure_rms(st: Stock, density_over_base=1.0, aperture_um=48.0, patch_mm=1.6, px_um=4.0,
                channels=(0, 1, 2), engine="cell"):
    """Virtual microdensitometer: expose a flat neutral patch to the given density, scan it at
    px_um, average transmittance over a circular aperture, report 1000 x std(density)."""
    s = np.linspace(-4, 7, 221)
    Dn = neutral_negative(st, s)
    if st.reversal:                                             # density falls with exposure
        stop = np.interp(density_over_base, (Dn[1] - Dn[1, -1])[::-1], s[::-1])
    else:
        stop = np.interp(density_over_base, Dn[1] - Dn[1, 0], s)   # "1.0 above minimum density"
    n = int(patch_mm * 1000 / px_um)
    stop -= getattr(st, "exposure_offset", 0.0)          # develop() re-applies the stock's offset
    lin = np.full((n, n, 3), 0.184 * 2 ** stop, np.float32)
    D = develop(st, lin, width_mm=patch_mm, engine=engine)
    r = aperture_um / 2 / px_um
    yy, xx = np.mgrid[-int(r) - 1:int(r) + 2, -int(r) - 1:int(r) + 2]
    disk = ((yy ** 2 + xx ** 2) <= r * r).astype(float); disk /= disk.sum()
    out = []
    for c in channels:
        T = ndimage.convolve(10.0 ** -D[c], disk, mode="reflect")
        m = int(r) + 2
        out.append(1000 * np.std(-np.log10(T[m:-m, m:-m])))
    return np.array(out), stop


def fit_granularity(st: Stock, target_rms, channel=1, iters=12, tol=0.02, verbose=True):
    """Scale crystal counts until the simulated film hits a published RMS granularity.
    Works in log space with a secant step (grain falls roughly as 1/sqrt(count))."""
    st = st.copy()
    if not hasattr(st, "n_scale"):
        set_counts(st, 0.02)
    hist = []
    for it in range(iters):
        rms = measure_rms(st, channels=(channel,))[0][0]
        hist.append((np.log(st.n_scale), np.log(rms)))
        if verbose:
            print(f"[grain] iter {it}: n_scale {st.n_scale:.4g} -> rms {rms:.2f} (target {target_rms})")
        if abs(rms / target_rms - 1) < tol:
            break
        slope = -0.5
        if len(hist) >= 2 and abs(hist[-1][0] - hist[-2][0]) > 1e-6:
            slope = np.clip((hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0]), -2.0, -0.1)
        step = np.clip((np.log(target_rms) - np.log(rms)) / slope, -6, 6)
        set_counts(st, st.n_scale * np.exp(step))
    st.rms_measured = rms
    return st


# ====================================================================== print stage
def print_negative(st: Stock, D, system_gamma=1.4, dmin=0.08, dmax=2.25, blur_px=0.4):
    T = np.stack([ndimage.gaussian_filter(10.0 ** -D[c], blur_px) for c in range(3)])
    logT = np.log10(np.clip(T, 1e-6, None))
    Dn = neutral_negative(st, np.array([-0.25, 0.0, 0.25]))
    neg_gamma = (Dn[:, 2] - Dn[:, 0]) / (0.5 * LOG2)
    rng_ = dmax - dmin
    p = (-np.log10(0.18)) / rng_
    a = system_gamma / (neg_gamma * rng_ * p * (1 - p))
    out = np.empty_like(logT)
    for c in range(3):
        x = logT[c] + Dn[c, 1]
        Dp = dmin + rng_ / (1 + np.exp(-a[c] * x) * (1 / p - 1))
        out[c] = 10.0 ** -(Dp - dmin)
    return np.moveaxis(out, 0, -1)


# ====================================================================== spectral print / viewing
def load_paper(film_path, target=None):
    """Print paper data (spektrafilm profile). Uses the film's intended paper if it is a paper,
    otherwise Kodak Portra Endura (cine stocks target projection print film)."""
    import os
    folder = os.path.dirname(film_path)
    name = target if target and os.path.exists(os.path.join(folder, f"{target}.json")) else None
    if name:
        info = json.load(open(os.path.join(folder, f"{name}.json")))["info"]
        if info.get("support") != "paper":
            name = None
    name = name or "kodak_portra_endura"
    raw = json.load(open(os.path.join(folder, f"{name}.json")))
    d = raw["data"]
    F = lambda a: np.nan_to_num(np.array(a, dtype=float))
    dyes = F(d["channel_density"])
    return dict(name=raw["info"]["name"], wl=F(d["wavelengths"]),
                sens=np.nan_to_num(10 ** np.array(d["log_sensitivity"], dtype=float)),
                spec=(dyes / dyes.max(0, keepdims=True)).T, base=F(d["base_density"]),
                le=F(d["log_exposure"]), dc=F(d["density_curves"]))


def _viewing_matrix(wl, illuminant="D50"):
    """Spectrum (reflectance/transmittance) -> linear sRGB, eye adapted to the viewing light."""
    import colour
    shape = colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])
    cmf = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape).values
    E = colour.SDS_ILLUMINANTS[illuminant].copy().align(shape).values
    V = E[:, None] * cmf
    V = V / V[:, 1].sum()
    w_src = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][illuminant])
    w_dst = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(w_src, w_dst, "Bradford")
    M = colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB @ cat
    return V @ M.T                                                   # (wl, 3)


def _neg_spectra(st: Stock):
    """Per-unit-density dye spectra of the negative. Validated: base + dyes reproduces Kodak's
    published mid-scale neutral spectrum for Portra 400 to 0.012 D RMS (420-700 nm)."""
    return st.spectral_data["spec"]


def _midgrey_dye(st: Stock, stop=0.0):
    e = 0.184 * 2.0 ** stop
    dmax = np.array([layer_dmax(st, i) for i in range(3)])
    f = np.array([layer_density(st, i, st.spectral[i].sum() * e * st.photons_per_unit)[0]
                  for i in range(3)]) / dmax
    return f * np.clip(1 - st.dir_matrix @ f, 0, 1) * dmax


def _print_setup(st: Stock):
    if getattr(st, "_print_cache", None) is not None:
        return st._print_cache
    from scipy.optimize import least_squares as lsq
    import colour
    sd, pp = st.spectral_data, st.paper
    wl = sd["wl"]
    enl = colour.sd_blackbody(3200, colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])).values
    K = enl[:, None] * pp["sens"]
    K = K / K.sum(0, keepdims=True)
    S = _neg_spectra(st)
    V = _viewing_matrix(wl, "D50")
    white = (10.0 ** -pp["base"]) @ V

    def paper_rgb(X, offs):
        T = 10.0 ** -(sd["base"][None] + X @ S)
        logH = np.log10(np.maximum(T @ K, 1e-30)) + offs[None]
        Dp = np.stack([np.interp(logH[:, l], pp["le"], pp["dc"][:, l]) for l in range(3)], 1)
        R = 10.0 ** -(pp["base"][None] + Dp @ pp["spec"])
        return (R @ V) / white[None]

    xmid = _midgrey_dye(st)[None]
    # printer lights: solve the three filtration offsets that print mid-grey as neutral 18% grey
    mid_le = pp["le"][np.argmin(np.abs(pp["dc"] - 0.75), 0)]
    T0 = 10.0 ** -(sd["base"] + xmid @ S)
    x0 = mid_le - np.log10(T0 @ K)[0]
    r = lsq(lambda o: paper_rgb(xmid, o)[0] - 0.18, x0)
    st._print_cache = (paper_rgb, r.x)
    return st._print_cache


def print_spectral(st: Stock, dye, chunk=400_000):
    """Spectral enlarger print: negative dye spectra -> paper exposure -> paper dyes -> D50 viewing."""
    paper_rgb, offs = _print_setup(st)
    dmax = np.array([layer_dmax(st, i) for i in range(3)])
    h, w = dye.shape[1:]
    X = (dye * dmax[:, None, None]).reshape(3, -1).T
    out = np.empty_like(X)
    for c0 in range(0, len(X), chunk):
        out[c0:c0 + chunk] = paper_rgb(X[c0:c0 + chunk], offs)
    return np.clip(out, 0, None).reshape(h, w, 3)


def slide_balance(st: Stock):
    """Per-channel gains so a mid-grey scene renders as neutral 18% grey (what a slide scanner's
    balance, or the eye adapting on a light table, does). Returned as the 'white' divisor."""
    sd = st.spectral_data
    V = _viewing_matrix(sd["wl"], "D50")
    T = 10.0 ** -(sd["base"] + _midgrey_dye(st, getattr(st, "exposure_offset", 0.0)) @ sd["spec"])
    mid = T @ V
    clear = (10.0 ** -sd["base"]) @ V
    return mid * (clear[1] / mid[1])          # clear film -> white; mid-grey -> neutral


def view_slide_spectral(st: Stock, dye, chunk=400_000):
    """A slide on a D50 light table: dye spectra -> transmittance -> colour, grey-balanced."""
    sd = st.spectral_data
    V = _viewing_matrix(sd["wl"], "D50")
    white = slide_balance(st)
    dmax = np.array([layer_dmax(st, i) for i in range(3)])
    h, w = dye.shape[1:]
    X = (dye * dmax[:, None, None]).reshape(3, -1).T
    out = np.empty_like(X)
    for c0 in range(0, len(X), chunk):
        T = 10.0 ** -(sd["base"][None] + X[c0:c0 + chunk] @ sd["spec"])
        out[c0:c0 + chunk] = (T @ V) / white[None]
    return np.clip(out, 0, None).reshape(h, w, 3)


def view_slide(st: Stock, D, blur_px=0.4):
    """A slide is the final image: light passes through it. Clear film (D-min) shows as white."""
    dmin = neutral_negative(st, np.array([12.0]))[:, 0]
    T = np.stack([ndimage.gaussian_filter(10.0 ** -(D[c] - dmin[c]), blur_px) for c in range(3)])
    return np.moveaxis(T, 0, -1)


def simulate(st: Stock, lin, width_mm=36.0, system_gamma=1.4, spectral=True):
    """Returns (sRGB image, negative/slide channel densities)."""
    D, dye = develop(st, lin, width_mm, return_dye=True)
    if spectral and st.spectral_data is not None:
        dye = np.stack([ndimage.gaussian_filter(dye[c], 0.4) for c in range(3)])   # enlarger/scan optics
        lin_out = view_slide_spectral(st, dye) if st.reversal else print_spectral(st, dye)
    else:
        lin_out = view_slide(st, D) if st.reversal else print_negative(st, D, system_gamma)
    return linear_to_srgb(lin_out), D


# ====================================================================== spectral upsampling LUT
PROFILE_DIR = "ext/spektrafilm/src/spektrafilm/data/profiles"


def build_spectral_lut(st: Stock, n=33, profile_dir=PROFILE_DIR, uv_cut_nm=400.0):
    """Camera RGB -> film layer exposure via Jakob & Hanika (2019) spectral upsampling.
    Each scene chromaticity gets a smooth physically plausible reflectance spectrum, lit by the
    film's reference illuminant and integrated against the film's measured spectral sensitivities.
    Unlike a 3x3 matrix this captures how saturated colours actually expose each layer.
    Table T[r_idx, g_idx, layer] is exposure per unit (R+G+B)/3, normalised so grey -> 1."""
    import os, warnings, colour
    warnings.filterwarnings("ignore")
    raw = json.load(open(os.path.join(profile_dir, st.profile["stock"] + ".json")))
    d = raw["data"]
    wl = np.array(d["wavelengths"], float)
    shape = colour.SpectralShape(wl[0], wl[-1], wl[1] - wl[0])
    sens = np.nan_to_num(10 ** np.array(d["log_sensitivity"], dtype=float))
    ill_name = raw["info"].get("reference_illuminant", "D55")
    film_ill = (colour.SDS_ILLUMINANTS[ill_name].copy().align(shape).values
                if ill_name in colour.SDS_ILLUMINANTS else colour.sd_blackbody(3200, shape).values)
    # camera lens glass + coatings absorb ultraviolet: transmission rises from ~380 to ~410 nm
    lens = 1 / (1 + np.exp(-(wl - uv_cut_nm) / 6.0))
    W = (film_ill * lens)[:, None] * sens                             # (wl, layer)
    grey = W.sum(0)
    cs = colour.RGB_COLOURSPACES["sRGB"]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    d65 = colour.SDS_ILLUMINANTS["D65"].copy().align(shape)
    T = np.full((n, n, 3), np.nan)
    ax = np.linspace(0, 1, n)
    for a, r in enumerate(ax):
        for b, g in enumerate(ax):
            bl = 1 - r - g
            if bl < -1e-9:
                continue
            c = np.array([r, g, max(bl, 0.0)])
            k = 0.9 / c.max()
            try:
                sd = colour.recovery.XYZ_to_sd_Jakob2019(colour.RGB_to_XYZ(c * k, cs), cmfs=cmfs,
                                                         illuminant=d65)
                T[a, b] = (sd.values @ W) / grey * 3 / (k * c.sum())
            except Exception:
                pass
    # fill unreachable cells (outside the sRGB triangle) from the nearest valid one
    bad = np.isnan(T[..., 0])
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        T = T[tuple(idx)]
    st.spectral_lut = T.astype(np.float32)
    return st


def rgb_to_layers(st: Stock, lin):
    """Scene-linear RGB (H,W,3) -> layer exposures (3,H,W): spectral LUT if present, else matrix."""
    lin = np.clip(lin, 0, None)
    T = getattr(st, "spectral_lut", None)
    if T is None:
        return np.einsum("lc,hwc->lhw", st.spectral, lin).astype(np.float32)
    s = lin.sum(-1)
    n = T.shape[0] - 1
    r = np.where(s > 1e-9, lin[..., 0] / np.maximum(s, 1e-9), 1 / 3) * n
    g = np.where(s > 1e-9, lin[..., 1] / np.maximum(s, 1e-9), 1 / 3) * n
    E = np.stack([ndimage.map_coordinates(T[..., l], [r, g], order=1, mode="nearest") for l in range(3)])
    return (E * (s / 3)[None]).astype(np.float32)


def set_slide_exposure(st: Stock, mid_density=0.9):
    """Place mid-grey at a normal slide density above D-min (datasheet exposure conventions vary)."""
    s_ = np.linspace(-6, 6, 241)
    Dn = neutral_negative(st, s_)
    g = Dn[1] - Dn[1, -1]
    st.exposure_offset = float(np.interp(mid_density, g[::-1], s_[::-1]))
    st._print_cache = None
    return st


def calibrate_slide_neutrality(st: Stock, stops=np.arange(-2.5, 2.51, 0.5)):
    """Datasheet slide curves are measured on a neutral grey scale, so every step of that scale
    must view as neutral. Fit one scale per dye spectrum (corrects the approximate densitometer
    responses) so the whole grey scale is neutral under D50 viewing, not just mid-grey."""
    from scipy.optimize import least_squares as lsq
    sd = st.spectral_data
    base_spec = sd["spec"].copy()
    V = _viewing_matrix(sd["wl"], "D50")
    off = getattr(st, "exposure_offset", 0.0)
    X = np.stack([_midgrey_dye(st, s + off) for s in stops])               # (N,3) density units

    def resid(lk2):   # green dye fixed (keeps datasheet contrast); red/blue scaled relative to it
        lk = np.array([lk2[0], 0.0, lk2[1]])
        S = base_spec * np.exp(lk)[:, None]
        rgb = np.maximum((10.0 ** -(sd["base"][None] + X @ S)) @ V, 1e-4)
        return np.concatenate([np.log(rgb[:, 0] / rgb[:, 1]), np.log(rgb[:, 2] / rgb[:, 1])])

    before = np.abs(resid(np.zeros(2))).max()
    r = lsq(resid, np.zeros(2), bounds=(-2, 2))
    sd["spec"] = base_spec * np.exp(np.array([r.x[0], 0.0, r.x[1]]))[:, None]
    st.neutrality_error = (before, np.abs(r.fun).max())
    st._print_cache = None
    return st
