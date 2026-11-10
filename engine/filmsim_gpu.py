"""
filmsim GPU engine (PyTorch). Runs on Apple Silicon GPUs via Metal ("mps"), NVIDIA ("cuda"),
or CPU. Same physics as filmsim3.develop_layer + spectral print; validated against it.

    import filmsim_gpu as g
    srgb = g.simulate(stock, lin, width_mm=36)
"""
import numpy as np
import torch
import torch.nn.functional as F
import filmsim3 as f3


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = pick_device()
T32 = dict(dtype=torch.float32)


def _t(a, dev):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)


def gblur(x, sigma):
    """Separable Gaussian blur of a (H,W) or (C,H,W) tensor, edge-replicated."""
    if sigma < 0.15:
        return x
    r = int(np.ceil(3.5 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, **T32) / sigma) ** 2)
    k = k / k.sum()
    squeeze = x.dim() == 2
    y = x[None, None] if squeeze else x[:, None]
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y[0, 0] if squeeze else y[:, 0]


def poisson_small(m, gen):
    """Poisson samples for small means by inverse CDF (works on every backend, incl. Metal)."""
    u = torch.rand(m.shape, generator=gen, device=m.device, **T32)
    p = torch.exp(-m)
    cdf = p.clone()
    n = torch.zeros_like(m)
    for k in range(1, 40):
        n += (u > cdf).float()
        p = p * m / k
        cdf = cdf + p
    return n


def _gen(dev, seed):
    try:
        g = torch.Generator(device=dev)
    except RuntimeError:
        g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def rgb_to_layers(st, lin, dev):
    """Camera RGB -> film layer exposure (spectral LUT via grid_sample, or matrix fallback)."""
    x = _t(np.clip(lin, 0, None), dev).permute(2, 0, 1)
    T = getattr(st, "spectral_lut", None)
    if T is None:
        return torch.einsum("lc,chw->lhw", _t(st.spectral, dev), x)
    s = x.sum(0)
    ok = s > 1e-9
    r = torch.where(ok, x[0] / s.clamp_min(1e-9), torch.full_like(s, 1 / 3))
    g = torch.where(ok, x[1] / s.clamp_min(1e-9), torch.full_like(s, 1 / 3))
    tab = _t(T, dev).permute(2, 0, 1)[None]                         # (1, 3, n_r, n_g)
    grid = torch.stack([g * 2 - 1, r * 2 - 1], -1)[None]            # x -> g axis, y -> r axis
    E = F.grid_sample(tab, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    return E * (s / 3)[None]


def develop(st, lin, width_mm=36.0, dev=DEV, n_bins=3):
    """Scene-linear RGB (H,W,3 numpy) -> dye amounts (3,H,W) tensor on dev."""
    h, w, _ = lin.shape
    frame = f3.Frame(h, w, width_mm)
    px = frame.px_um
    light = rgb_to_layers(st, lin, dev) * float(2.0 ** getattr(st, "exposure_offset", 0.0))
    glow = gblur(light.mean(0), st.halation_um / px)
    light = light + _t(np.array(st.halation), dev)[:, None, None] * glow[None]
    light = torch.stack([gblur(light[i], st.scatter_um[i] / px) for i in range(3)]) * st.photons_per_unit

    rho, u = float(st.grain_correlation), float(st.uniformity)
    a_, b_ = rho ** 0.5, (1 - rho) ** 0.5
    min_cloud = min(sl.cloud_um for L in st.sub for sl in L)
    ss = 1 if min_cloud < 0.3 * px else int(np.clip(np.ceil(px / min_cloud), 1, 3))
    cell = px / ss
    L = light if ss == 1 else F.interpolate(light[None], scale_factor=ss, mode="bilinear",
                                            align_corners=False)[0]
    g_shared = _gen(dev, st.seed * 7919 + 1)
    g_own = _gen(dev, st.seed * 7919 + 2)
    D = torch.zeros_like(L)
    nsub = len(st.sub[0])
    for j in range(nsub):
        for bi in range(n_bins):
            if f3.PROGRESS is not None:
                f3.PROGRESS((j * n_bins + bi) / (nsub * n_bins))
            z_sh = torch.randn(L.shape[1:], generator=g_shared, device=dev, **T32)
            for i in range(3):
                sl = st.sub[i][j]
                lg, P, adev, asd, amean = f3._bin_tables(sl, st.k, n_bins, reversal=st.reversal)[bi]
                lg0, dlg, nl = lg[0], lg[1] - lg[0], len(lg)
                x = ((torch.log10(L[i].clamp_min(1e-30)) - lg0) / dlg).clamp(0, nl - 1.001)
                idx = x.long(); fr = x - idx
                look = lambda T: (lambda tt: tt[idx] * (1 - fr) + tt[idx + 1] * fr)(_t(T, dev))
                Pc = look(P); A = look(adev); S = look(asd)
                m = sl.n_per_um2 * cell * cell / n_bins * Pc
                z = a_ * z_sh + b_ * torch.randn(m.shape, generator=g_own, device=dev, **T32)
                var = m * ((1 - u * Pc) * A * A + S * S)
                tot = (m * A + var.sqrt() * z).clamp_min(0)
                small = m < 12
                Ns = poisson_small(m.clamp_max(12), g_own)
                Ns = (m + (Ns - m) * (1 - u * Pc).clamp_min(0).sqrt()).clamp_min(0)
                tot_s = (Ns * A + Ns.sqrt() * S *
                         torch.randn(m.shape, generator=g_own, device=dev, **T32)).clamp_min(0)
                tot = torch.where(small, tot_s, tot)
                acc = tot * (sl.amp / sl.n_per_um2 / (cell * cell))
                sig = max(sl.cloud_um ** 2 * amean - cell ** 2 / 12, 0) ** 0.5 / cell
                D[i] += gblur(acc, sig)
    if ss > 1:
        D = -torch.log10(F.avg_pool2d((10.0 ** -D)[None], ss)[0])
    dmax = _t(np.array([f3.layer_dmax(st, i) for i in range(3)]), dev)
    f = D / dmax[:, None, None]
    inh = gblur(f, st.dir_um / px)
    dye = f * (1 - torch.einsum("ij,jhw->ihw", _t(st.dir_matrix, dev), inh)).clamp(0, 1)
    return dye


def _paper_lookup(logH, le, dc):
    """Paper characteristic curve lookup (uniform log-exposure grid)."""
    le0, dle, nl = float(le[0]), float(le[1] - le[0]), le.shape[0]
    x = ((logH - le0) / dle).clamp(0, nl - 1.001)
    idx = x.long(); fr = x - idx
    return dc[idx] * (1 - fr) + dc[idx + 1] * fr


def finish(st, dye, dev=DEV, medium="photo_paper", bleach=0.0):
    """Dye (3,H,W) -> sRGB numpy (H,W,3): same print LUT as the CPU engine (spectral print +
    lab-scan levels, or slide viewing), applied with trilinear 3D lookup on the GPU."""
    dye = gblur(dye, 0.4)
    lut = _t(f3.print_lut(st, medium, bleach), dev)                 # (n, n, n, 3), axes = dye C, M, Y
    n = lut.shape[0]
    vol = lut.permute(3, 0, 1, 2)[None]                             # (1, 3, D=C, H=M, W=Y)
    x = (dye.clamp(0, f3.LUT_MAX) / f3.LUT_MAX) * 2 - 1             # normalised coords
    h, w = dye.shape[1:]
    grid = torch.stack([x[2], x[1], x[0]], -1)[None, None]          # grid_sample wants (x=W, y=H, z=D)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]
    rgb = out.permute(1, 2, 0).clamp_min(0)
    srgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.clamp_min(1e-12) ** (1 / 2.4) - 0.055)
    return srgb.clamp(0, 1).cpu().numpy()


def simulate(st, lin, width_mm=36.0, dev=DEV, return_dye=False, medium="photo_paper", bleach=0.0):
    with torch.no_grad():
        dye = develop(st, lin, width_mm, dev)
        out = finish(st, dye, dev, medium, bleach)
    return (out, dye.cpu().numpy()) if return_dye else out


def measure_rms(st, density_over_base=1.0, aperture_um=48.0, patch_mm=1.6, px_um=4.0, dev=DEV):
    """Virtual microdensitometer on the GPU engine (green channel), for validation."""
    s = np.linspace(-4, 7, 221)
    Dn = f3.neutral_negative(st, s)
    if st.reversal:
        stop = np.interp(density_over_base, (Dn[1] - Dn[1, -1])[::-1], s[::-1])
    else:
        stop = np.interp(density_over_base, Dn[1] - Dn[1, 0], s)
    n = int(patch_mm * 1000 / px_um)
    stop -= getattr(st, "exposure_offset", 0.0)          # develop() re-applies the stock's offset
    lin = np.full((n, n, 3), 0.184 * 2 ** stop, np.float32)
    with torch.no_grad():
        dye = develop(st, lin, patch_mm, dev).cpu().numpy()
    D = f3.dye_to_density(st, dye)[1]
    from scipy import ndimage
    r = aperture_um / 2 / px_um
    yy, xx = np.mgrid[-int(r) - 1:int(r) + 2, -int(r) - 1:int(r) + 2]
    disk = ((yy ** 2 + xx ** 2) <= r * r).astype(float); disk /= disk.sum()
    T = ndimage.convolve(10.0 ** -D, disk, mode="reflect")
    m = int(r) + 2
    return 1000 * np.std(-np.log10(T[m:-m, m:-m]))
