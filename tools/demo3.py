"""
v3 demo: python demo3.py
Renders the RAW samples with the Portra 400 stock built from published data, plus
a full-resolution 100% crop to judge grain, and plots of the data fit.
"""
import json
import pickle
import time
import numpy as np
from skimage import transform, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import filmsim as v2
import filmsim3 as f3

PROFILE = "ext/spektrafilm/src/spektrafilm/data/profiles/kodak_portra_400.json"
st = pickle.load(open("kodak_portra_400_v3.pkl", "rb"))
old = v2.make_stock()
save = lambda n, a: io.imsave(n, (np.clip(a, 0, 1) * 255).astype(np.uint8), quality=95)


def render(path_npy, name, width=1600, crop=None):
    lin_full = np.load(path_npy)
    H, W = lin_full.shape[:2]
    h = int(round(H * width / W))
    lin = transform.resize(lin_full, (h, width), anti_aliasing=True).astype(np.float32)
    t = time.time(); out3, _ = f3.simulate(st, lin); t3 = time.time() - t
    out2, _ = v2.simulate(old, lin)
    print(f"{name}: v3 {width}x{h} in {t3:.0f}s")
    save(f"{name}_portra400_v3.jpg", out3)
    res = dict(digital=v2.linear_to_srgb(lin), v2=out2, v3=out3)
    if crop:   # 100% crop at full camera resolution: physical grain size is preserved
        y, x, n = crop
        c = lin_full[y:y + n, x:x + n]
        mm = 36.0 * n / W
        t = time.time(); c3, _ = f3.simulate(st, c, width_mm=mm); print(f"  100% crop {n}px in {time.time() - t:.0f}s")
        c2, _ = v2.simulate(old, c)
        res.update(c_digital=v2.linear_to_srgb(c), c_v2=c2, c_v3=c3)
        save(f"{name}_crop100_v3.png", c3)
    return res


def main():
    dng = render("samples/dng_lin.npy", "mountain", crop=(1650, 3500, 520))
    arw = render("samples/lin_full.npy", "temple", crop=(2350, 3900, 520))

    # ---------------- image comparison
    fig = plt.figure(figsize=(18, 21))
    gs = fig.add_gridspec(4, 3)
    up = lambda a: np.kron(np.clip(a, 0, 1), np.ones((2, 2, 1)))
    labels = ["Digital", "v2 (placeholder stock, pixel grain)", "v3 (Portra 400 from published data)"]
    for col, key in enumerate(["digital", "v2", "v3"]):
        for row, (r, ttl) in enumerate([(dng, ""), (arw, "")]):
            ax = fig.add_subplot(gs[row * 2, col]); ax.imshow(np.clip(r[key], 0, 1)); ax.axis("off")
            ax.set_title(labels[col], fontsize=13)
            ax = fig.add_subplot(gs[row * 2 + 1, col]); ax.imshow(up(r["c_" + key]), interpolation="nearest")
            ax.axis("off"); ax.set_title("100% crop at full camera resolution (shown 2x)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.suptitle("filmsim v3 — Kodak Portra 400 built from published datasheet data", fontsize=16)
    fig.savefig("v3_comparison.png", dpi=72)

    # ---------------- data plots
    d = json.load(open(PROFILE))["data"]
    le = np.array(d["log_exposure"], float); dc = np.array(d["density_curves"], float)
    sel = (le > -2.6) & (le < 3.4)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    model = f3.neutral_negative(st, le[sel] / f3.LOG2)
    cols = ["#d0332e", "#2f9a44", "#2f5fd0"]
    for c in range(3):
        ax[0].plot(le[sel][::6], (dc[sel] + st.base[None])[::6, c], "o", mfc="none", color=cols[c], ms=5,
                   label=f"{'RGB'[c]} datasheet (digitised)")
        ax[0].plot(le[sel], model[c], color=cols[c], lw=2, label=f"{'RGB'[c]} grain physics fit")
    ax[0].set_xlabel("log10 exposure (0 = mid-grey)"); ax[0].set_ylabel("Status-M density")
    ax[0].set_title(f"Characteristic curves: fit RMS error {st.fit_rms:.4f} D"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=9, ncol=2)
    rc = np.load("rms_curve.npy")[:-1]
    for c in range(3):
        ax[1].plot(rc[:, 0], rc[:, c + 1], "o-", color=cols[c], label=f"{'RGB'[c]} layer")
    ax[1].axhline(5, color="grey", ls="--", lw=1); ax[1].text(0.15, 5.1, "fitted target: RMS 5 @ D=1.0 (green)", color="grey")
    ax[1].set_xlabel("Density above D-min"); ax[1].set_ylabel("Diffuse RMS granularity (48 µm aperture)")
    ax[1].set_title("Simulated granularity vs density (virtual microdensitometer)"); ax[1].grid(alpha=.3)
    ax[1].legend(); ax[1].set_ylim(0, 6.5)
    fig.tight_layout(); fig.savefig("v3_data.png", dpi=90)


if __name__ == "__main__":
    main()
