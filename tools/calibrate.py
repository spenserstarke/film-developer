"""
Fit filmsim stock parameters to MEASURED film data.

How to get real data for a stock (the part that needs a darkroom/lab, not code):
  1. Photograph a grey card (or use a calibrated step wedge / sensitometer) on the film at a
     ladder of exposures, e.g. -5 to +7 stops in 1/3-stop steps, keeping lighting fixed.
  2. Develop normally (note lab + date: chemistry varies).
  3. Read each patch with a densitometer in Status M (R, G, B), or scan with a calibrated
     scanner with all auto-correction OFF and convert to density = -log10(transmittance).
  4. Save as CSV:  stops,D_R,D_G,D_B     (stops relative to correct mid-grey exposure)

Then:  python calibrate.py wedge.csv   ->   fitted stock + fit plot.

With no CSV, it runs a self-test: it invents a "measured" stock with different parameters,
simulates a noisy wedge reading, and checks the fitter recovers its curves.
"""
import sys
import numpy as np
from scipy.optimize import least_squares
import filmsim as fs

PARAMS = ["speed", "fast_sens", "fast_sigma", "slow_sigma", "fast_weight", "dmax"]
LO = np.array([0.1, 1.5, 0.4, 0.4, 0.05, 0.8])
HI = np.array([10., 40., 3.0, 3.0, 0.80, 3.5])


def _get(stock):
    v = []
    for i, L in enumerate(stock.layers):
        f, s = L.emulsions
        v += [stock.spectral[i].sum(), f.median_sens / s.median_sens, f.sigma, s.sigma, f.weight, L.dmax]
    return np.array(v, float)


def _set(stock, v):
    st = stock.copy()
    v = v.reshape(3, 6)
    for i, L in enumerate(st.layers):
        speed, fsens, fsig, ssig, fw, dmax = v[i]
        st.spectral[i] *= speed / st.spectral[i].sum()
        f, s = L.emulsions
        f.median_sens, f.sigma, f.weight = fsens * s.median_sens, fsig, fw
        s.sigma, s.weight = ssig, 1 - fw
        L.dmax = dmax
    return st


def fit_stock(stops, D_measured, start=None, verbose=True):
    """stops: (N,), D_measured: (N,3) Status-M densities incl. base. Returns (stock, rms)."""
    start = start or fs.make_stock()
    D_measured = np.asarray(D_measured, float)
    x0 = np.clip(_get(start), np.tile(LO, 3) + 1e-6, np.tile(HI, 3) - 1e-6)

    def model(v, st_base):
        st = _set(st_base, v)
        return fs.neutral_negative(st, stops).T, st

    # base+fog: solve for it so the toe matches (base is linear, so solve exactly per step)
    def resid(v):
        D, st = model(v, start)
        base_fix = np.mean(D_measured - D, axis=0)
        return ((D + base_fix) - D_measured).ravel()

    r = least_squares(resid, x0, bounds=(np.tile(LO, 3), np.tile(HI, 3)), x_scale="jac")
    D, st = model(r.x, start)
    st.base = tuple(np.array(st.base) + np.mean(D_measured - D, axis=0))
    rms = float(np.sqrt(np.mean((fs.neutral_negative(st, stops).T - D_measured) ** 2)))
    if verbose:
        print(f"fit RMS error: {rms:.4f} density units  ({r.nfev} model evaluations)")
    return st, rms


def read_wedge_csv(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a["stops"], np.stack([a["D_R"], a["D_G"], a["D_B"]], 1)


def plot_fit(stops, D_meas, stock, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.linspace(stops.min() - 0.5, stops.max() + 0.5, 200)
    D = fs.neutral_negative(stock, s)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, col in enumerate(["#d0332e", "#2f9a44", "#2f5fd0"]):
        ax.plot(s * np.log10(2), D[c], color=col, lw=2, label=f"fitted {'RGB'[c]}")
        ax.plot(stops * np.log10(2), D_meas[:, c], "o", color=col, ms=4, mfc="none", label=f"measured {'RGB'[c]}")
    ax.set_xlabel("log10 exposure (0 = mid-grey)")
    ax.set_ylabel("Status-M density")
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def self_test():
    rng = np.random.default_rng(42)
    truth = fs.make_stock("Pretend measured stock")
    tv = _get(truth).reshape(3, 6)
    tv[:, 0] *= [0.8, 1.0, 1.25]          # layer speeds differ
    tv[:, 1] = [5.5, 7.0, 10.0]           # fast/slow sensitivity ratio
    tv[:, 2] = [1.0, 1.2, 1.4]
    tv[:, 3] = [1.8, 1.6, 1.3]
    tv[:, 4] = [0.30, 0.40, 0.45]
    tv[:, 5] = [1.7, 2.2, 2.4]
    truth = _set(truth, tv.ravel())
    truth.base = (0.10, 0.20, 0.30)
    stops = np.arange(-5, 7.01, 1 / 3)
    D_meas = fs.neutral_negative(truth, stops).T + rng.normal(0, 0.01, (len(stops), 3))  # densitometer noise
    np.savetxt("example_wedge_synthetic.csv", np.column_stack([stops, D_meas]), delimiter=",",
               header="stops,D_R,D_G,D_B", comments="", fmt="%.4f")
    fitted, rms = fit_stock(stops, D_meas)
    true_curve = fs.neutral_negative(truth, stops).T
    curve_err = np.abs(fs.neutral_negative(fitted, stops).T - true_curve).max()
    print(f"max deviation from TRUE curve: {curve_err:.4f} density (noise was 0.01)")
    plot_fit(stops, D_meas, fitted, "calibration_fit.png",
             f"Calibration self-test: fit to a noisy synthetic step wedge (RMS {rms:.3f} D)")
    return fitted, rms, curve_err


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stops, D = read_wedge_csv(sys.argv[1])
        st, rms = fit_stock(stops, D)
        plot_fit(stops, D, st, "calibration_fit.png", f"Fit to {sys.argv[1]} (RMS {rms:.3f} D)")
        print(np.round(_get(st).reshape(3, 6), 3), "\nparams per layer:", PARAMS)
    else:
        self_test()
