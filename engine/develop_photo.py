"""
Run your own photos through the film simulation.

  python3 develop_photo.py photo1.dng photo2.ARW ...

Options:
  --stock kodak_portra_400   which film (see --list)
  --list                     show available film stocks
  --width 2400               output width in pixels (bigger = slower)
  --format 35 | 120          film format (120 = finer-looking grain)
  --exposure 0.5             exposure adjustment in stops
  --negative                 also save the negative (negative films only)

Output, saved next to each input: <name>_<stock>.tif  (16-bit, sRGB-tagged)
"""
import argparse
import glob
import os
import pickle
import time
import numpy as np
from skimage import transform
import filmsim3 as f3
import iphone

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK_DIR = os.path.join(HERE, "stocks")
FORMAT_MM = {"35": 36.0, "120": 56.0}
RAW_EXTS = {"." + e for e in f3.__dict__.get("RAW_EXTS", [])} | {
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr", ".iiq"}


def list_stocks():
    out = {}
    for p in sorted(glob.glob(os.path.join(STOCK_DIR, "*.pkl"))):
        key = os.path.splitext(os.path.basename(p))[0]
        out[key] = p
    return out


def srgb_icc():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def save_tiff16(path, srgb):
    import tifffile
    tifffile.imwrite(path, (np.clip(srgb, 0, 1) * 65535 + 0.5).astype(np.uint16),
                     photometric="rgb", compression="zlib", iccprofile=srgb_icc())


def save_jpeg(path, srgb):
    from PIL import Image
    Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        path, quality=95, icc_profile=srgb_icc())


def save_png16(path, srgb):
    import imageio.v3 as iio
    iio.imwrite(path, (np.clip(srgb, 0, 1) * 65535 + 0.5).astype(np.uint16))


def save_cineon(path, st, D):
    """Cineon printing-density log: code = 95 + (D - D_min) / 0.002 (10-bit), stored in 16 bits.
    Loads in Resolve/Nuke as 'Cineon log' for print-film emulation or grading."""
    import tifffile
    dmin = f3.neutral_negative(st, np.array([-12.0]))[:, 0]
    code = 95 + (np.moveaxis(D, 0, -1) - dmin) / 0.002
    tifffile.imwrite(path, (np.clip(code, 0, 1023) * 64).astype(np.uint16), photometric="rgb",
                     compression="zlib")


def save_outputs(src_path, stock_key, srgb):
    """Every input is saved as one 16-bit, sRGB-tagged TIFF next to the original."""
    path = f"{os.path.splitext(src_path)[0]}_{stock_key}.tif"
    save_tiff16(path, srgb)
    return [path]


class Progress:
    """Writes 'token<TAB>percent<TAB>title<TAB>detail' to $FD_PROGRESS for the app's progress bar."""
    def __init__(self, n):
        self.path, self.n, self.last = os.environ.get("FD_PROGRESS"), n, 0.0
        self.token = f"dev{os.getpid()}"

    def __call__(self, i, frac, detail, force=False):
        if not self.path:
            return
        now = time.time()
        if not force and now - self.last < 0.15:
            return
        self.last = now
        pct = int(100 * ((i - 1) + min(max(frac, 0.0), 1.0)) / self.n)
        title = f"Developing photo {i} of {self.n}" if self.n > 1 else "Developing photo"
        try:
            with open(self.path + ".tmp", "w") as fh:
                fh.write(f"{self.token}\t{pct}\t{title}\t{detail}\n")
            os.replace(self.path + ".tmp", self.path)
        except OSError:
            pass


Y_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], np.float32)


def reference_brightness(path):
    """Median brightness (sRGB) of how the photo is meant to look: the camera's / editor's embedded
    preview for RAW files, or the image itself for JPEG/PNG/TIFF. None if unavailable."""
    import io
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in RAW_EXTS:
            import rawpy
            from PIL import Image
            with rawpy.imread(path) as r:
                th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                img = np.asarray(Image.open(io.BytesIO(th.data)).convert("RGB"), np.float32) / 255
            else:
                img = np.asarray(th.data, np.float32) / 255
            if min(img.shape[:2]) < 120:          # tiny thumbnails aren't reliable
                return None
        else:
            img = f3.linear_to_srgb(f3.load_image(path, meter="none"))
        return float(np.median(img @ Y_WEIGHTS))
    except Exception:
        return None


CAMERA_BASELINE_STOPS = 1.3     # fallback when a RAW has no usable preview (typical camera rendering)


def match_exposure(st, lin, target, width_mm):
    """Exposure offset (stops) so the film result has the same median brightness as the reference."""
    h, w = lin.shape[:2]
    small = transform.resize(lin, (max(8, int(h * 360 / w)), 360), anti_aliasing=True).astype(np.float32)
    ks = np.arange(-6.0, 5.01, 1.0)
    meds = []
    for k in ks:
        out, _ = f3.simulate(st, small * np.float32(2.0 ** k), width_mm=width_mm)
        meds.append(float(np.median(out @ Y_WEIGHTS)))
    meds = np.maximum.accumulate(np.array(meds))          # monotonic for interpolation
    # very dark scenes: the print's black (paper D-max) puts a floor under the median. Don't chase a
    # target below what film can print - stop where lowering exposure no longer changes the result.
    floor = meds[0] + 0.15 * (meds[-1] - meds[0]) * 0.1
    if target < floor:
        target = floor
    if target <= meds[0]:
        return float(ks[0])
    if target >= meds[-1]:
        return float(ks[-1])
    i = int(np.searchsorted(meds, target))
    k0, k1, m0, m1 = ks[i - 1], ks[i], meds[i - 1], meds[i]
    k = k0 + (k1 - k0) * (target - m0) / max(m1 - m0, 1e-6)
    out, _ = f3.simulate(st, small * np.float32(2.0 ** k), width_mm=width_mm)    # refine once
    m = float(np.median(out @ Y_WEIGHTS))
    slope = (m1 - m0) / (k1 - k0)
    return float(np.clip(k + (target - m) / max(slope, 1e-3), -3.0, ks[-1]))


def main():
    ap = argparse.ArgumentParser(description="Develop photos on simulated film (filmsim v3)")
    ap.add_argument("photos", nargs="*")
    ap.add_argument("--stock", default="kodak_portra_400")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--format", choices=FORMAT_MM, default="35")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--engine", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--no-iphone-mode", action="store_true",
                    help="treat iPhone HEIC/JPEG like any other image (no gain-map/contrast/de-sharpen)")
    ap.add_argument("--cineon", action="store_true",
                    help="also save the scanned negative as a 16-bit Cineon log TIFF (for grading)")
    ap.add_argument("--list-tsv", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    stocks = list_stocks()
    if a.list_tsv:   # for the Mac app's stock picker
        for k, p in stocks.items():
            print(f"{k}\t{getattr(pickle.load(open(p, 'rb')), 'display_name', k)}")
        return
    if a.list:
        for k, p in stocks.items():
            st = pickle.load(open(p, "rb"))
            print(f"{k:<28} {getattr(st, 'display_name', k)}")
        return
    if a.stock not in stocks:
        raise SystemExit(f"Unknown stock '{a.stock}'. Use --list to see available stocks.")
    st = pickle.load(open(stocks[a.stock], "rb"))
    gpu = None
    if a.engine in ("auto", "gpu"):
        try:
            import filmsim_gpu as gpu
            if a.engine == "auto" and gpu.DEV.type == "cpu":
                gpu = None                     # numpy engine is faster than torch on CPU
            else:
                print(f"Using GPU engine on {gpu.DEV.type}", flush=True)
        except Exception as e:
            if a.engine == "gpu":
                raise SystemExit(f"GPU engine unavailable: {e}")
            gpu = None
    prog = Progress(len(a.photos))
    for n, path in enumerate(a.photos, 1):
        print(f"PROGRESS {n}/{len(a.photos)} {os.path.basename(path)}", flush=True)
        t = time.time()
        print(f"Developing {os.path.basename(path)} on {getattr(st, 'display_name', a.stock)} ...", flush=True)
        name = os.path.basename(path)
        prog(n, 0.0, f"{name}: reading", force=True)
        ext = os.path.splitext(path)[1].lower()
        iphone_mode = (not a.no_iphone_mode and ext in (".heic", ".heif", ".jpg", ".jpeg")
                       and iphone.is_iphone(path))
        if iphone_mode:
            lin = iphone.load_iphone(path, log=lambda m: print(m, flush=True))
        else:
            lin = f3.load_image(path, meter="none")         # keep the photographer's exposure
        prog(n, 0.12, f"{name}: preparing", force=True)
        H, W = lin.shape[:2]
        w = min(a.width, W)
        lin = transform.resize(lin, (int(round(H * w / W)), w), anti_aliasing=True).astype(np.float32)
        prog(n, 0.16, f"{name}: matching your exposure", force=True)
        target = reference_brightness(path)
        if target is not None and iphone_mode:
            target *= iphone.scene_params(path)["darken"]   # night mode over-brightens; daylight barely
        if target is not None:
            k = match_exposure(st, lin, target, FORMAT_MM[a.format])
            how = f"matched preview brightness {target:.2f}"
        else:
            k = CAMERA_BASELINE_STOPS if os.path.splitext(path)[1].lower() in RAW_EXTS else 0.0
            how = "camera baseline"
        k += a.exposure
        lin = lin * np.float32(2.0 ** k)
        print(f"  exposure {k:+.2f} stops ({how}{', adjusted %+.1f' % a.exposure if a.exposure else ''})", flush=True)
        prog(n, 0.18, f"{name}: exposing and developing the film", force=True)
        f3.PROGRESS = lambda fr, n=n, name=name: prog(
            n, 0.18 + 0.62 * fr, f"{name}: " + ("making the print" if fr >= 0.999
                                                 else "exposing and developing the film"), force=fr >= 0.999)
        if gpu is not None:
            out, dye = gpu.simulate(st, lin, width_mm=FORMAT_MM[a.format], return_dye=True)
            D = f3.dye_to_density(st, dye)
        else:
            out, D = f3.simulate(st, lin, width_mm=FORMAT_MM[a.format])
        f3.PROGRESS = None
        prog(n, 0.9, f"{name}: saving 16-bit TIFF", force=True)
        saved = save_outputs(path, a.stock, out)
        prog(n, 1.0, f"{name}: done", force=True)
        if a.cineon and not st.reversal:
            save_cineon(os.path.splitext(path)[0] + f"_{a.stock}_cineon.tif", st, D)
        if a.negative and not st.reversal and D is not None:
            neg = f3.linear_to_srgb(np.clip(np.moveaxis(10.0 ** -D, 0, -1) * 2, 0, 1))
            save_jpeg(os.path.splitext(path)[0] + f"_{a.stock}_negative.jpg", neg)
        print("  saved " + ", ".join(os.path.basename(s) for s in saved) + f"  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
