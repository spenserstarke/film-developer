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

Output, saved next to each input:
  RAW / DNG in   -> <name>_<stock>.tif  (16-bit, sRGB-tagged, for editing)  +  .jpg (for sharing)
  JPEG in        -> .jpg     PNG in -> .png (16-bit)     TIFF in -> .tif (16-bit)
"""
import argparse
import glob
import os
import pickle
import time
import numpy as np
from skimage import transform
import filmsim3 as f3

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
    base, ext = os.path.splitext(src_path)
    ext = ext.lower()
    stem = f"{base}_{stock_key}"
    if ext in RAW_EXTS or ext in (".tif", ".tiff"):
        save_tiff16(stem + ".tif", srgb)
        saved = [stem + ".tif"]
        if ext in RAW_EXTS:
            save_jpeg(stem + ".jpg", srgb); saved.append(stem + ".jpg")
    elif ext == ".png":
        save_png16(stem + ".png", srgb); saved = [stem + ".png"]
    else:
        save_jpeg(stem + ".jpg", srgb); saved = [stem + ".jpg"]
    return saved


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
    for n, path in enumerate(a.photos, 1):
        print(f"PROGRESS {n}/{len(a.photos)} {os.path.basename(path)}", flush=True)
        t = time.time()
        print(f"Developing {os.path.basename(path)} on {getattr(st, 'display_name', a.stock)} ...", flush=True)
        lin = f3.load_image(path, exposure_stops=a.exposure)
        H, W = lin.shape[:2]
        w = min(a.width, W)
        lin = transform.resize(lin, (int(round(H * w / W)), w), anti_aliasing=True).astype(np.float32)
        if gpu is not None:
            out, dye = gpu.simulate(st, lin, width_mm=FORMAT_MM[a.format], return_dye=True)
            D = f3.dye_to_density(st, dye)
        else:
            out, D = f3.simulate(st, lin, width_mm=FORMAT_MM[a.format])
        saved = save_outputs(path, a.stock, out)
        if a.cineon and not st.reversal:
            save_cineon(os.path.splitext(path)[0] + f"_{a.stock}_cineon.tif", st, D)
        if a.negative and not st.reversal and D is not None:
            neg = f3.linear_to_srgb(np.clip(np.moveaxis(10.0 ** -D, 0, -1) * 2, 0, 1))
            save_jpeg(os.path.splitext(path)[0] + f"_{a.stock}_negative.jpg", neg)
        print("  saved " + ", ".join(os.path.basename(s) for s in saved) + f"  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
