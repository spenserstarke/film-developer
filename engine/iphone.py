"""
iPhone mode: turn a processed iPhone HEIC back into something closer to the light that was there,
before the film simulation sees it.

  1. Highlights: rebuild real highlight brightness from Apple's HDR gain map
     (Apple's published formula: HDR = SDR * (1 + (headroom - 1) * gain), headroom from maker-note
     tags 33 and 48). Lamps, windows and light sources regain up to ~2-3 stops above SDR white,
     so film's highlight roll-off and halation behave as they would on a real negative.
  2. Contrast: Smart HDR / Night mode lift shadows and flatten the scene. Restore global contrast
     around the mid-tones (luminance power curve; colour ratios preserved).
  3. Detail: soften the iPhone's sharpening halos (film never had them; film grain sits badly on
     top of them).
Each step degrades gracefully: if a file has no gain map / maker note, that step is skipped.
"""
import io
import struct
import numpy as np
from scipy import ndimage

Y_W = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _apple_makernote(mk):
    if not mk or mk[:9] != b"Apple iOS":
        return {}
    bo = ">" if mk[12:14] == b"MM" else "<"
    n = struct.unpack(bo + "H", mk[14:16])[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    out = {}
    for i in range(n):
        try:
            tag, typ, cnt, val = struct.unpack(bo + "HHI4s", mk[16 + 12 * i:28 + 12 * i])
            size = sizes.get(typ, 1) * cnt
            if size <= 4:
                data = val[:size]
            else:
                off = struct.unpack(bo + "I", val)[0]
                data = mk[off:off + size]
            if typ in (5, 10) and cnt == 1:
                a, b = struct.unpack(bo + ("II" if typ == 5 else "ii"), data)
                out[tag] = a / b if b else 0.0
            elif typ in (4, 9) and cnt == 1:
                out[tag] = struct.unpack(bo + ("I" if typ == 4 else "i"), data)[0]
        except struct.error:
            continue
    return out


def _headroom(t33, t48):
    """Apple's documented mapping from maker-note tags 33/48 to HDR headroom."""
    if t33 is None or t48 is None:
        return None
    if t33 < 1.0:
        stops = -20.0 * t48 + 1.8 if t48 <= 0.01 else -0.101 * t48 + 1.601
    else:
        stops = -70.0 * t48 + 3.0 if t48 <= 0.01 else -0.303 * t48 + 2.303
    return float(2.0 ** max(stops, 0.0))


def is_iphone(path):
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        return Image.open(path).getexif().get(0x010F, "") == "Apple"
    except Exception:
        return False


def load_iphone(path, contrast=1.3, desharpen_px=1.0, boost_stops=3.0, log=print):
    """Scene-linear sRGB float32 (H,W,3) from an iPhone HEIC/JPEG, with the steps above."""
    from PIL import Image, ImageOps, ImageCms
    import pillow_heif
    pillow_heif.register_heif_opener()
    im = Image.open(path)
    exif = im.getexif()
    mk = exif.get_ifd(0x8769).get(0x927C)
    orient = exif.get(0x0112, 1)
    icc = im.info.get("icc_profile")
    rgb_im = im.convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if "srgb" not in (ImageCms.getProfileDescription(src) or "").lower():
                rgb_im = ImageCms.profileToProfile(rgb_im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    sdr = _srgb_to_linear(np.asarray(rgb_im, np.float32) / 255.0).astype(np.float32)

    # 1) HDR gain map -> real highlight brightness
    hr = _headroom(*(lambda m: (m.get(33), m.get(48)))(_apple_makernote(mk)))
    gain = None
    try:
        h = pillow_heif.open_heif(path)
        aux = h.info.get("aux", {})
        ids = aux.get("urn:com:apple:photo:2020:aux:hdrgainmap")
        if ids:
            g = np.asarray(h.get_aux_image(ids[0]).to_pillow().convert("L"), np.float32) / 255.0
            gain = _srgb_to_linear(g)
            gain = ndimage.zoom(gain, (sdr.shape[0] / gain.shape[0], sdr.shape[1] / gain.shape[1]), order=1)
            gain = gain[:sdr.shape[0], :sdr.shape[1]]
    except Exception as e:
        log(f"  iPhone mode: no gain map ({e})")
    if gain is not None and hr:
        sdr = sdr * (1.0 + (hr - 1.0) * gain[..., None]).astype(np.float32)
        log(f"  iPhone mode: highlights restored from HDR gain map (up to {np.log2(hr):.1f} stops above white)")

    # 1b) light sources: the gain map tops out ~2-3 stops above white, but bulbs, lamps and the sun are
    #     6-10 stops brighter than the room. Film responds to that (halation, bloom), so extend
    #     regions that are clipped in the image AND maxed out in the gain map.
    if boost_stops:
        Ysdr = np.asarray(rgb_im.convert("L"), np.float32) / 255.0
        clipped = np.clip((Ysdr - 0.88) / 0.1, 0, 1)
        if gain is not None:
            clipped = clipped * np.clip((gain - 0.6) / 0.35, 0, 1)
        clipped = ndimage.gaussian_filter(clipped, 1.5)
        if clipped.max() > 0.05:
            sdr = sdr * (2.0 ** (boost_stops * clipped))[..., None].astype(np.float32)
            log(f"  iPhone mode: light sources extended by up to {boost_stops:.1f} stops "
                f"({100 * np.mean(clipped > 0.5):.2f}% of the frame)")

    # 2) undo the shadow lift: restore global contrast around the median, keep colour ratios
    if contrast and contrast != 1.0:
        Y = np.maximum(sdr @ Y_W, 1e-6)
        mid = float(np.median(Y))
        Yn = mid * (Y / mid) ** contrast
        sdr = sdr * (Yn / Y)[..., None]
        log(f"  iPhone mode: global contrast restored (x{contrast} around mid-tones)")

    # 3) soften the sharpening halos
    if desharpen_px:
        sdr = np.stack([ndimage.gaussian_filter(sdr[..., c], desharpen_px) for c in range(3)], -1)

    # orientation (pillow-heif already applies HEIF rotation; EXIF only matters for JPEG)
    if path.lower().endswith((".jpg", ".jpeg")):
        sdr = _rotate(sdr, orient)
    return np.ascontiguousarray(sdr, dtype=np.float32)


def _rotate(a, orient):
    return {3: lambda x: np.rot90(x, 2), 6: lambda x: np.rot90(x, -1), 8: lambda x: np.rot90(x, 1)}.get(
        orient, lambda x: x)(a)
