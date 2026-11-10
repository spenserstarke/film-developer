"""
Creative looks: a colour grade applied ON TOP of the finished film image (display-referred sRGB).

The same maths runs in two places and must stay identical:
  * apply_look() here (numpy, on the full-resolution 16-bit TIFF when you save)
  * the GLSL fragment shader in finish_ui.html (real-time preview)

Per pixel x (sRGB 0..1):
  1. tonal zone weights from luma Y:  shadows (1-2Y)^2, highlights (2Y-1)^2, mids = rest
  2. zone tints  x += strength * (ws*SH + wm*MID + wh*HI)   (+ user warmth/tint, any strength)
  3. saturation  x = Y + (x - Y) * (1 + strength*(sat-1) + user_sat)
  4. contrast    S-curve that keeps 0 and 1 fixed: x += k*(x-0.5)*4x(1-x),  k = strength*(contrast-1) + user_contrast
  5. fade        x = f + x*(1-f),  f = strength*fade + user_fade
Preset tints are defined as CIELAB a*/b* shifts per zone (measured from reference frames, see
looks.json); they are converted once to sRGB offsets so the shader only adds vectors.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
ZONE_L = {"shadows": 20.0, "mids": 50.0, "highlights": 80.0}
CONTENT_DISCOUNT = 0.7     # part of each reference's average colour is set/costume/light, not grade

# Measured on ShotDeck reference frames (35mm/16mm titles, ~15-39 frames each): CIELAB a*/b* means of
# shadows (L<25), mids, highlights (L>70) + median chroma. Names are descriptive; the reference
# field is for our own bookkeeping and isn't shown in the app.
PRESETS = [
    dict(id="warm_print", name="Warm Print", ref="Phantom Thread", sh=(4.3, 2.2), mid=(6.2, 8.6), hi=(2.2, 4.5), chroma=15.2, contrast=1.05, fade=0.0),
    dict(id="golden_hour", name="Golden Hour", ref="Once Upon a Time in Hollywood", sh=(2.6, 4.1), mid=(3.8, 16.5), hi=(-0.2, 18.8), chroma=21.1, contrast=1.08, fade=0.0),
    dict(id="italian_summer", name="Italian Summer", ref="Call Me by Your Name", sh=(-3.4, 5.6), mid=(-6.8, 16.8), hi=(-9.7, 15.9), chroma=19.6, contrast=1.0, fade=0.03),
    dict(id="neon_romance", name="Neon Romance", ref="In the Mood for Love", sh=(6.9, 3.6), mid=(12.2, 9.3), hi=(-3.7, 17.7), chroma=21.7, contrast=1.1, fade=0.0),
    dict(id="gilded", name="Gilded", ref="Amelie", sh=(7.3, 11.3), mid=(11.2, 36.2), hi=(1.4, 26.7), chroma=37.6, contrast=1.05, fade=0.0),
    dict(id="confection", name="Confection", ref="The Grand Budapest Hotel", sh=(7.5, 2.6), mid=(13.0, 8.5), hi=(9.7, 6.5), chroma=24.0, contrast=1.0, fade=0.02),
    dict(id="tokyo_blue", name="Tokyo Blue", ref="Lost in Translation", sh=(1.9, -4.8), mid=(2.1, -9.5), hi=(1.6, -11.8), chroma=12.5, contrast=0.97, fade=0.04),
    dict(id="winter_16mm", name="Winter 16mm", ref="Carol", sh=(0.1, 3.9), mid=(-5.6, 6.3), hi=(-16.3, 7.4), chroma=14.3, contrast=1.0, fade=0.03),
    dict(id="digital_rain", name="Digital Rain", ref="The Matrix", sh=(-2.5, 3.9), mid=(-13.4, 9.5), hi=(-3.6, 0.9), chroma=17.5, contrast=1.08, fade=0.0),
    dict(id="fluorescent", name="Fluorescent", ref="Fight Club", sh=(-1.6, 1.4), mid=(-7.6, 7.6), hi=(-5.6, 5.3), chroma=15.5, contrast=1.08, fade=0.0),
    dict(id="silver_grit", name="Silver Grit", ref="Se7en", sh=(-1.1, 2.4), mid=(-3.8, 9.0), hi=(-8.1, 7.7), chroma=13.6, contrast=1.15, fade=0.0),
    dict(id="steel_night", name="Steel Night", ref="Heat", sh=(-1.2, -1.3), mid=(-1.6, -1.5), hi=(-0.4, 3.2), chroma=8.3, contrast=1.08, fade=0.0),
    dict(id="battlefield", name="Battlefield", ref="Saving Private Ryan", sh=(0.4, 1.0), mid=(0.4, 0.9), hi=(-0.6, -0.1), chroma=8.5, contrast=1.18, fade=0.0),
]
NEUTRAL_CHROMA = 19.0      # typical median Lab chroma of an ungraded film frame


def _lab_offset(L, a, b):
    """sRGB offset that moves a neutral grey of lightness L to (L, a, b)."""
    import colour
    ref = colour.XYZ_to_sRGB(colour.Lab_to_XYZ([L, 0, 0]))
    tgt = colour.XYZ_to_sRGB(colour.Lab_to_XYZ([L, a, b]))
    enc = lambda v: colour.cctf_encoding(np.clip(v, 0, None), "sRGB")
    return (enc(tgt) - enc(ref)).astype(np.float32)


def build_presets(path=os.path.join(HERE, "looks.json")):
    """Convert measured Lab tints to the sRGB offsets the shader and apply_look() use."""
    out = {"zones": ZONE_L, "neutral_chroma": NEUTRAL_CHROMA, "content_discount": CONTENT_DISCOUNT,
           "unit_warmth": _lab_offset(50, 0, 1).tolist(), "unit_tint": _lab_offset(50, 1, 0).tolist(),
           "presets": [dict(id="none", name="No look", sh=[0, 0, 0], mid=[0, 0, 0], hi=[0, 0, 0],
                            sat=1.0, contrast=1.0, fade=0.0)]}
    for p in PRESETS:
        k = CONTENT_DISCOUNT
        out["presets"].append(dict(
            id=p["id"], name=p["name"], ref=p["ref"],
            sh=_lab_offset(ZONE_L["shadows"], k * p["sh"][0], k * p["sh"][1]).round(5).tolist(),
            mid=_lab_offset(ZONE_L["mids"], k * p["mid"][0], k * p["mid"][1]).round(5).tolist(),
            hi=_lab_offset(ZONE_L["highlights"], k * p["hi"][0], k * p["hi"][1]).round(5).tolist(),
            sat=round(float(np.clip(p["chroma"] / NEUTRAL_CHROMA, 0.4, 1.6)), 3),
            contrast=p["contrast"], fade=p["fade"]))
    json.dump(out, open(path, "w"), indent=1)
    return out


def load_presets(path=os.path.join(HERE, "looks.json")):
    return json.load(open(path))


def apply_look(srgb, preset, strength=1.0, warmth=0.0, tint=0.0, saturation=0.0, contrast=0.0,
               fade=0.0, lib=None):
    """srgb: float array (...,3) in 0..1. preset: dict from looks.json. User sliders are additive:
    warmth/tint in Lab units (-20..20), saturation/contrast as deltas (-1..1), fade 0..0.15."""
    lib = lib or load_presets()
    x = np.asarray(srgb, np.float32).copy()
    s = np.float32(strength)
    Y = x @ LUMA
    ws = np.clip(1 - 2 * Y, 0, 1) ** 2
    wh = np.clip(2 * Y - 1, 0, 1) ** 2
    wm = 1 - ws - wh
    off = (ws[..., None] * np.float32(preset["sh"]) + wm[..., None] * np.float32(preset["mid"])
           + wh[..., None] * np.float32(preset["hi"]))
    x = x + s * off + np.float32(warmth) * np.float32(lib["unit_warmth"]) + np.float32(tint) * np.float32(lib["unit_tint"])
    Y = x @ LUMA
    x = Y[..., None] + (x - Y[..., None]) * np.float32(1 + s * (preset["sat"] - 1) + saturation)
    k = np.float32(s * (preset["contrast"] - 1) + contrast)
    x = x + k * (x - 0.5) * 4 * x * (1 - x)
    f = np.float32(s * preset["fade"] + fade)
    x = f + x * (1 - f)
    return np.clip(x, 0, 1)


if __name__ == "__main__":
    lib = build_presets()
    for p in lib["presets"]:
        print(f"{p['name']:<16} sat {p['sat']:.2f} contrast {p['contrast']:.2f} mid offset {np.round(p['mid'], 3)}")
