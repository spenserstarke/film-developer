"""
Grain targets for every camera stock, with where each number comes from.

RMS granularity = diffuse RMS x1000, 48 µm aperture, density 1.0 (datasheet convention).
  "datasheet"  - published by the manufacturer (verified)
  "from PGI"   - Kodak publishes Print Grain Index instead of RMS. Kodak: 4 PGI units = 1 just-
                 noticeable difference; a JND ≈ 6% change in RMS granularity (Zwick & Brothers 1977,
                 cited in Kodak patent US5629769). So RMS = RMS_ref * 1.06 ** (ΔPGI / 4), anchored on
                 Portra 400 ≈ 4 (the ISO 400 pro negative Fuji measured, Pro 400H, is RMS 4).
  "estimate"   - no published figure found; reasoned guess, change freely.
"""
P400 = 4.0
pgi = lambda d_pgi: round(P400 * 1.06 ** (d_pgi / 4), 2)

STOCKS = {
    # name                          display name                    RMS          source
    "kodak_portra_160":            ("Kodak Portra 160",            pgi(28 - 37), "from PGI (28 vs 37 at 4x6)"),
    "kodak_portra_400":            ("Kodak Portra 400",            P400,         "anchor (≈ Fuji Pro 400H datasheet RMS 4)"),
    "kodak_portra_800":            ("Kodak Portra 800",            pgi(48 - 37), "from PGI (48 vs 37 at 4x6)"),
    "kodak_portra_800_push1":      ("Kodak Portra 800 (push +1)",  round(pgi(11) * 1.15, 2), "estimate (push raises grain)"),
    "kodak_portra_800_push2":      ("Kodak Portra 800 (push +2)",  round(pgi(11) * 1.3, 2),  "estimate (push raises grain)"),
    "kodak_ektar_100":             ("Kodak Ektar 100",             pgi(38 - 59), "from PGI (38 vs 59 at 8x10)"),
    "kodak_gold_200":              ("Kodak Gold 200",              pgi(44 - 37), "from PGI (44 vs 37 at 4x6)"),
    "kodak_ultramax_400":          ("Kodak UltraMax 400",          pgi(46 - 37), "from PGI (46 vs 37 at 4x6)"),
    "kodak_verita_200d":           ("Kodak Verita 200D",           4.2,          "estimate"),
    "kodak_vision3_50d":           ("Kodak Vision3 50D",           3.0,          "estimate"),
    "kodak_vision3_250d":          ("Kodak Vision3 250D",          3.8,          "estimate"),
    "kodak_vision3_200t":          ("Kodak Vision3 200T",          3.8,          "estimate"),
    "kodak_vision3_500t":          ("Kodak Vision3 500T",          4.8,          "estimate"),
    "fujifilm_c200":               ("Fujifilm C200",               4.0,          "datasheet"),
    "fujifilm_pro_400h":           ("Fujifilm Pro 400H",           4.0,          "datasheet"),
    "fujifilm_xtra_400":           ("Fujifilm Superia X-TRA 400",  5.0,          "estimate"),
    "fujifilm_provia_100f":        ("Fujifilm Provia 100F (slide)", 8.0,         "datasheet"),
    "fujifilm_velvia_100":         ("Fujifilm Velvia 100 (slide)", 8.0,          "datasheet"),
    "kodak_ektachrome_100":        ("Kodak Ektachrome E100 (slide)", 8.0,        "datasheet"),
    "kodak_kodachrome_64":         ("Kodak Kodachrome 64 (slide)", 10.0,         "estimate (commonly cited)"),
}
