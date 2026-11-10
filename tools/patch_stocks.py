import glob, pickle, os, time, copy
import filmsim3 as f3
HAL = {"strong": (0.02, 0.004, 0.0), "weak": (0.05, 0.01, 0.0)}
for p in sorted(glob.glob("stocks/*.pkl")):
    if "no_remjet" in p:
        continue
    t = time.time()
    st = pickle.load(open(p, "rb"))
    st.halation = HAL.get(st.profile.get("antihalation", "strong"), HAL["strong"])
    f3.build_spectral_lut(st)
    st._print_cache = None
    pickle.dump(st, open(p, "wb"))
    print(f"OK {st.display_name:<32} halation {st.profile.get('antihalation')}  {time.time()-t:.0f}s", flush=True)
# rem-jet removed variants (what CineStill does to Vision3): no anti-halation backing at all
for src, name in [("kodak_vision3_500t", "Vision3 500T, no rem-jet (CineStill 800T-style)"),
                  ("kodak_vision3_50d", "Vision3 50D, no rem-jet (CineStill 50D-style)")]:
    st = pickle.load(open(f"stocks/{src}.pkl", "rb"))
    st = copy.deepcopy(st)
    st.halation, st.halation_um = (0.16, 0.035, 0.0), 220.0
    st.display_name, st.rms_source = name, st.rms_source + "; halation: no anti-halation layer"
    st._print_cache = None
    pickle.dump(st, open(f"stocks/{src}_no_remjet.pkl", "wb"))
    print("OK", name, flush=True)
