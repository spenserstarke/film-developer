import glob, pickle, filmsim3 as f3
for p in sorted(glob.glob("stocks/*.pkl")) + sorted(glob.glob("stocks_experimental/*.pkl")):
    st = pickle.load(open(p, "rb")); f3.build_spectral_lut(st); st._print_cache = None
    pickle.dump(st, open(p, "wb")); print("OK", p, flush=True)
