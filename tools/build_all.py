import sys, pickle, time, os
import filmsim3 as f3
from stocks_table import STOCKS
P = "ext/spektrafilm/src/spektrafilm/data/profiles/"
os.makedirs("stocks", exist_ok=True)
for key in sys.argv[1:]:
    name, rms, src = STOCKS[key]
    t = time.time()
    try:
        st = f3.from_spektrafilm(P + key + ".json", verbose=False)
        st = f3.fit_granularity(st, rms, verbose=False)
        r, _ = f3.measure_rms(st, channels=(1,))
        st.display_name, st.rms_target, st.rms_source = name, rms, src
        pickle.dump(st, open(f"stocks/{key}.pkl", "wb"))
        print(f"OK  {name:<32} curve err {st.fit_rms:.4f}  rms {r[0]:.2f}/{rms}  {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"ERR {name}: {e!r}", flush=True)
