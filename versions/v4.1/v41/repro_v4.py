"""Step 2: re-run stored V4 TEST configurations and check they reproduce exactly.
Nominal: first 2 trials per hit (TRIALS=2), occlusion: all 16 trials; Baseline D and V4."""
import os, sys, json, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ["TRIALS"] = "2"
from multiprocessing import Pool
import numpy as np
import benchmark_v4 as B4
TMP = tempfile.mkdtemp(prefix="repro_", dir="/tmp/claude-0/-home-claude/092a0d58-f4fa-5ded-9401-5638676f1bf6/scratchpad")
B4.RAW4 = TMP

def job(a):
    return a, B4.run_one(a)

if __name__ == "__main__":
    jobs = [(c, g) for c in ("nominal", "occlusion") for g in ("geo+sup", "v4+sup-s0")]
    out = {}
    with Pool(2) as pool:
        for (c, g), path in pool.imap_unordered(job, jobs):
            new = {(r["kick"], r["trial"]): r for r in json.load(open(path))["rows"]}
            old = {(r["kick"], r["trial"]): r for r in json.load(open(f"results/v4/raw/{c}/{g}.json"))["rows"]}
            keys = sorted(set(new) & set(old))
            exact = all(new[k]["survived"] == old[k]["survived"] and abs(new[k]["peak_deg"] - old[k]["peak_deg"]) < 1e-9
                        and abs(new[k]["est_th_deg"] - old[k]["est_th_deg"]) < 1e-9 for k in keys)
            out[f"{c}|{g}"] = dict(trials=len(keys), exact=exact,
                                   survival_new=float(np.mean([new[k]["survived"] for k in keys])),
                                   survival_stored=float(np.mean([old[k]["survived"] for k in keys])))
            print(c, g, out[f"{c}|{g}"], flush=True)
    os.makedirs("results/v41", exist_ok=True)
    json.dump(out, open("results/v41/repro_v4.json", "w"), indent=1)
