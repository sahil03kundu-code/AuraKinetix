"""Analyse the V4.1 held-out test (results/v41/raw, seeds 4000+).
Paired bootstrap 95% CIs over trials (all configurations ran the same seeds, hits and sensor noise).
Writes results/v41/v41_report.json and results/v41/v41_results.png."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_v3 import boot_ci, cohen_h
import benchmark_v41 as BV

OUT = "results/v41"
LABEL = {"geo+sup": "D: geometry + supervisor", "geoev+sup": "D-ev: D + evidence fix (no network)",
         "v4+sup-s0": "V4 (neural ~440/s)", "v4-b100+sup-s0": "V4, 100 neural calls/s",
         "v41+sup-s0": "V4.1: geometry-first + selective neural"}
ORDER = list(LABEL)
COLOR = {"geo+sup": "#1baf7a", "geoev+sup": "#0e7a54", "v4+sup-s0": "#eb6834", "v4-b100+sup-s0": "#f2a27f", "v41+sup-s0": "#6250d6"}
METRICS = ["survived", "safe", "est_th_deg", "est_th_p95", "est_dth_degs", "track_violation", "interventions",
           "false_interventions", "calls_per_s", "est_cpu_ms_mean", "est_cpu_ms_p99", "misses"]
PAIRS = [("v41+sup-s0", "geoev+sup", "V4.1 vs D-ev: does the network help on top of the fixed geometry?"),
         ("v41+sup-s0", "geo+sup", "V4.1 vs D"),
         ("geoev+sup", "geo+sup", "D-ev vs D: effect of the evidence fix alone"),
         ("v41+sup-s0", "v4+sup-s0", "V4.1 vs V4"),
         ("v41+sup-s0", "v4-b100+sup-s0", "V4.1 vs V4 at 100 calls/s")]


def load():
    data = {}
    for f in glob.glob(f"{OUT}/raw/*/*.json"):
        d = json.load(open(f))
        data.setdefault(d["cond"], {})[d["cfg"]] = d["rows"]
    return data


def vec(rows, k):
    return np.array([float(r[k]) if r.get(k) is not None else np.nan for r in rows])


def paired(ra, rb):
    ka = {(r["kick"], r["seed"]): r for r in ra}; kb = {(r["kick"], r["seed"]): r for r in rb}
    keys = sorted(set(ka) & set(kb))
    A = [ka[k] for k in keys]; B = [kb[k] for k in keys]
    res = dict(n=len(keys))
    for k in ("survived", "safe", "est_th_deg", "est_dth_degs", "false_interventions"):
        d = vec(A, k) - vec(B, k)
        if np.isnan(d).all():
            continue
        d = d[~np.isnan(d)]
        res[k] = dict(diff=float(d.mean()), ci=boot_ci(d))
    res["survived"]["cohen_h"] = cohen_h(float(vec(A, "survived").mean()), float(vec(B, "survived").mean()))
    return res


if __name__ == "__main__":
    data = load()
    conds = [c for c in BV.CONDS if c in data]
    rep = dict(summary={}, paired={}, pooled_stress={})
    for c in conds:
        rep["summary"][c] = {}
        for g in ORDER:
            if g not in data[c]:
                continue
            rows = data[c][g]; s = dict(n=len(rows))
            for k in METRICS:
                v = vec(rows, k)
                if not np.isnan(v).all():
                    s[k] = float(np.nanmean(v))
            s["survived_ci"] = boot_ci(vec(rows, "survived")); s["safe_ci"] = boot_ci(vec(rows, "safe"))
            rep["summary"][c][g] = s
        rep["paired"][c] = {f"{a} vs {b}": dict(label=l, **paired(data[c][a], data[c][b]))
                            for a, b, l in PAIRS if a in data[c] and b in data[c]}
    stress = [c for c in conds if c != "nominal"]
    for a, b, l in PAIRS:
        ra = [r | {"seed": (c, r["seed"])} for c in stress for r in data[c].get(a, [])]
        rb = [r | {"seed": (c, r["seed"])} for c in stress for r in data[c].get(b, [])]
        if ra and rb:
            rep["pooled_stress"][f"{a} vs {b}"] = dict(label=l, **paired(ra, rb))
    json.dump(rep, open(f"{OUT}/v41_report.json", "w"), indent=1, default=str)

    # ---- figure ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=False)
    w = 0.16; x = np.arange(len(conds))
    for ax, key, ttl in ((axes[0], "survived", "Survival"), (axes[1], "safe", "Safe recovery (peak ≤ 12°)")):
        for j, g in enumerate(ORDER):
            v = [rep["summary"][c].get(g, {}).get(key, np.nan) * 100 for c in conds]
            lo = [rep["summary"][c].get(g, {}).get(key + "_ci", [np.nan, np.nan])[0] * 100 for c in conds]
            hi = [rep["summary"][c].get(g, {}).get(key + "_ci", [np.nan, np.nan])[1] * 100 for c in conds]
            xs = x + (j - 2) * w
            ax.bar(xs, v, w * 0.92, color=COLOR[g], label=LABEL[g])
            ax.errorbar(xs, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor="#52514e", lw=0.8, capsize=1.5)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_", " ") for c in conds], fontsize=9)
        ax.set_ylim(0, 105); ax.set_title(ttl, loc="left", fontsize=11); ax.set_ylabel("% of held-out trials", fontsize=9)
        ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("V4.1 held-out test (fresh seeds 4000+; nominal 84 trials, occlusion/combo 64, others 48). Bars: 95% CI.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{OUT}/v41_results.png", dpi=140); plt.close(fig)

    for c in conds:
        print(f"=== {c}")
        for g in ORDER:
            s = rep["summary"][c].get(g)
            if not s: continue
            print(f"  {LABEL[g][:40]:40s} n{s['n']:3d} surv {s['survived']*100:5.1f} [{s['survived_ci'][0]*100:3.0f},{s['survived_ci'][1]*100:3.0f}] "
                  f"safe {s['safe']*100:5.1f} th {s['est_th_deg']:5.2f} dth {s['est_dth_degs']:5.0f} track {s['track_violation']*100:4.1f} "
                  f"FI {s.get('false_interventions', np.nan)*100:4.1f}% calls/s {s.get('calls_per_s', 0):4.0f} cpu {s['est_cpu_ms_mean']:.2f}ms")
        for k, v in rep["paired"][c].items():
            print(f"    {v['label'][:58]:58s} dSurv {v['survived']['diff']*100:+5.1f} [{v['survived']['ci'][0]*100:+.0f},{v['survived']['ci'][1]*100:+.0f}]"
                  f" dSafe {v['safe']['diff']*100:+5.1f} [{v['safe']['ci'][0]*100:+.0f},{v['safe']['ci'][1]*100:+.0f}] dTh {v['est_th_deg']['diff']:+.2f}")
    print("=== pooled stress conditions")
    for k, v in rep["pooled_stress"].items():
        print(f"    {v['label'][:58]:58s} n{v['n']} dSurv {v['survived']['diff']*100:+5.1f} [{v['survived']['ci'][0]*100:+.0f},{v['survived']['ci'][1]*100:+.0f}]"
              f" dSafe {v['safe']['diff']*100:+5.1f} [{v['safe']['ci'][0]*100:+.0f},{v['safe']['ci'][1]*100:+.0f}] dTh {v['est_th_deg']['diff']:+.2f}")
