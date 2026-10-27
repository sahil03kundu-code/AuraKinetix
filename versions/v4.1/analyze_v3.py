"""Summarise the v3 benchmark: families pooled over training seeds, paired bootstrap CIs,
effect sizes, operating envelope, figures.

Statistics
  * A family = one configuration trained with seeds 0/1/2 (or a single unseeded one).
    Family metric = mean over its seeds; seed spread = min..max of per-seed means.
  * Paired comparison A vs B: per test trial i (same hit, same initial state, same
    sensor noise seed), d_i = mean over A's seeds - mean over B's seeds; 95% CI by
    bootstrap over trials (5000 resamples).
  * Effect sizes: Cohen's h for survival / safe rates, paired Cohen's d_z for peak tilt.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_v3 import STRESS, RAW, OUT

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"oracle": "#8a8984", "imu": "#1baf7a", "event": "#2a78d6", "eventonly": "#6250d6", "frame": "#eb6834"}
LABEL = {
    "oracle": "Oracle (perfect state)", "oracle+shield": "Oracle + shield",
    "imu": "IMU + encoder EKF", "imu+shield": "IMU + encoder EKF + shield",
    "ev-dense": "Event, dense (v2-style)", "ev-dense+shield": "Event, dense + shield",
    "ev-trig": "Event, triggered", "ev-trig+shield": "Event, triggered + shield (full v3)",
    "ev-periodic+shield": "Event, fixed 2 ms schedule + shield",
    "ens-dense+shield": "Event 3-seed ensemble, dense + shield",
    "eventonly-dense": "Event-only (no APS), dense", "eventonly-trig+shield": "Event-only, triggered + shield",
    "frame120": "Learned frame 120 fps", "frame60": "Learned frame 60 fps", "frame30": "Learned frame 30 fps",
    "frame120+shield": "Learned frame 120 fps + shield", "frame30+shield": "Learned frame 30 fps + shield",
}


def family_of(cfg):
    base, _, sfx = cfg.rpartition("-s")
    return base if (base and sfx.isdigit()) else cfg


def colour(fam):
    for k in ("eventonly", "oracle", "imu", "frame"):
        if fam.startswith(k):
            return COL[k]
    return COL["event"]


def load(cond):
    fams = {}
    for f in sorted(glob.glob(f"{RAW}/{cond}/*.json")):
        d = json.load(open(f))
        fams.setdefault(family_of(d["cfg"]), {})[d["cfg"]] = d["rows"]
    return fams


def per_trial(runs, key):
    """(n_trials,) mean over seeds of a per-trial metric."""
    arr = np.array([[float(r[key]) for r in rows] for rows in runs.values()])
    return arr.mean(0), arr.mean(1)


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cohen_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def summarise(fams):
    out = {}
    for fam, runs in fams.items():
        s, s_seed = per_trial(runs, "survived")
        sf, sf_seed = per_trial(runs, "safe")
        pk, _ = per_trial(runs, "peak_deg")
        et, et_seed = per_trial(runs, "est_th_deg")
        row = dict(n_trials=len(s), n_seeds=len(runs), survival=float(s.mean()), survival_ci=boot_ci(s),
                   survival_seed_range=[float(s_seed.min()), float(s_seed.max())],
                   safe=float(sf.mean()), safe_ci=boot_ci(sf), safe_seed_range=[float(sf_seed.min()), float(sf_seed.max())],
                   peak_deg=float(pk.mean()), est_th_deg=float(et.mean()),
                   est_th_seed_range=[float(et_seed.min()), float(et_seed.max())])
        rows = [r for rr in runs.values() for r in rr]
        if "calls_per_s" in rows[0]:
            row.update(calls_per_s=float(np.mean([r["calls_per_s"] for r in rows])),
                       miss_rate=float(np.mean([r["miss_rate"] for r in rows])))
        if "shield_modified" in rows[0]:
            row.update(shield_modified=float(np.mean([r["shield_modified"] for r in rows])),
                       shield_infeasible=float(np.mean([r["shield_infeasible"] for r in rows])),
                       shield_stale=float(np.mean([r["shield_stale"] for r in rows])),
                       ctrl_p99_ms=float(np.percentile([r["ctrl_p99_ms"] for r in rows], 50)),
                       ctrl_max_ms=float(np.max([r["ctrl_max_ms"] for r in rows])))
        out[fam] = row
    return out


def _seed(cfg):
    base, _, sfx = cfg.rpartition("-s")
    return int(sfx) if (base and sfx.isdigit()) else None


def compare(fams, a, b):
    """Paired comparison. When both families are seeded, only the training seeds they
    share are used (a seed-0-only ablation is compared against seed 0, not the 3-seed mean)."""
    if a not in fams or b not in fams:
        return None
    ra, rb = fams[a], fams[b]
    sa = {_seed(c) for c in ra}; sb = {_seed(c) for c in rb}
    if None not in sa and None not in sb:
        common = sa & sb
        if not common:
            return None
        ra = {c: v for c, v in ra.items() if _seed(c) in common}
        rb = {c: v for c, v in rb.items() if _seed(c) in common}
    res = {"seeds": sorted(x for x in (sa & sb) if x is not None) if (None not in sa and None not in sb) else "all"}
    for key in ("survived", "safe", "peak_deg"):
        xa, _ = per_trial(ra, key); xb, _ = per_trial(rb, key)
        d = xa - xb
        res[key] = dict(diff=float(d.mean()), ci=boot_ci(d))
        if key == "peak_deg":
            res[key]["d_z"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
        else:
            res[key]["cohen_h"] = cohen_h(xa.mean(), xb.mean())
    return res


COMPARISONS = [
    ("ev-trig+shield", "ev-dense", "Full v3 vs v2-style event stack"),
    ("ev-dense+shield", "ev-dense", "Shield on the event stack (B)"),
    ("ev-trig+shield", "ev-dense+shield", "Triggered vs dense correction, both shielded (C)"),
    ("ev-trig+shield", "ev-periodic+shield", "Trigger policy vs periodic at matched compute (C)"),
    ("ens-dense+shield", "ev-dense+shield", "3-seed ensemble vs single model, both shielded (A)"),
    ("frame120+shield", "frame120", "Shield on the 120 fps frame stack (B)"),
    ("ev-trig+shield", "frame120", "Full v3 vs learned frame 120 fps"),
    ("ev-trig+shield", "frame30", "Full v3 vs learned frame 30 fps"),
    ("ev-dense", "frame120", "Dense event vs learned frame 120 fps"),
    ("ev-trig", "ev-dense", "Triggered vs dense correction, no shield (C)"),
    ("ev-trig", "frame120", "Triggered event vs learned frame 120 fps"),
    ("ev-dense", "eventonly-dense", "APS frame channel: event+APS vs event-only (D)"),
    ("imu", "ev-trig+shield", "IMU + encoder vs full v3"),
    ("imu", "frame120", "IMU + encoder vs learned frame 120 fps"),
]
ENVELOPE_FAMS = ["oracle", "imu", "ev-dense", "ev-trig", "eventonly-dense", "frame120", "frame30"]


def style(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8)


def plot_nominal(summ):
    fams = sorted(summ, key=lambda f: summ[f]["survival"] + 0.01 * summ[f]["safe"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.36 * len(fams) + 1.6), sharey=True)
    y = np.arange(len(fams))
    for ax, key, ttl in ((axes[0], "survival", "Survival (pole stays up)"),
                         (axes[1], "safe", "Safe recovery (peak tilt ≤ 12°)")):
        style(ax)
        for i, f in enumerate(fams):
            v, (lo, hi) = summ[f][key] * 100, [c * 100 for c in summ[f][key + "_ci"]]
            ax.plot([lo, hi], [i, i], color=colour(f), lw=2, solid_capstyle="round")
            ax.plot(v, i, "o", color=colour(f), ms=7, markeredgecolor="white", markeredgewidth=1.2)
            if summ[f]["n_seeds"] > 1:
                r = [c * 100 for c in summ[f][key + "_seed_range"]]
                ax.plot(r, [i - 0.28, i - 0.28], color=colour(f), lw=1, alpha=0.5)
        ax.set_xlim(-2, 102); ax.set_title(ttl, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("% of test trials (dot = mean, bar = 95% CI, thin = seed range)", fontsize=8, color=INK2)
    ax = axes[2]; style(ax)
    for i, f in enumerate(fams):
        ax.barh(i, summ[f]["est_th_deg"], color=colour(f), height=0.6)
        ax.text(summ[f]["est_th_deg"], i, f" {summ[f]['est_th_deg']:.2f}°", va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_title("Closed-loop tilt-estimate error", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("mean |estimated − true tilt| (deg, log scale)", fontsize=8, color=INK2)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL.get(f, f) for f in fams], fontsize=8.5, color=INK)
    fig.suptitle("Nominal conditions: 7 hit strengths × 16 trials per seed. Families with a thin seed-range line "
                 "pool 3 training seeds; the rest are seed 0 or unseeded",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/v3_nominal.png", dpi=140); plt.close(fig)


def plot_envelope(env):
    conds = [c for c in ["nominal"] + STRESS if c in env]
    fams = [f for f in ENVELOPE_FAMS if any(f in env[c] for c in conds)]
    M = np.array([[env[c].get(f, {}).get("survival", np.nan) * 100 for f in fams] for c in conds])
    fig, ax = plt.subplots(figsize=(1.35 * len(fams) + 3.2, 0.5 * len(conds) + 2.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(fams)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > 60 else INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([LABEL.get(f, f).replace(", ", ",\n").replace(" + ", "\n+ ") for f in fams],
                                                       fontsize=7.5, color=INK)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("survival %", fontsize=8, color=INK2)
    ax.set_title("Operating envelope: survival % per condition\n"
                 "Nominal: 7 hit levels x 16 trials. Stress: hits 1.5 and 3.0 rad/s (fast: 4 and 5) x 8 trials, so rows\n"
                 "are not comparable with each other. Event dense and frame 120 fps pool 3 seeds; others seed 0. "
                 "Oracle n/a where only the sensor changes.", fontsize=9, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/v3_envelope.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    report = dict(conditions={}, comparisons={}, envelope={})
    for cond in ["nominal"] + STRESS:
        fams = load(cond)
        if not fams:
            continue
        report["conditions"][cond] = summarise(fams)
        report["comparisons"][cond] = {f"{a} vs {b}": dict(label=lbl, **(compare(fams, a, b) or {}))
                                       for a, b, lbl in COMPARISONS if compare(fams, a, b)}
        report["envelope"][cond] = {f: report["conditions"][cond][f] for f in ENVELOPE_FAMS if f in fams}
    # who wins each condition (camera families only), with a significance flag from the paired CI
    wins = {}
    for cond, comps in report["comparisons"].items():
        c = comps.get("ev-dense vs frame120")
        if c:
            d, (lo, hi) = c["survived"]["diff"], c["survived"]["ci"]
            wins[cond] = dict(event_minus_frame120_survival=d, ci=[lo, hi],
                              verdict="event wins" if lo > 0 else "frame wins" if hi < 0 else "no significant difference")
    report["event_vs_frame120"] = wins
    json.dump(report, open(f"{OUT}/v3_report.json", "w"), indent=1)
    if "nominal" in report["conditions"]:
        plot_nominal(report["conditions"]["nominal"])
    if report["envelope"]:
        plot_envelope(report["envelope"])
    for cond, rows in report["conditions"].items():
        print(f"=== {cond}")
        for f, r in sorted(rows.items(), key=lambda kv: -kv[1]["survival"]):
            extra = f" calls/s {r['calls_per_s']:.0f} miss {r['miss_rate']*100:.1f}%" if "calls_per_s" in r else ""
            print(f"  {f:26s} surv {r['survival']*100:5.1f} [{r['survival_ci'][0]*100:.0f},{r['survival_ci'][1]*100:.0f}] "
                  f"safe {r['safe']*100:5.1f} peak {r['peak_deg']:5.1f} est {r['est_th_deg']:5.2f}{extra}")
    for cond, w in wins.items():
        print(f"{cond:16s} event-frame120 survival {w['event_minus_frame120_survival']*100:+.0f} pts "
              f"[{w['ci'][0]*100:+.0f},{w['ci'][1]*100:+.0f}] -> {w['verdict']}")
