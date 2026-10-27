"""Analyse the V4 benchmark (spec sections 9-11).

Merges results/v4/raw (new runs) with results/v3/raw (unchanged V3 configurations on the
same test trials); a V4 re-run of a configuration overrides the V3 file (used where V4
fixed the simulation: frame camera under 'ood_plant').
Statistics: family = configuration over its training seeds; paired bootstrap CIs over
trials restricted to the seeds both sides share; Cohen's h (rates), d_z (peak tilt).
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_v3 import family_of, boot_ci, cohen_h, _seed
import benchmark_v3 as B3
import benchmark_v4 as B4

OUT = "results/v4"
CONDS = ["nominal"] + B3.STRESS + B4.NEW_SCENARIOS
LABEL = {
    "ev-dense": "A  V3 event, dense", "ev-trig": "B  V3 event, triggered", "ev-trig+shield": "C  V3 + uncertainty + shield",
    "geo": "D  event geometry (no network)", "geo+sup": "D  event geometry + supervisor",
    "frame120": "E  learned frame 120 fps", "frame30": "E  learned frame 30 fps",
    "imu": "F  IMU + encoder EKF", "imu+shield": "F  IMU + encoder + shield", "oracle": "F  oracle (perfect state)",
    "v4+sup": "V4 (proposed)", "v4": "V4 without supervisor", "v4-nocorr+sup": "V4 without state correction",
    "v4-noinfl+sup": "V4 without uncertainty feedback", "v4-dense+sup": "V4 without triggered scheduling",
    "eventonly-v4+sup": "V4 event-only (no APS)", "v4-ow+sup": "V4 first design (neural overwrite)",
    "v4-b250+sup": "V4, 250 neural corrections/s", "v4-b100+sup": "V4, 100 neural corrections/s",
}
KEEP = list(LABEL)
METRICS = ["survived", "safe", "peak_deg", "est_th_deg", "est_th_median", "est_th_max", "frac_err_gt10", "wrong_sign",
           "track_violation", "sat_frac", "effort", "max_du", "calls_per_s", "miss_rate"]


def load(cond):
    files = {}
    for root in ("results/v3/raw", "results/v4/raw"):          # v4 overrides v3 for the same configuration
        for f in glob.glob(f"{root}/{cond}/*.json"):
            files[os.path.basename(f)] = f
    fams = {}
    for f in files.values():
        d = json.load(open(f))
        fam = family_of(d["cfg"])
        if fam in KEEP:
            fams.setdefault(fam, {})[d["cfg"]] = d["rows"]
    return fams


def arr(runs, key):
    a = []
    for rows in runs.values():
        a.append([float(r.get(key) if r.get(key) is not None else np.nan) for r in rows])
    return np.array(a)


def summarise(fams):
    out = {}
    for fam, runs in fams.items():
        row = dict(n_seeds=len(runs), n_trials=len(next(iter(runs.values()))))
        for key in METRICS:
            a = arr(runs, key)
            if np.isnan(a).all():
                continue
            per_trial = np.nanmean(a, 0)
            row[key] = float(np.nanmean(per_trial))
            if key in ("survived", "safe"):
                row[key + "_ci"] = boot_ci(per_trial)
                row[key + "_seed_range"] = [float(np.nanmean(x)) for x in (a.min(0), a.max(0))] if len(runs) > 1 else None
        out[fam] = row
    return out


def compare(fams, a, b):
    if a not in fams or b not in fams:
        return None
    ra, rb = fams[a], fams[b]
    sa = {_seed(c) for c in ra}; sb = {_seed(c) for c in rb}
    if None not in sa and None not in sb:
        common = sa & sb
        if not common:
            return None
        ra = {c: v for c, v in ra.items() if _seed(c) in common}; rb = {c: v for c, v in rb.items() if _seed(c) in common}
        seeds = sorted(common)
    else:
        seeds = "all"
    res = dict(seeds=seeds)
    for key in ("survived", "safe", "peak_deg", "est_th_deg"):
        xa, xb = np.nanmean(arr(ra, key), 0), np.nanmean(arr(rb, key), 0)
        d = xa - xb
        res[key] = dict(diff=float(d.mean()), ci=boot_ci(d))
        if key in ("survived", "safe"):
            res[key]["cohen_h"] = cohen_h(float(np.clip(xa.mean(), 0, 1)), float(np.clip(xb.mean(), 0, 1)))
        else:
            res[key]["d_z"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
    return res


COMPARISONS = [
    ("v4+sup", "ev-dense", "V4 vs A (V3 event, dense)"),
    ("v4+sup", "ev-trig", "V4 vs B (V3 event, triggered)"),
    ("v4+sup", "ev-trig+shield", "V4 vs C (V3 + uncertainty + shield)"),
    ("v4+sup", "geo+sup", "V4 vs D (event geometry + supervisor)"),
    ("v4+sup", "geo", "V4 vs D (event geometry, no supervisor)"),
    ("v4+sup", "frame120", "V4 vs E (learned frame 120 fps)"),
    ("imu", "v4+sup", "F (IMU + encoder) vs V4"),
    ("v4+sup", "v4", "Ablation: safety supervisor"),
    ("v4+sup", "v4-nocorr+sup", "Ablation: state correction"),
    ("v4+sup", "v4-noinfl+sup", "Ablation: uncertainty feedback"),
    ("v4+sup", "v4-dense+sup", "Ablation: triggered vs dense scheduling"),
    ("v4+sup", "eventonly-v4+sup", "Ablation: event+APS vs event-only"),
    ("v4+sup", "v4-ow+sup", "Ablation: position-only vs overwrite neural fusion"),
    ("v4+sup", "v4-b250+sup", "Budget: unlimited (~450/s) vs 250/s"),
    ("v4+sup", "v4-b100+sup", "Budget: unlimited (~450/s) vs 100/s"),
]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def colour(f):
    if f.startswith("v4") or f.startswith("eventonly-v4"):
        return "#eb6834"
    if f.startswith("geo"):
        return "#1baf7a"
    if f.startswith("frame"):
        return "#8a8984"
    if f.startswith("imu") or f.startswith("oracle"):
        return "#6250d6"
    return "#2a78d6"


def style(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_nominal(summ):
    fams = [f for f in KEEP if f in summ]
    fams.sort(key=lambda f: summ[f]["survived"] + 0.01 * summ[f]["safe"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 0.36 * len(fams) + 1.8), sharey=True)
    y = np.arange(len(fams))
    for ax, key, ttl in ((axes[0], "survived", "Survival (pole stays up)"), (axes[1], "safe", "Safe recovery (peak ≤ 12°)")):
        style(ax)
        for i, f in enumerate(fams):
            v = summ[f][key] * 100; lo, hi = [c * 100 for c in summ[f][key + "_ci"]]
            ax.plot([lo, hi], [i, i], color=colour(f), lw=2, solid_capstyle="round")
            ax.plot(v, i, "o", color=colour(f), ms=7, markeredgecolor="white", markeredgewidth=1.2)
            if summ[f].get(key + "_seed_range"):
                r = [c * 100 for c in summ[f][key + "_seed_range"]]
                ax.plot(r, [i - 0.28] * 2, color=colour(f), lw=1, alpha=0.5)
        ax.set_xlim(-2, 102); ax.set_title(ttl, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("% of test trials (dot = mean, bar = 95% CI, thin = seed range)", fontsize=8, color=INK2)
    ax = axes[2]; style(ax)
    for i, f in enumerate(fams):
        ax.barh(i, summ[f]["est_th_deg"], color=colour(f), height=0.6)
        ax.text(summ[f]["est_th_deg"], i, f" {summ[f]['est_th_deg']:.2f}°", va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_title("Closed-loop tilt-estimate error", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("mean |estimated − true tilt| (deg, log scale)", fontsize=8, color=INK2)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[f] for f in fams], fontsize=8.5, color=INK)
    fig.suptitle("V4 vs baselines and ablations, nominal conditions (7 hit strengths × 16 test trials per seed)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/v4_nominal.png", dpi=140); plt.close(fig)


ENV = ["ev-dense", "ev-trig", "geo+sup", "frame120", "imu", "v4+sup"]


def plot_envelope(report):
    conds = [c for c in CONDS if c in report]
    M = np.array([[report[c].get(f, {}).get("survived", np.nan) * 100 for f in ENV] for c in conds])
    fig, ax = plt.subplots(figsize=(11, 0.46 * len(conds) + 2.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(ENV)):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                    color=INK2 if np.isnan(M[i, j]) else ("white" if M[i, j] > 60 else INK))
    ax.set_xticks(range(len(ENV))); ax.set_xticklabels([LABEL[f].replace("  ", "\n", 1) for f in ENV], fontsize=8, color=INK)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=8.5, color=INK)
    ax.axhline(len(B3.STRESS) + 0.5, color="white", lw=3)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("survival %", fontsize=8, color=INK2)
    ax.set_title("Survival % per condition. Nominal 112 trials/seed; others 16 trials/seed (hits 1.5 and 3.0 rad/s; fast: 4 and 5).\n"
                 "Rows below the white line are the new V4 scenarios. Columns pool their seeds (A, E: 3 seeds in nominal and\n"
                 "most stress rows; V4: 3 seeds nominal, seed 0 elsewhere).", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/v4_envelope.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    report = dict(conditions={}, comparisons={})
    for cond in CONDS:
        fams = load(cond)
        if not fams:
            continue
        report["conditions"][cond] = summarise(fams)
        report["comparisons"][cond] = {f"{a} vs {b}": dict(label=l, **c) for a, b, l in COMPARISONS
                                       if (c := compare(fams, a, b)) is not None}
    json.dump(report, open(f"{OUT}/v4_report.json", "w"), indent=1)
    if "nominal" in report["conditions"]:
        plot_nominal(report["conditions"]["nominal"])
    plot_envelope(report["conditions"])
    for cond, rows in report["conditions"].items():
        print(f"=== {cond}")
        for f in sorted(rows, key=lambda f: -rows[f]["survived"]):
            r = rows[f]
            print(f"  {LABEL.get(f, f)[:38]:38s} n{r['n_seeds']} surv {r['survived']*100:5.1f} [{r['survived_ci'][0]*100:.0f},{r['survived_ci'][1]*100:.0f}]"
                  f" safe {r['safe']*100:5.1f} est {r.get('est_th_deg', np.nan):5.2f} >10° {r.get('frac_err_gt10', np.nan)*100:5.1f}%"
                  f" wrongsign {r.get('wrong_sign', np.nan)*100:4.1f}% track {r.get('track_violation', np.nan)*100:4.1f}%"
                  + (f" calls/s {r['calls_per_s']:.0f}" if "calls_per_s" in r else ""))
