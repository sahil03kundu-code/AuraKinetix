"""Failure-case visualisation (development set, validation seeds).

For representative falls of the V3 observer: true tilt, V3 estimate with its
+-2 sigma band, the independent event-geometry measurements, the V4 corrected
estimate (open-loop replay), and the detector score with its alarm.
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from aurasync.v4.geometry import EventGeometry

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_TRUE, C_V3, C_V4, C_GEO, C_DET = "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def pick_cases(n=3):
    cat = json.load(open("results/v4/dev_catalogue.json"))
    falls = [c for c in cat if c["file"].startswith("ev-trig-s0_") and c["fail"] >= 0]
    falls.sort(key=lambda c: -c["max_err_over_sigma"])
    chosen, kicks = [], set()
    for c in falls:                                   # distinct hit strengths, most confident errors first
        k = c["file"].rsplit("_", 1)[1]
        if k not in kicks:
            chosen.append(c["file"]); kicks.add(k)
        if len(chosen) == n:
            break
    return chosen


if __name__ == "__main__":
    rep = np.load("results/v4/dev_replay.npz")
    cases = pick_cases()
    geo = EventGeometry()
    fig, axes = plt.subplots(2, len(cases), figsize=(5.2 * len(cases), 6.4), sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 1.2]))
    for c, f in enumerate(cases):
        d = dict(np.load(f"data/v4dev/{f}"))
        s = d["s"]; t = (np.arange(len(s)) + 1)
        e3, s3 = rep[f"{f}|v3|est"], rep[f"{f}|v3|sig"]; e4 = rep[f"{f}|full|est"]; det = rep[f"{f}|v3|det"]
        gm = [geo.measure(d["surf"][k].astype(np.float32), e3[k][0]) for k in range(len(s))]
        gk = [k for k, m in enumerate(gm) if m is not None and m["valid"]]
        ax = axes[0, c]
        ax.fill_between(t, np.degrees(e3[:, 2] - 2 * s3[:, 2]), np.degrees(e3[:, 2] + 2 * s3[:, 2]), color=C_V3, alpha=0.15, lw=0,
                        label="V3 ±2σ")
        ax.plot(t, np.degrees(s[:, 2]), color=C_TRUE, lw=2, label="true tilt")
        ax.plot(t, np.degrees(e3[:, 2]), color=C_V3, lw=1.6, label="V3 estimate")
        ax.plot(np.array(gk) + 1, [np.degrees(gm[k]["theta"]) for k in gk], ".", color=C_GEO, ms=2.2, label="event geometry")
        ax.plot(t, np.degrees(e4[:, 2]), color=C_V4, lw=1.6, ls="--", label="V4 corrected (replay)")
        ax.axvline(200, color=INK2, lw=0.8, ls=":"); ax.text(204, 0.92, "hit", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2)
        if d["fail"] >= 0:
            ax.axvline(int(d["fail"]), color="#e34948", lw=1); ax.text(int(d["fail"]) - 4, 0.92, "falls", ha="right",
                                                                       transform=ax.get_xaxis_transform(), fontsize=8, color="#e34948")
        ax.set_ylim(-60, 60); ax.set_title(f"hit {float(d['kick']):+.1f} rad/s (validation seed {f.split('_')[1]})", fontsize=10,
                                           loc="left", color=INK)
        ax2 = axes[1, c]
        ax2.plot(t, det[:, 0], color=C_DET, lw=1.4)
        ax2.axhline(1.0, color=INK2, lw=0.8, ls=(0, (2, 3)))
        al = np.nonzero(det[:, 1] > 0)[0]
        if len(al):
            ax2.axvline(al[0] + 1, color=C_DET, lw=0.8, ls="--")
            ax2.text(al[0] + 4, 0.8, f"alarm {al[0]+1} ms", transform=ax2.get_xaxis_transform(), fontsize=8, color=C_DET)
        ax2.set_ylim(0, max(2.5, min(det[:, 0].max() * 1.1, 6)))
        ax2.set_xlabel("time (ms)", fontsize=8, color=INK2)
        for a in (ax, ax2):
            a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
            a.tick_params(colors=INK2, labelsize=8)
        if c == 0:
            ax.set_ylabel("pole tilt (deg)", fontsize=9, color=INK2); ax2.set_ylabel("detector score\n(1 = alarm)", fontsize=9, color=INK2)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("Confidently wrong V3 estimates before a fall, the independent event geometry, and V4's replayed correction",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig("results/v4/fig_failure_cases.png", dpi=140)
    print("saved", cases)
