"""V4.1 failure cases on HELD-OUT trials (deterministic re-runs of stored configurations).
For each case: true vs estimated tilt (+-2 sigma), the evidence state per tick, neural-call ticks,
and supervisor interventions. Also writes per-tick CSV logs. -> results/v41/fig_v41_cases.png"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)
import benchmark_v41 as BV, benchmark_v4 as B4
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.safety import SafetyShield
from aurasync.simulate import run_episode
from aurasync.v41.pipeline import STATE_NAMES

OUT = "results/v41/cases"; os.makedirs(OUT, exist_ok=True)
SC = ["#d9d8d4", "#f2a27f", "#1baf7a", "#6250d6", "#d03b3b", "#0b0b0b"]

def first_fail(cond, cfg, other=None):
    rows = json.load(open(f"results/v41/raw/{cond}/{cfg}.json"))["rows"]
    alt = {(r["kick"], r["seed"]): r["survived"] for r in json.load(open(f"results/v41/raw/{cond}/{other}.json"))["rows"]} if other else {}
    for r in rows:      # prefer a trial the comparison system survived
        if not r["survived"] and alt.get((r["kick"], r["seed"]), False):
            return r
    return next(r for r in rows if not r["survived"])

def rerun(cond_name, cfg, row):
    cond, plant, snap, kicks, n, fault = BV.spec(cond_name)
    pipe, _ = BV.build(cfg); calls = []
    if pipe.models is not None:
        f = pipe._neural
        def rec(*a):
            calls.append(pipe.k); return f(*a)
        pipe._neural = rec
    p = PlantParams(); sup = BV.v41_tuning()["supervisor_by_family"]["v41" if cfg.startswith("v41") else "geoev"]
    ctrl = BV.RecShield(LQRController(p), SafetyShield(p, **sup), pipe)
    seed = row["seed"]; i = seed - 4000
    rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
    lg = run_episode(plant, ctrl, pipe, BV.N_TICKS, s0, kicks=[(BV.KICK_TICK, row["kick"] * (1 if i % 2 == 0 else -1))],
                     snap=snap, cam_seed=seed, cond=cond)
    return lg, pipe, calls, np.array(ctrl.modes, float)

cases = [("occlusion", "v41+sup-s0", "geoev+sup", "(a) V4.1 fall under occlusion (D-ev survived)"),
         ("ood_plant", "v41+sup-s0", "geoev+sup", "(b) V4.1 track-end fall, unfamiliar plant"),
         ("combo", "v41+sup-s0", None, "(c) V4.1 fall, combo (low light + 5x noise + OOD)")]
fig, axes = plt.subplots(3, len(cases), figsize=(17, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 0.5, 1]))
meta = []
for j, (cond, cfg, other, title) in enumerate(cases):
    row = first_fail(cond, cfg, other)
    lg, pipe, calls, modes = rerun(cond, cfg, row)
    n = len(lg["s"]); t = np.arange(n)
    th, thh = np.degrees(lg["s"][:, 2]), np.degrees(lg["s_hat"][:, 2]); sg = np.degrees(np.array(pipe.log["sigma_th"]))[:n]
    ax = axes[0, j]
    ax.plot(t, th, color="#0b0b0b", lw=1.4, label="true tilt"); ax.plot(t, thh, color="#6250d6", lw=1.1, label="V4.1 estimate")
    ax.fill_between(t, thh - 2 * sg, thh + 2 * sg, color="#6250d6", alpha=0.15, lw=0, label="±2σ")
    ax.axvline(BV.KICK_TICK, color="#8a8984", ls=":"); ax.axvline(lg["fail_tick"], color="#d03b3b", lw=2, alpha=0.6)
    for c in calls: ax.axvline(c, ymin=0.96, ymax=1.0, color="#eb6834", lw=0.8)
    ax.set_ylim(-45, 45); ax.set_title(f"{title}\nhit {row['kick']} rad/s, seed {row['seed']}: fell at tick {lg['fail_tick']}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1, j]
    ax.imshow(np.array(pipe.log["state_seq"])[None, :n], aspect="auto", cmap=ListedColormap(SC), vmin=-0.5, vmax=5.5, extent=(0, n, 0, 1))
    ax.set_yticks([]); ax.set_ylabel("evidence", fontsize=7)
    ax = axes[2, j]
    x = lg["s"][:, 0]
    ax.plot(t, x, color="#0b0b0b", lw=1.2, label="cart x (m)"); ax.axhline(1.0, color="#d03b3b", ls="--", lw=0.8); ax.axhline(-1.0, color="#d03b3b", ls="--", lw=0.8)
    ax.fill_between(t[:len(modes)], -1.1, -1.1 + 0.2 * modes, color="#eb6834", alpha=0.5, lw=0, label="supervisor intervening")
    ax.set_ylim(-1.15, 1.15); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_xlabel("tick (ms)", fontsize=8)
    meta.append(dict(cond=cond, cfg=cfg, kick=row["kick"], seed=row["seed"], fail_tick=lg["fail_tick"], neural_calls=len(calls),
                     state_share={STATE_NAMES[k]: float(np.mean(np.array(pipe.log["state_seq"]) == k)) for k in range(6)}))
    np.savetxt(f"{OUT}/{cond}_{cfg}_seed{row['seed']}_k{row['kick']}.csv",
               np.column_stack([t, lg["s"][:, 0], lg["s"][:, 2], lg["s_hat"][:, 2], np.radians(sg), np.array(pipe.log["state_seq"])[:n], modes[:n]]),
               delimiter=",", header="tick,x_true,theta_true,theta_est,sigma_theta,evidence_state,supervisor_intervening", comments="", fmt="%.6g")
for a in axes.flat:
    a.tick_params(labelsize=7)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SC]
fig.legend(handles, STATE_NAMES, loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.suptitle("V4.1 held-out failure cases (simulation). Orange ticks on top: neural calls. Dotted: hit. Red: fall.", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("results/v41/fig_v41_cases.png", dpi=130)
json.dump(meta, open(f"{OUT}/cases.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
