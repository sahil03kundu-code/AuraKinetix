"""Deliverable 8: failure / recovery visualisations on TEST trials (re-runs stored configurations
deterministically; same seeds and code path as benchmark_v4.run_one).

Panels: true vs estimated tilt, estimator sigma, detector score / alarm, V4 actions.
Cases: (a) sign flip, V3+shield vs V4 on the same trial, (b) first V4 fall under occlusion,
(c) first V4 fall in the combo scenario. Also writes a per-tick CSV log for each case.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(1)
import benchmark_v4 as B4
from aurasync.simulate import run_episode

OUT = "results/v4/test_cases"
os.makedirs(OUT, exist_ok=True)


def rerun(cond_name, cfg, kick, i):
    cond, plant, snap, kicks, n, fault = B4.scenario_spec(cond_name)
    pipe, ctrl_f = B4.build(cfg)
    seed = (800 if cond_name == "snap" else 500) + i
    rng = np.random.default_rng(seed); s0 = [rng.uniform(-0.1, 0.1), 0, rng.uniform(-0.03, 0.03), 0]
    run_pipe = B4.Scenario(pipe, fault, np.random.default_rng(seed + 7)) if fault else pipe
    log = run_episode(plant, ctrl_f(run_pipe), run_pipe, B4.N_TICKS, s0,
                      kicks=[(B4.KICK_TICK, kick * (1 if i % 2 == 0 else -1))], snap=snap, cam_seed=seed, cond=cond)
    sig = np.degrees(np.asarray(pipe.log["sigma_th"], float)) if hasattr(pipe, "log") and "sigma_th" in pipe.log else None
    det = np.array(pipe.det_log, dtype=object) if hasattr(pipe, "det_log") else None
    return log, sig, det, getattr(pipe, "ev_log", [])


def first_fail(cond, cfg):
    rows = json.load(open(f"results/v4/raw/{cond}/{cfg}.json"))["rows"]
    for r in rows:
        if not r["survived"]:
            return r["kick"], r["trial"]
    return rows[0]["kick"], rows[0]["trial"]


def panel(axs, title, log, sig, det, ev, mark=None):
    n = len(log["s"]); t = np.arange(n)
    th, thh = np.degrees(log["s"][:, 2]), np.degrees(log["s_hat"][:n, 2])
    ax = axs[0]
    ax.plot(t, th, color="#0b0b0b", lw=1.4, label="true tilt")
    ax.plot(t, thh, color="#eb6834", lw=1.1, label="estimate")
    if sig is not None:
        m = min(n, len(sig)); s = sig[:m]; ax.fill_between(t[:m], thh[:m] - 2 * s, thh[:m] + 2 * s, color="#eb6834", alpha=0.15, lw=0, label="±2σ")
    ax.axvline(B4.KICK_TICK, color="#8a8984", ls=":", lw=1)
    if mark: ax.axvline(mark, color="#d03b3b", ls="--", lw=1)
    if log["fail_tick"] is not None: ax.axvline(log["fail_tick"], color="#d03b3b", lw=2, alpha=0.6)
    ax.set_ylim(-40, 40); ax.set_title(title, fontsize=9, loc="left"); ax.set_ylabel("deg", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax = axs[1]
    if det is not None and len(det):
        sc = np.array([d[0] for d in det], float)[:n]; al = np.array([d[1] for d in det], bool)[:n]
        ax.plot(t[:len(sc)], sc, color="#2a78d6", lw=1, label="detector score (alarm at 1)")
        ax.fill_between(t[:len(al)], 0, al * 1.0, color="#d03b3b", alpha=0.15, lw=0, label="alarm")
        for e in ev:
            if e["action"] == "reinit_detail":
                ax.axvline(e["k"], color="#1baf7a", lw=1.5)
        ax.plot([], [], color="#1baf7a", lw=1.5, label="re-initialisation")
        ax.set_ylim(0, 3); ax.legend(fontsize=7, loc="upper right", frameon=False)
    else:
        ax.text(0.5, 0.5, "V3: no detector", transform=ax.transAxes, ha="center", fontsize=8, color="#52514e")
    ax.set_xlabel("tick (ms)", fontsize=8)
    for a in axs:
        a.tick_params(labelsize=7); a.grid(color="#e4e3df", lw=0.6)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)


def save_csv(name, log, sig, det):
    n = len(log["s"])
    cols = [np.arange(n), log["s"][:, 2], log["s_hat"][:n, 2], np.asarray(log["u"])[:n] if len(log["u"]) >= n else np.pad(np.asarray(log["u"]), (0, n - len(log["u"])))]
    hdr = "tick,theta_true,theta_est,u"
    if sig is not None:
        cols.append(np.radians(sig[:n])); hdr += ",sigma_theta"
    if det is not None and len(det):
        cols += [np.array([d[0] for d in det], float)[:n], np.array([d[1] for d in det], float)[:n],
                 np.array([d[2] for d in det], float)[:n], np.array([d[3] for d in det], float)[:n]]
        hdr += ",det_score,alarm,fused,reinit"
    m = min(len(c) for c in cols)
    np.savetxt(f"{OUT}/{name}.csv", np.column_stack([c[:m] for c in cols]), delimiter=",", header=hdr, comments="", fmt="%.6g")


if __name__ == "__main__":
    k_sf, i_sf = first_fail("sign_flip", "ev-trig+shield-s0")
    cases = [("sign_flip", "ev-trig+shield-s0", k_sf, i_sf, "(a) sign flip, V3 + shield", B4.KICK_TICK + 60),
             ("sign_flip", "v4+sup-s0", k_sf, i_sf, "(a) same trial, V4", B4.KICK_TICK + 60),
             ("occlusion", "v4+sup-s0", *first_fail("occlusion", "v4+sup-s0"), "(b) V4 fall under occlusion", None),
             ("combo", "v4+sup-s0", *first_fail("combo", "v4+sup-s0"), "(c) V4 fall in combo scenario", None)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    meta = []
    for j, (cond, cfg, kick, i, title, mark) in enumerate(cases):
        log, sig, det, ev = rerun(cond, cfg, kick, i)
        res = "survived" if log["fail_tick"] is None else f"fell at tick {log['fail_tick']}"
        panel(axes[:, j], f"{title}\nhit {kick} rad/s, test trial {i}: {res}", log, sig, det, ev, mark)
        save_csv(f"{cond}_{cfg}_k{kick}_t{i}", log, sig, det)
        meta.append(dict(cond=cond, cfg=cfg, kick=kick, trial=i, fail_tick=log["fail_tick"],
                         actions=[e for e in ev if e["action"] != "reinit_detail"][:20]))
    fig.suptitle("Test-set cases (simulation). Dotted: hit. Dashed red: fault injected. Solid red: fall.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig("results/v4/fig_test_cases.png", dpi=130)
    json.dump(meta, open(f"{OUT}/cases.json", "w"), indent=1, default=str)
    print(json.dumps([{k: m[k] for k in ("cond", "cfg", "kick", "trial", "fail_tick")} for m in meta], indent=1))
