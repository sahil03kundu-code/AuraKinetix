"""V4 Steps 3-5 on the development set (validation seeds only, open-loop replay).

Each recorded closed-loop run is replayed through V4 variants. The trajectory is
fixed (the controller already acted), so only the estimate changes: this isolates
DETECTION quality and whether CORRECTION reduces estimation error, before any
closed-loop test.

Variants
  v3          V4Pipeline(correct=False, inflate=False)   == V3 Baseline B (checked to 1e-17)
  inflate     detection + uncertainty inflation only
  fuse        gated Kalman fusion only (no re-initialisation)
  full        final V4: fusion + confirmed re-initialisation + position-only neural fusion, hidden reset
  full-keeph  same, observer memory kept on re-initialisation
  full-ow     first V4 design (neural correction overwrites incl. velocities)
Ground truth is used only for scoring.
"""
import os, sys, json, glob, time
from multiprocessing import Pool
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import benchmark_v3 as B
from aurasync.v4.pipeline import V4Pipeline

OUT = "results/v4"
VARIANTS = {
    "v3": dict(correct=False, inflate=False),
    "inflate": dict(correct=False, inflate=True),
    "fuse": dict(correct=True, inflate=False, gate=9.0, cooldown=10 ** 9, neural_fusion="kalman_pos"),
    "full": dict(correct=True, inflate=True, hidden_reset="zero", neural_fusion="kalman_pos"),
    "full-keeph": dict(correct=True, inflate=True, hidden_reset="keep", neural_fusion="kalman_pos"),
    "full-ow": dict(correct=True, inflate=True, hidden_reset="zero", neural_fusion="overwrite"),
}


def make(variant):
    cal, tu = B.calibration(), B.tuning()
    kw = dict(temp=np.array(cal["temp"]["varhead"]), latency_samples=cal["latency"], Q=np.diag(cal["Q_diag"]), **tu["trigger"])
    v = dict(VARIANTS[variant])
    if v.get("cooldown") == 10 ** 9:          # fusion-only: re-initialisation disabled
        v["confirm_n"] = 10 ** 6
        v.pop("cooldown")
    return V4Pipeline(B.event_model("event_s0"), mode="triggered", **kw, **v)


def replay_file(path):
    torch.set_num_threads(1)
    d = dict(np.load(path))                    # load every array ONCE (NpzFile re-reads on each key access)
    surf = d["surf"].astype(np.float32)
    res = {}
    n = len(d["s"])
    for v in VARIANTS:
        p = make(v); p.reset(d["s0"].astype(float))
        est, sig = np.zeros((n, 4)), np.zeros((n, 4))
        for k in range(n):
            p.update(k, (k + 1) * 1e-3, None, surf[k], float(d["u"][k]))
            est[k], S = p.out[-1]; sig[k] = np.sqrt(np.clip(np.diag(S), 1e-16, None))
        dl = np.array(p.det_log, dtype=float)                 # score, alarm, fuse, reinit, valid
        res[v] = dict(est=est, sig=sig, det=dl, events=p.ev_log)
    return path, dict(s=d["s"], fail=int(d["fail"]), kick=float(d["kick"])), res


def metrics(s, est, sig, det, fail):
    e = np.degrees(np.abs(est[50:, 2] - s[50:, 2])); th = np.degrees(s[50:, 2]); eh = np.degrees(est[50:, 2])
    sg = np.degrees(sig[50:, 2])
    wrong_sign = (np.sign(eh) != np.sign(th)) & (np.abs(th) > 3) & (np.abs(eh) > 1)
    z = e / np.maximum(sg, 1e-9)
    return dict(mean=float(e.mean()), median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(e.max()),
                frac_gt5=float((e > 5).mean()), frac_gt10=float((e > 10).mean()), wrong_sign=float(wrong_sign.mean()),
                conf_wrong=float(((e > 5) & (z > 3)).mean()), cov1=float((z < 1).mean()), cov2=float((z < 2).mean()),
                reinits=int(det[:, 3].sum()), fuses=int(det[:, 2].sum()), valid_meas=float(det[:, 4].mean()))


def detection(s, est_v3, det, fail):
    """Per-tick detection of unreliable estimates (|error| > 5 deg) by the alarm, using the
    detect-only run (the detector sees the uncorrected V3 estimate)."""
    e = np.degrees(np.abs(est_v3[:, 2] - s[:, 2]))
    bad = e > 5
    alarm = det[:, 1] > 0
    first_alarm = int(np.argmax(alarm)) if alarm.any() else None
    lead = (fail - first_alarm) if (fail >= 0 and first_alarm is not None and first_alarm < fail) else None
    return dict(tp=int((alarm & bad).sum()), fp=int((alarm & ~bad).sum()), fn=int((~alarm & bad).sum()),
                tn=int((~alarm & ~bad).sum()), first_alarm=first_alarm, lead=lead, any_alarm=bool(alarm.any()),
                first_bad=int(np.argmax(bad)) if bad.any() else None)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob("data/v4dev/*.npz"))
    t0 = time.time(); store = {}
    with Pool(2) as pool:
        for i, (path, info, res) in enumerate(pool.imap_unordered(replay_file, files)):
            store[os.path.basename(path)] = (info, res)
            if (i + 1) % 40 == 0:
                print(f"{i+1}/{len(files)} replayed {time.time()-t0:.0f}s", flush=True)
    summary = {}
    for cfg in ("ev-trig-s0", "ev-trig+shield-s0"):
        keys = [k for k in store if k.startswith(cfg + "_")]
        summ = {}
        for v in VARIANTS:
            M = [metrics(store[k][0]["s"], store[k][1][v]["est"], store[k][1][v]["sig"], store[k][1][v]["det"], store[k][0]["fail"]) for k in keys]
            summ[v] = {m: float(np.mean([x[m] for x in M])) for m in M[0]}
            summ[v]["max"] = float(np.max([x["max"] for x in M]))
        D = [detection(store[k][0]["s"], store[k][1]["v3"]["est"], store[k][1]["v3"]["det"], store[k][0]["fail"]) for k in keys]
        tp, fp, fn = sum(x["tp"] for x in D), sum(x["fp"] for x in D), sum(x["fn"] for x in D)
        falls = [x for x in D if store[keys[D.index(x)]][0]["fail"] >= 0]
        leads = [x["lead"] for x in D if x["lead"] is not None]
        summ["detection"] = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                                 falls=len(falls), falls_warned=len(leads), median_lead_ms=float(np.median(leads)) if leads else None,
                                 runs_with_alarm=sum(x["any_alarm"] for x in D), runs=len(D))
        summary[cfg] = summ
        print(f"\n=== {cfg} ({len(keys)} runs)")
        for v in VARIANTS:
            s = summ[v]
            print(f"  {v:11s} mean {s['mean']:5.2f}  median {s['median']:5.2f}  p95 {s['p95']:5.1f}  max {s['max']:5.1f}  >5deg {s['frac_gt5']*100:5.1f}%  "
                  f">10deg {s['frac_gt10']*100:5.1f}%  wrong-sign {s['wrong_sign']*100:4.1f}%  conf-wrong {s['conf_wrong']*100:4.1f}%  "
                  f"cov1/2 {s['cov1']:.2f}/{s['cov2']:.2f}  reinits/run {s['reinits']:.2f}")
        print("  detection:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ["detection"].items()}))
    json.dump(summary, open(f"{OUT}/dev_eval.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/dev_replay.npz", **{f"{k}|{v}|{f}": store[k][1][v][f] for k in store for v in VARIANTS
                                                     for f in ("est", "sig", "det")},
                        **{f"{k}|s": store[k][0]["s"] for k in store})
