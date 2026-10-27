"""Side-by-side GIF: frame-camera stack vs AuraSync event stack recovering from the same hit."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from aurasync.physics import PlantParams
from aurasync.controller import LQRController
from aurasync.simulate import run_episode, FramePipeline, EventPipeline
from benchmark import load_model

KICK = float(os.environ.get("KICK", 3.0))
p = PlantParams(); ctrl = LQRController(p); model = load_model()
runs = [("Frame camera 30 fps (+25 ms)", FramePipeline(30, 0.025), "#4dabf7"),
        ("AuraSync event engine (1 ms)", EventPipeline(model), "#d6336c")]
logs = [run_episode(p, ctrl, pipe, 900, [0, 0, 0.0, 0], kicks=[(150, KICK)], cam_seed=3, stop_on_fail=False)
        for _, pipe, _ in runs]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
artists = []
for ax, (title, _, col) in zip(axes, runs):
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 0.65); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.axhline(-0.04, color="#555", lw=1); ax.set_xticks([]); ax.set_yticks([])
    cart = plt.Rectangle((0, -0.04), 0.2, 0.08, color="#333")
    ax.add_patch(cart)
    pole, = ax.plot([], [], lw=4, color=col, solid_capstyle="round")
    txt = ax.text(-1.05, 0.58, "", fontsize=9, family="monospace")
    artists.append((cart, pole, txt))


def draw(f):
    k = f * 5
    for (cart, pole, txt), log in zip(artists, logs):
        kk = min(k, len(log["s"]) - 1)
        x, _, th, _ = log["s"][kk]
        th = np.clip(th, -1.6, 1.6)
        cart.set_x(x - 0.1)
        pole.set_data([x, x + 0.5 * np.sin(th)], [0.04, 0.04 + 0.5 * np.cos(th)])
        state = "FALLEN" if abs(log["s"][kk][2]) > 0.8 else ("HIT!" if 150 <= kk < 200 else "")
        txt.set_text(f"t={kk:4d} ms  tilt={np.degrees(log['s'][kk][2]):+6.1f}°  {state}")
    return []


anim = FuncAnimation(fig, draw, frames=180, interval=40)
os.makedirs("results", exist_ok=True)
anim.save("results/demo.gif", writer=PillowWriter(fps=25))
print("saved results/demo.gif", [l["fail_tick"] for l in logs])
