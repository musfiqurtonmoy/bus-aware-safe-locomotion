"""Training curve of one run: episode return against environment steps, from the SB3 Monitor log.

    python scripts/plot_training.py --run baseline_s0

Reads results/<run>/monitor.monitor.csv. Writes to results/<run>/training/:
  fig_return.png/.pdf     episode returns (dots) and a moving mean over --window episodes
  training_summary.json   episode count, steps, episodes that terminated early, mean/std return
                          in the last two blocks of --block steps, and the least-squares slope of
                          return over the last --tail steps with its 95 % confidence interval
Plateau check (fixed here, reported, not tuned): the run counts as plateaued if the 95 % CI of
the slope over the last --tail steps contains zero.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, MUTED, GRIDC, AXISC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
})
T975 = 1.96  # normal approximation; tail windows hold ~100 episodes


def load_monitor(path):
    """SB3 Monitor CSV (one JSON comment line, then a header) -> dict of arrays."""
    lines = pathlib.Path(path).read_text().splitlines()
    head = lines[1].split(",")
    rows = np.array([[float(x) for x in ln.split(",")] for ln in lines[2:] if ln.strip()])
    return {h: rows[:, i] for i, h in enumerate(head)}


def slope_ci(x, y):
    """Least-squares slope of y on x and its 95 % half-width (normal approximation)."""
    xm = x - x.mean()
    b = float(np.sum(xm * (y - y.mean())) / np.sum(xm ** 2))
    resid = y - y.mean() - b * xm
    se = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2) / np.sum(xm ** 2)))
    return b, T975 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, default=20, help="moving-mean window, episodes")
    ap.add_argument("--block", type=int, default=30_000, help="steps per comparison block")
    ap.add_argument("--tail", type=int, default=100_000, help="steps for the slope fit")
    a = ap.parse_args()
    run_dir = ROOT / "results" / a.run
    m = load_monitor(run_dir / "monitor.monitor.csv")
    r, ln = m["r"], m["l"].astype(int)
    steps = np.cumsum(ln)                  # environment steps at the end of each episode
    total = int(steps[-1])
    max_len = int(ln.max())

    def block(lo, hi):
        sel = (steps > lo) & (steps <= hi)
        return dict(steps=[int(lo), int(hi)], n=int(sel.sum()),
                    mean=float(r[sel].mean()), std=float(r[sel].std(ddof=1)))

    last = block(total - a.block, total)
    prev = block(total - 2 * a.block, total - a.block)
    tail = steps > total - a.tail
    b, hw = slope_ci(steps[tail].astype(float), r[tail])
    per = 100_000
    summary = dict(
        run=a.run, episodes=int(len(r)), total_steps=total, episode_length_max=max_len,
        terminated_early=int((ln < max_len).sum()),
        first_episode_return=float(r[0]), max_episode_return=float(r.max()),
        last_block=last, previous_block=prev,
        tail_slope=dict(steps=[total - a.tail, total], n=int(tail.sum()),
                        per_100k_steps=b * per, ci95_half_width_per_100k_steps=hw * per),
        plateau_rule="95 % CI of the slope over the tail contains zero",
        plateaued=bool(abs(b) <= hw),
    )
    out = run_dir / "training"
    out.mkdir(exist_ok=True)
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))

    w = min(a.window, len(r))
    mov = np.convolve(r, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(steps / 1e3, r, "o", ms=2.2, color=MUTED, alpha=0.6, label="episode return")
    ax.plot(steps[w - 1:] / 1e3, mov, color=BLUE, lw=1.6, label=f"mean of last {w} episodes")
    ax.axvspan((total - a.tail) / 1e3, total / 1e3, color=GRIDC, alpha=0.5, lw=0,
               label=f"slope window ({a.tail // 1000}k steps)")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("episode return")
    ax.set_title(f"{a.run}: return per {max_len}-step episode", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_return.{ext}", dpi=150)
    plt.close(fig)

    print(f"{summary['episodes']} episodes, {total} steps, {summary['terminated_early']} terminated early")
    print(f"return: first {r[0]:.1f}, max {r.max():.1f}")
    for nm, bl in (("previous", prev), ("last", last)):
        print(f"{nm} block {bl['steps'][0]}-{bl['steps'][1]}: n={bl['n']}, mean {bl['mean']:.1f} "
              f"+- {bl['std']:.1f} (std)")
    print(f"slope over last {a.tail} steps: {b * per:+.1f} +- {hw * per:.1f} per 100k steps (95 % CI); "
          f"plateaued={summary['plateaued']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
