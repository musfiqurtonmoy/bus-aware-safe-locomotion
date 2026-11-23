"""Train SAC on the hopping leg, from scratch or warm-started from a trained run.

  python scripts/train.py --mode baseline --steps 300000 --seed 0
  python scripts/train.py --mode filter --steps 150000 --seed 0 --init baseline_s0
  python scripts/train.py --mode penalty --penalty_w 0.05 --steps 150000 --seed 0 --init baseline_s0
  python scripts/train.py --mode lagrangian --steps 150000 --seed 0 --init baseline_s0
Output: results/<run>/ with model.zip, monitor.monitor.csv, config.json; run names from
busaware/runs.run_name (baseline_s0, filter_ws_s0, penalty_w0.05_ws_s0, ...). Refuses to
overwrite a run that already has model.zip.
Sized for a 4-core CPU with 8 GB RAM: CPU device, small networks, 200k buffer.

Warm start (--init <run>): SAC.load restores the source run's actor, both critics and their
target networks, the entropy coefficient and the Adam optimiser states, and the
hyperparameters (buffer 200k, batch 256, net [128, 128], learning_starts 5000). The replay
buffer is NOT loaded: train.py never saved one, and its rewards were computed with the source
run's reward (in penalty or filter mode they would be wrong) and, for filter mode, with
unfiltered dynamics. The new buffer starts empty; for the first learning_starts steps it is
filled with the loaded policy's own stochastic actions (SB3's default would use uniformly
random actions) and no gradient steps are taken. These steps count towards --steps.

Lagrangian (--mode lagrangian, PROMPTS Step 6): baseline environment wrapped by
busaware/lagrangian.LagrangianCostWrapper; the multiplier is updated after every episode by
LagrangianCallback with limit --cost_limit (d) and step size --lag_lr, starting at --lag_init,
and logged to results/<run>/lagrangian.csv.
"""
import sys, pathlib, argparse, time, json, datetime
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import torch
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from busaware.leg_env import HoppingLegEnv
from busaware.runs import run_name, env_mode, DEFAULT_PENALTY_W
from busaware.lagrangian import LagrangianCostWrapper, LagrangianCallback


class WarmStartSAC(SAC):
    """SAC whose warm-up steps (num_timesteps < learning_starts) act with the current policy
    instead of uniformly random actions. Saved models load with plain SAC.load."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="baseline", choices=["baseline", "penalty", "filter", "lagrangian"])
ap.add_argument("--steps", type=int, default=300_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--w_energy", type=float, default=0.0)
ap.add_argument("--penalty_w", type=float, default=DEFAULT_PENALTY_W,
                help="penalty mode: weight on the UV + OC substep fraction per step")
ap.add_argument("--init", default=None, help="warm start from results/<init>/model.zip")
ap.add_argument("--cost_limit", type=float, default=0.001,
                help="lagrangian: limit d on the mean episode cost (Step 6: 0.001)")
ap.add_argument("--lag_lr", type=float, default=0.1,
                help="lagrangian: multiplier step size per episode (ASSUMED, CHANGELOG #22)")
ap.add_argument("--lag_init", type=float, default=0.0, help="lagrangian: initial multiplier")
ap.add_argument("--device", default="cpu")      # small MLPs: CPU is usually faster than a GTX 1050 Ti
a = ap.parse_args()

torch.set_num_threads(2)
name = run_name(a.mode, a.seed, a.penalty_w, a.init)
out = ROOT / "results" / name
if (out / "model.zip").exists():
    sys.exit(f"{out / 'model.zip'} exists; not overwriting")
out.mkdir(parents=True, exist_ok=True)
base = HoppingLegEnv(mode=env_mode(a.mode), w_energy=a.w_energy, penalty_w=a.penalty_w, seed=a.seed)
lag = LagrangianCostWrapper(base, lam=a.lag_init) if a.mode == "lagrangian" else None
env = Monitor(lag if lag is not None else base, str(out / "monitor"), info_keywords=("uv", "oc", "ov", "filt"))
callback = (LagrangianCallback(lag, a.cost_limit, a.lag_lr, out / "lagrangian.csv")
            if lag is not None else None)
if a.init:
    src = ROOT / "results" / a.init / "model.zip"
    model = WarmStartSAC.load(src, env=env, device=a.device)
    model.set_random_seed(a.seed)
    assert model.replay_buffer.size() == 0, "warm start must begin with an empty replay buffer"
else:
    model = SAC("MlpPolicy", env, seed=a.seed, device=a.device, verbose=1,
                buffer_size=200_000, batch_size=256, learning_starts=5_000,
                policy_kwargs=dict(net_arch=[128, 128]))
cfg = dict(run=name, mode=a.mode, seed=a.seed, steps=a.steps, penalty_w=a.penalty_w,
           w_energy=a.w_energy, init=a.init,
           lagrangian=dict(cost_limit=a.cost_limit, lag_lr=a.lag_lr, lag_init=a.lag_init,
                           cost="fraction of substeps with undervoltage or overcurrent",
                           log="lagrangian.csv") if lag is not None else None,
           init_loaded="actor, critics, target critics, entropy coefficient, optimisers; "
                       "replay buffer not loaded" if a.init else None,
           buffer_size=model.buffer_size, batch_size=model.batch_size,
           learning_starts=model.learning_starts, sb3=stable_baselines3.__version__,
           started=datetime.datetime.now().isoformat(timespec="seconds"))
(out / "config.json").write_text(json.dumps(cfg, indent=2))
t0 = time.time()
model.learn(a.steps, log_interval=20, reset_num_timesteps=True, callback=callback)
model.save(out / "model")
print(f"done {name}: {a.steps} steps in {(time.time() - t0) / 60:.1f} min -> {out}")
