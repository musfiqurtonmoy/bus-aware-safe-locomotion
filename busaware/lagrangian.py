"""SAC-Lagrangian baseline (PROMPTS.md Step 6): a gymnasium wrapper plus an SB3 callback.

Cost of an environment step: c = (substeps with undervoltage OR overcurrent) / substeps,
from HoppingLegEnv's info["uvoc"] (the plant is unchanged; mode "baseline" underneath).
Episode cost: mean of c over the episode's steps = fraction of the episode's substeps
with a violation.
Reward seen by SAC: r - lam * c, with lam the current multiplier.
Multiplier (dual ascent, after every episode):
    lam <- max(0, lam + lr * (episode_cost - d)),   d = cost limit (Step 6: 0.001).
Each update is logged to a CSV: episode, timesteps, episode length, episode cost, the
multiplier used during the episode and after the update, and the unshaped episode return.

Known approximation (as specified: wrapper + callback, no new libraries): transitions in the
replay buffer keep the shaped reward computed with the multiplier at the time they were
collected; they are not re-weighted when lam changes.
"""
import csv
import pathlib

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


class LagrangianCostWrapper(gym.Wrapper):
    def __init__(self, env, lam=0.0):
        super().__init__(env)
        self.lam = float(lam)
        self._cost_sum, self._len, self._ret = 0.0, 0, 0.0

    def reset(self, **kw):
        self._cost_sum, self._len, self._ret = 0.0, 0, 0.0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        c = info["uvoc"] / self.env.unwrapped.substeps
        self._cost_sum += c
        self._len += 1
        self._ret += r
        info = dict(info, cost=c, lam=self.lam, reward_unshaped=r)
        if term or trunc:
            info["episode_cost"] = self._cost_sum / self._len
            info["episode_return_unshaped"] = self._ret
            info["episode_len"] = self._len
        return obs, r - self.lam * c, term, trunc, info


class LagrangianCallback(BaseCallback):
    """Updates wrapper.lam after every finished episode and appends a row to csv_path."""
    FIELDS = ["episode", "timesteps", "episode_len", "episode_cost", "lam_used", "lam_new",
              "episode_return_unshaped"]

    def __init__(self, wrapper, cost_limit, lr, csv_path, verbose=0):
        super().__init__(verbose)
        self.wrapper, self.d, self.lr = wrapper, float(cost_limit), float(lr)
        self.path = pathlib.Path(csv_path)
        self.n_ep = 0

    def _on_training_start(self):
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def update(self, episode_cost):
        lam_used = self.wrapper.lam
        self.wrapper.lam = max(0.0, lam_used + self.lr * (episode_cost - self.d))
        return lam_used, self.wrapper.lam

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode_cost" in info:
                self.n_ep += 1
                lam_used, lam_new = self.update(info["episode_cost"])
                with open(self.path, "a", newline="") as f:
                    csv.writer(f).writerow([self.n_ep, self.num_timesteps, info["episode_len"],
                                            info["episode_cost"], lam_used, lam_new,
                                            info["episode_return_unshaped"]])
        return True
