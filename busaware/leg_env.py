"""Single hopping leg on a vertical rail with two drives on one battery-fed bus.

mode:
  "baseline" : PD torque -> per-joint current/voltage limits only.
  "penalty"  : baseline + reward penalty on bus violations and electrical power.
  "filter"   : barrier filter projects the current commands before the drives.
                 estimator="true": the filter's EMF is the plant's, its R0 the model's (Params.r0);
                 estimator="ekf": both from the battery EKF (busaware/ekf.py) on noisy bus-voltage
                 and battery-current measurements; margin_mode="confidence" then scales the
                 margins with the EKF's confidence (Step 8, CHANGELOG #30).
  "perjoint" : evaluation-only comparison (CHANGELOG #38): the filter's power budget (true state,
                 same gamma and margins) split equally between the drives, each joint limited on
                 its own (safety_filter.project_per_joint). Same correction penalty as "filter".
The plant (bus, battery, drives) is identical in all modes; only the controller
side differs. Violations are always logged in `info`.
"""
from pathlib import Path
import numpy as np
import gymnasium as gym
import mujoco

from .params import Params
from .battery import Battery
from .bus import Bus
from .drive import Drive, current_lag, output_torque
from .safety_filter import power_budget, project_currents, project_per_joint
from .ekf import BatteryEKF

ASSET = Path(__file__).resolve().parent.parent / "assets" / "leg.xml"
Q_STAND = np.array([0.7, -1.4])
SENSOR_SEED = 2008                   # EKF sensor-noise stream, separate from env.np_random
# Task constants, unchanged since the first training run; named (CHANGELOG #34) so that
# scripts/export_paper.py can report them. Reward per control step, every mode:
#   REWARD_HEIGHT_W * max(z - REWARD_HEIGHT_MIN, 0) + REWARD_ALIVE - REWARD_SMOOTH_W |a - a_prev|^2
REWARD_HEIGHT_W = 8.0                # 1/m, on torso height above REWARD_HEIGHT_MIN
REWARD_HEIGHT_MIN = 0.08             # m, torso height above the stand below which no height reward
REWARD_ALIVE = 0.05                  # per control step
REWARD_SMOOTH_W = 0.01               # on the squared change of the action
FALL_HEIGHT = -0.16                  # m, torso height (rail, relative to the stand) that ends the episode
RESET_JOINT_NOISE = 0.05             # rad, uniform perturbation of hip and knee at reset
OBS_QVEL_SCALE = 0.1                 # joint velocities in the observation
OBS_V_SCALE = 6.0                    # V, bus voltage in the observation: (V - v_obs_ref) / OBS_V_SCALE


class HoppingLegEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode="baseline", soc_range=(0.15, 1.0), params=None,
                 episode_s=4.0, substeps=8, kp=40.0, kd=1.0, act_scale=0.9,
                 gamma=0.3, margin_v=0.2, margin_i=0.5, est_noise=0.0,
                 r0_scale=1.0, w_filter=0.02, w_energy=0.0, penalty_w=0.5, seed=None,
                 estimator="true", margin_mode="fixed", margin_v_max=0.8, margin_i_max=2.0):
        assert mode in ("baseline", "penalty", "filter", "perjoint")
        assert estimator in ("true", "ekf") and margin_mode in ("fixed", "confidence")
        assert mode != "perjoint" or estimator == "true", "the per-joint limiter uses the true state only"
        assert margin_mode == "fixed" or estimator == "ekf", "confidence margin needs the EKF"
        # estimator/margin act only in filter mode. Confidence margin: margin_v in
        # [margin_v, margin_v_max] and margin_i in [margin_i, margin_i_max], larger when the EKF's
        # confidence is lower (margin = min + (1 - confidence) (max - min)).
        self.estimator, self.margin_mode = estimator, margin_mode
        self.margin_v_max, self.margin_i_max = margin_v_max, margin_i_max
        self.ekf, self._n_resets = None, 0
        self.mode, self.soc_range = mode, soc_range
        self.p = params or Params()
        self.p_plant = Params(**{**self.p.__dict__, "r0": self.p.r0 * r0_scale})
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.dt_sim = self.model.opt.timestep
        self.substeps = substeps
        self.max_steps = int(episode_s / (self.dt_sim * substeps))
        self.kp, self.kd, self.act_scale = kp, kd, act_scale
        self.gamma, self.margin_v, self.margin_i = gamma, margin_v, margin_i
        self.est_noise, self.w_filter, self.w_energy = est_noise, w_filter, w_energy
        self.penalty_w = penalty_w                 # penalty mode: weight on UV + OC substep fraction
        self.drive = Drive(self.p)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _obs(self):
        d = self.data
        return np.array([
            d.qpos[0], d.qvel[0], *(d.qpos[1:] - Q_STAND), *(OBS_QVEL_SCALE * d.qvel[1:]),
            *self.prev_action, float(d.ncon > 0),
            (self.bus.v - self.p.v_obs_ref) / OBS_V_SCALE, self.bat.soc], dtype=np.float32)   # trained scaling

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[1:] += self.np_random.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, 2)
        mujoco.mj_forward(self.model, self.data)
        soc = (options or {}).get("soc", self.np_random.uniform(*self.soc_range))
        self.bat = Battery(self.p_plant, soc)
        self.bus = Bus(self.p_plant, self.bat)
        self.ekf = None
        if self.mode == "filter" and self.estimator == "ekf":
            # sensors: own random stream (seeded by the reset seed), so env.np_random draws, incl.
            # the est_noise factors, are the same as with the true-state filter
            self._n_resets += 1
            self.sensor_rng = np.random.default_rng(
                [SENSOR_SEED, seed if seed is not None else 10 ** 9 + self._n_resets])
            self.ekf = BatteryEKF(self.p, self.dt_sim)     # filter's model; R0 prior = Params.r0
            self.ekf.init_from_rest(self.bus.v + self.p_plant.sigma_v_meas * self.sensor_rng.standard_normal())
        self.prev_action = np.zeros(2)
        self.iq_act = np.zeros(2)            # actual drive currents (plant)
        self._prev_collapse = False          # A6 diagnostic only
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1, 1)
        q_des = Q_STAND + self.act_scale * action
        log = dict(uv=0, ov=0, oc=0, uvoc=0, collapse=0, filt=0, infeas=0, a3_fail=0, rem1_fail=0,
                   a4_fail=0, a6_empty=0, a6_collapse=0, a6_prev_collapse=0, a6_uv=0, e_elec=0.0, dcorr=0.0,
                   v_min=np.inf, v_max=-np.inf, i_max=-np.inf, p_peak=0.0)
        if self.ekf is not None:      # EKF diagnostics (logged only; per-step means or end values)
            log.update(ekf_e_abs=0.0, ekf_r0_rel=np.nan, ekf_soc_err=np.nan, ekf_conf=np.nan,
                       margin_v=0.0, margin_i=0.0)
        for _ in range(self.substeps):
            # proof assumptions checked on the plant's state at the start of the substep
            # (paper Sec. filter): A3  E > 0, V >= 0, max(V, E) <= 2 V_c,off;
            # Remark 1 bound  V <= 2E + 2 R0 I_chg + I_chg dt / C. Counted in every mode.
            pp, v_k, e_k = self.p_plant, self.bus.v, self.bat.emf()
            log["a3_fail"] += int(not (e_k > 0.0 and v_k >= 0.0 and max(v_k, e_k) <= 2.0 * pp.v_chop_off))
            log["rem1_fail"] += int(v_k > 2.0 * e_k + 2.0 * pp.r0 * pp.i_chg_max
                                    + pp.i_chg_max * self.dt_sim / pp.c_bus)
            q, w = self.data.qpos[1:], self.data.qvel[1:]
            iq_des = self.drive.iq_from_torque(self.kp * (q_des - q) - self.kd * w)
            iq_des = np.clip(iq_des, -self.p.i_lim, self.p.i_lim)
            lo, hi = self.drive.bounds(w, self.bus.v)   # current limit + voltage circle
            # A6 counter: substeps where some joint's voltage circle and current box do not
            # intersect (bounds() then gives an unrealisable best-effort point). Counted only.
            a6 = bool(np.any(self.drive.circle_box_empty(w, self.bus.v)))
            log["a6_empty"] += int(a6)
            # A6 diagnostics (counted only): the start-of-substep voltage is below v_min, or was
            # left by a collapse substep (the previous one)
            log["a6_uv"] += int(a6 and self.bus.v < self.p_plant.v_min)
            log["a6_prev_collapse"] += int(a6 and self._prev_collapse)
            iq_lim = np.clip(iq_des, lo, hi)            # what the drives alone would do
            iq_cmd = iq_lim
            if self.mode in ("filter", "perjoint"):
                # estimates of battery EMF and R0. True state: the plant's EMF and the model's R0,
                # times (1 + est_noise N(0,1)). EKF: the EKF's estimates; est_noise then multiplies
                # the EKF's two measurements instead (same draws n, below the bus step).
                n = 1.0 + self.est_noise * self.np_random.standard_normal(2)
                margin_v, margin_i = self.margin_v, self.margin_i
                if self.ekf is None:
                    e_hat, r0_hat = self.bat.emf() * n[0], self.p.r0 * n[1]
                else:
                    e_hat, r0_hat = self.ekf.emf(), self.ekf.r0
                    if self.margin_mode == "confidence":
                        u = 1.0 - self.ekf.confidence
                        margin_v += u * (self.margin_v_max - self.margin_v)
                        margin_i += u * (self.margin_i_max - self.margin_i)
                    log["ekf_e_abs"] += abs(e_hat - e_k) / self.substeps
                    log["margin_v"] += margin_v / self.substeps
                    log["margin_i"] += margin_i / self.substeps
                chopper = (self.p.v_chop_off, self.p.v_chop, self.p.r_chop)
                budget = power_budget(
                    self.bus.v, e_hat, r0_hat, self.p.c_bus, self.dt_sim,
                    self.p.v_min, self.p.i_bat_max, self.bus.i_bat,
                    self.gamma, margin_v, margin_i, chopper) - 2 * self.p.p_idle - self.p.p_aux
                a, b = self.drive.power_coeffs(w)
                project = project_currents if self.mode == "filter" else project_per_joint
                iq_cmd, status = project(iq_des, a, b, budget, lo, hi)
                log["filt"] += int(status != "inactive")
                log["infeas"] += int(status == "infeasible")
                # filter correction only: drive-limit clipping is not charged to the filter
                log["dcorr"] += float(np.sum((iq_cmd - iq_lim) ** 2))
            if not np.allclose(self.drive.limit(iq_cmd, w, self.bus.v), iq_cmd, rtol=0.0, atol=1e-9):
                raise AssertionError("commanded iq outside drive bounds")
            # plant: optional first-order current-loop lag (Params.i_loop_bw, off by default).
            # The filter above does not know about it; the actual current stays within the
            # drives' present limits.
            iq = np.clip(current_lag(self.iq_act, iq_cmd, self.p_plant.i_loop_bw, self.dt_sim), lo, hi)
            # plant: if the bus cannot deliver the demand, the drives get less current
            # plant: computer/sensor share p_aux is a constant load beside the drives
            p_aux = self.p_plant.p_aux
            p_max = self.bus.max_power(self.dt_sim)
            iq, scaled = self.drive.scale_to_power(iq, w, p_max - p_aux)
            p_drives = float(np.sum(self.drive.power(iq, w)))
            p_load = min(p_drives + p_aux, p_max)                         # min: rounding only
            self.iq_act = iq
            x_star = self.bus.turning_point(self.dt_sim)   # A4 counter only (state before the step)
            s = self.bus.step(p_load, self.dt_sim)
            # A4 counter: plant's end-of-step voltage below the turning point x* (wrong branch).
            log["a4_fail"] += int(s["v"] < x_star)
            if self.ekf is not None:
                # EKF: measured bus voltage and battery current of this substep (sensor noise,
                # then the est_noise factors), update + predict for the next substep
                w_s = self.sensor_rng.standard_normal(2)
                self.ekf.step((s["v"] + self.p_plant.sigma_v_meas * w_s[0]) * n[0],
                              (s["i_bat"] + self.p_plant.sigma_i_meas * w_s[1]) * n[1])
            # plant: optional gear loss on motoring torque (Params.gear_eff, 1 = off)
            self.data.ctrl[:] = output_torque(self.drive.torque(iq), w, self.p_plant.gear_eff)
            mujoco.mj_step(self.model, self.data)
            for k in ("uv", "ov", "oc"):
                log[k] += int(s[k])
            log["uvoc"] += int(s["uv"] or s["oc"])                        # substeps with UV or OC (Lagrangian cost)
            log["collapse"] += int(scaled or s["collapse"])
            log["a6_collapse"] += int(a6 and (scaled or s["collapse"]))   # A6 in a collapse substep (count only)
            self._prev_collapse = bool(scaled or s["collapse"])
            log["e_elec"] += p_drives * self.dt_sim                      # drives only (reward term unchanged)
            log["v_min"] = min(log["v_min"], s["v"]); log["v_max"] = max(log["v_max"], s["v"])
            log["i_max"] = max(log["i_max"], s["i_bat"]); log["p_peak"] = max(log["p_peak"], p_drives)

        if self.ekf is not None:
            log.update(ekf_r0_rel=self.ekf.r0 / self.p_plant.r0, ekf_soc_err=self.ekf.soc - self.bat.soc,
                       ekf_conf=self.ekf.confidence)
        z = self.data.qpos[0]                      # torso height above stand
        n = self.substeps
        reward = (REWARD_HEIGHT_W * max(z - REWARD_HEIGHT_MIN, 0.0) + REWARD_ALIVE
                  - REWARD_SMOOTH_W * float(np.sum((action - self.prev_action) ** 2)))
        if self.mode == "penalty":
            reward -= self.penalty_w * (log["uv"] + log["oc"]) / n
        if self.mode in ("filter", "perjoint"):
            reward -= self.w_filter * log["dcorr"] / n / self.p.i_max
        reward -= self.w_energy * log["e_elec"]
        self.prev_action = action
        self.t += 1
        terminated = bool(z < FALL_HEIGHT)         # fell
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, log


def stand_static_torques(model=None):
    """Hip and knee torques (N m) that hold the stand keyframe at rest, foot on the floor.

    Statics: qfrc_bias = tau + J^T f, with qvel = 0 so qfrc_bias is gravity only. The rail is
    unactuated, so its row gives the vertical ground force f_z; the ground force is taken as
    purely vertical (the rail carries any horizontal load). J is the foot sphere's centre;
    for rotation about y its vertical velocity equals that of the contact point.
    Returns (tau[hip, knee], f_z).
    """
    model = model or mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacGeom(model, data, jacp, None, gid)
    bias = data.qfrc_bias.copy()
    f_z = bias[0] / jacp[2, 0]
    return bias[1:] - jacp[2, 1:] * f_z, float(f_z)
