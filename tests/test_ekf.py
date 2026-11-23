"""Battery EKF (Step 8, busaware/ekf.py) and its use in the filter env (CHANGELOG #30)."""
from dataclasses import replace

import numpy as np
import pytest

from busaware.battery import Battery
from busaware.ekf import BatteryEKF
from busaware.leg_env import HoppingLegEnv
from busaware.params import Params
from busaware.rollout import EKF_KEYS, episode_metrics, rollout
from busaware.runs import eval_filename

DT = 5e-4


def run_ekf(p_plant, p_model, soc_true, soc0, T=4.0, seed=0):
    """Plant battery under a pulsed current (hop-like: 6 A pushes, -1 A regeneration, 1 A rest,
    plus N(0, 0.5 A)); the EKF gets V = E - R0 I and I with the stated sensor noise.
    Returns per-step arrays t, soc error, E error (estimate at the start of the step), R0 ratio,
    confidence."""
    rng, bat = np.random.default_rng(seed), Battery(p_plant, soc_true)
    ekf = BatteryEKF(p_model, DT, soc0=soc0)
    rows = []
    for k in range(int(round(T / DT))):
        ph = (k * DT) % 0.25
        i = (6.0 if ph < 0.08 else -1.0 if ph < 0.12 else 1.0) + 0.5 * rng.standard_normal()
        e = bat.emf()
        rows.append((k * DT, ekf.soc - bat.soc, ekf.emf() - e, ekf.r0 / p_plant.r0, ekf.confidence))
        ekf.step(e - p_plant.r0 * i + p_model.sigma_v_meas * rng.standard_normal(),
                 i + p_model.sigma_i_meas * rng.standard_normal())
        bat.step(i, DT)
    return np.array(rows).T


@pytest.mark.parametrize("soc_true,soc0", [(0.5, 0.8), (0.2, 0.5), (0.9, 0.6)])
def test_converges_from_a_wrong_initial_soc(soc_true, soc0):
    """From a 0.3 SOC error: after 1 s the SOC error is below 10 % of it and still shrinking, and
    the EMF (what the barrier filter uses) is within 10 mV. SOC itself is only weakly observable
    where the OCV curve is flat (about 2.9 V per unit SOC at 0.9 against 5.9 at 0.5): there
    part of the offset sits in V_rc and decays with the RC time constant (25 s), CHANGELOG #30."""
    t, soc_err, e_err, r0_rel, conf = run_ekf(Params(), Params(), soc_true, soc0)
    e0 = abs(soc0 - soc_true)
    assert abs(soc_err[0]) == pytest.approx(e0)
    late = t >= 1.0
    assert np.max(np.abs(soc_err[late])) < 0.1 * e0
    assert abs(soc_err[-1]) < abs(soc_err[late][0])
    assert np.max(np.abs(e_err[late])) < 0.01                  # V
    assert np.max(np.abs(r0_rel[late] - 1.0)) < 0.03
    assert np.min(conf[late]) > 0.8
    assert np.min(conf[t <= 0.05]) < 0.5                       # inconsistent at the start


@pytest.mark.parametrize("factor", [2.0, 0.5])
def test_converges_from_a_2x_wrong_r0(factor):
    """Plant R0 = factor x the model's R0, which is the EKF's initial R0 (2x wrong either way)."""
    p = Params()
    t, soc_err, e_err, r0_rel, conf = run_ekf(replace(p, r0=factor * p.r0), p, 0.5, 0.5)
    assert r0_rel[0] == pytest.approx(1.0 / factor)
    late = t >= 1.0
    assert np.max(np.abs(r0_rel[late] - 1.0)) < 0.03
    assert np.max(np.abs(e_err[late])) < 0.01
    assert np.max(np.abs(soc_err[late])) < 0.01


def test_confidence_is_zero_before_any_update_and_within_unit_interval():
    ekf = BatteryEKF(Params(), DT, soc0=0.5)
    assert ekf.confidence == 0.0
    rng = np.random.default_rng(3)
    for _ in range(500):
        ekf.step(ekf.ocv(0.5) + 5.0 * rng.standard_normal(), 2.0)    # far noisier than assumed
        assert 0.0 <= ekf.confidence <= 1.0
    assert ekf.confidence == 0.0


def test_update_skipped_at_the_charge_current_limit():
    p = Params()
    ekf = BatteryEKF(p, DT, soc0=0.5)
    x0 = ekf.x.copy()
    assert ekf.step(30.0, -p.i_chg_max) is None                 # wildly wrong V: ignored
    assert ekf.x[0] == pytest.approx(x0[0] + DT * p.i_chg_max / (3600 * p.q_ah))
    assert ekf.x[2] == x0[2]


def test_init_from_rest_inverts_the_ocv_table():
    p = Params()
    ekf = BatteryEKF(p, DT)
    for soc in (0.1, 0.37, 0.5, 0.93):
        ekf.init_from_rest(Battery(p, soc).ocv())
        assert ekf.soc == pytest.approx(soc, abs=1e-9)
        assert ekf.v_rc == 0.0


def roll(env, n=60, seed=1000, soc=0.5):
    rng = np.random.default_rng(11)
    obs, _ = env.reset(seed=seed, options=dict(soc=soc))
    out = []
    for _ in range(n):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
        out.append((obs.copy(), r, info))
        if term:
            break
    return out


def test_true_estimator_env_has_no_ekf_and_unchanged_info():
    env = HoppingLegEnv(mode="filter", seed=0)
    out = roll(env, 5)
    assert env.ekf is None
    assert not any(k in out[-1][2] for k in EKF_KEYS)


@pytest.mark.parametrize("margin", ["fixed", "confidence"])
def test_ekf_env_is_deterministic_and_leaves_np_random_alone(margin):
    a = HoppingLegEnv(mode="filter", seed=0, estimator="ekf", margin_mode=margin, est_noise=0.05)
    b = HoppingLegEnv(mode="filter", seed=0, estimator="ekf", margin_mode=margin, est_noise=0.05)
    t = HoppingLegEnv(mode="filter", seed=0, est_noise=0.05)
    ra, rb, _ = roll(a), roll(b), roll(t)
    assert len(ra) == len(rb)
    for (oa, xa, ia), (ob, xb, ib) in zip(ra, rb):
        assert np.array_equal(oa, ob) and xa == xb
        assert all(ia[k] == ib[k] for k in EKF_KEYS)
    # the sensor noise has its own stream: env.np_random (est_noise factors) is consumed alike
    assert a.np_random.bit_generator.state == t.np_random.bit_generator.state
    lo_v, lo_i = a.margin_v, a.margin_i
    hi_v, hi_i = (a.margin_v_max, a.margin_i_max) if margin == "confidence" else (lo_v, lo_i)
    for _, _, info in ra:
        assert lo_v - 1e-12 <= info["margin_v"] <= hi_v + 1e-12
        assert lo_i - 1e-12 <= info["margin_i"] <= hi_i + 1e-12
        assert 0.0 <= info["ekf_conf"] <= 1.0


def test_ekf_env_tracks_a_2x_r0_plant():
    env = HoppingLegEnv(mode="filter", seed=0, estimator="ekf", r0_scale=2.0)
    out = roll(env, 300, soc=0.5)
    info = out[-1][2]
    assert abs(info["ekf_r0_rel"] - 1.0) < 0.05
    assert np.mean([o[2]["ekf_e_abs"] for o in out[-100:]]) < 0.02


def test_confidence_margin_needs_the_ekf_and_filenames():
    with pytest.raises(AssertionError):
        HoppingLegEnv(mode="filter", margin_mode="confidence")
    assert eval_filename() == "eval.csv"
    assert eval_filename(estimator="ekf") == "eval_est-ekf.csv"
    assert eval_filename(estimator="ekf", margin="confidence", r0_scale=2.0) == "eval_est-ekfconf_r2.0_n0.0.csv"
    assert eval_filename(estimator="ekf", est_noise=0.05) == "eval_est-ekf_r1.0_n0.05.csv"
    with pytest.raises(ValueError):
        eval_filename(margin="confidence")


def test_episode_metrics_report_the_ekf_columns():
    env = HoppingLegEnv(mode="filter", seed=0, estimator="ekf", margin_mode="confidence", episode_s=0.2)
    r = rollout(env, lambda obs: np.zeros(2), 0.5, 1000)
    m = episode_metrics(r, env.dt_sim, env.substeps, env.max_steps)
    for k in ("ekf_e_abs_mean", "ekf_e_abs_max", "ekf_r0_rel_final", "ekf_soc_err_final",
              "ekf_conf_mean", "ekf_conf_min", "margin_v_mean", "margin_i_mean"):
        assert np.isfinite(m[k]), k
    env_t = HoppingLegEnv(mode="filter", seed=0, episode_s=0.2)
    m_t = episode_metrics(rollout(env_t, lambda obs: np.zeros(2), 0.5, 1000), env_t.dt_sim,
                          env_t.substeps, env_t.max_steps)
    assert "ekf_e_abs_mean" not in m_t
