"""Proof-assumption counters A4 (plant root below x*) and A6 (empty voltage-circle / current-box
intersection), CHANGELOG #25. They are counted only: control and plant behaviour is unchanged."""
import importlib.util
import pathlib

import numpy as np
import pytest

from busaware.bus import Bus
from busaware.drive import Drive
from busaware.leg_env import HoppingLegEnv
from busaware.params import Params
from busaware.rollout import PROOF_COUNTERS, episode_metrics, rollout

DATA = pathlib.Path(__file__).resolve().parent / "data"
_spec = importlib.util.spec_from_file_location("reftraj", DATA / "make_reference_trajectories.py")
reftraj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reftraj)
REF = np.load(DATA / "reference_trajectories.npz")


def _assert_same_as_reference(name, got):
    for k, v in got.items():
        ref = REF[f"{name}/{k}"]
        assert ref.shape == v.shape, (name, k)
        np.testing.assert_array_equal(v, ref, err_msg=f"{name}/{k}")


@pytest.mark.parametrize("name,kw,soc,seed", reftraj.CASES, ids=[c[0] for c in reftraj.CASES])
def test_counters_leave_control_and_plant_unchanged(name, kw, soc, seed):
    """Every observation, reward, termination flag, earlier info counter and plant state is
    bit-identical to the trajectories recorded at commit 7a895f5, before the counters existed
    (tests/data/make_reference_trajectories.py). Cases: every mode, SOC 1.0 to 0.2, bus collapse,
    estimator noise, scaled plant R0."""
    _assert_same_as_reference(name, reftraj.run_case(kw, soc, seed))


def test_counter_values_do_not_feed_back(monkeypatch):
    """Forcing both counters' inputs to their opposite extremes (every joint empty, x* = +inf)
    leaves the trajectory unchanged, and the counters then count every substep."""
    monkeypatch.setattr(Drive, "circle_box_empty", lambda self, w, v: np.ones(np.shape(w), bool))
    monkeypatch.setattr(Bus, "turning_point", lambda self, dt: np.inf)
    name, kw, soc, seed = reftraj.CASES[4]                 # filter mode, SOC 0.2
    _assert_same_as_reference(name, reftraj.run_case(kw, soc, seed))
    env = HoppingLegEnv(seed=0, **kw)
    env.reset(seed=0, options=dict(soc=soc))
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a4_fail"] == env.substeps and info["a6_empty"] == env.substeps


def test_circle_box_empty_matches_the_realisable_set():
    """Where circle_box_empty is False, both ends of bounds() lie on or inside the voltage circle
    and within the current limit; where it is True, bounds() is a single point outside the
    circle. Random joint speeds and bus voltages, including the low-voltage collapse range."""
    p, d, rng = Params(), Drive(Params()), np.random.default_rng(3)
    n_empty = n_full = 0
    for _ in range(5000):
        w, v = rng.uniform(-60.0, 60.0, 2), rng.uniform(0.0, 27.5)
        empty = d.circle_box_empty(w, v)
        lo, hi = d.bounds(w, v)
        we = p.pole_pairs * p.gear * w
        circ = lambda iq: (p.r_phase * iq + we * p.psi) ** 2 + (we * p.l_phase * iq) ** 2 - (v / np.sqrt(3.0)) ** 2
        tol = 1e-9 * max(1.0, (v / np.sqrt(3.0)) ** 2)
        for i in range(2):
            if empty[i]:
                n_empty += 1
                assert lo[i] == hi[i] and circ(lo[i])[i] > -tol
            else:
                n_full += 1
                assert -p.i_lim <= lo[i] <= hi[i] <= p.i_lim
                assert circ(lo[i])[i] <= tol and circ(hi[i])[i] <= tol
    assert n_empty > 100 and n_full > 100


def test_a6_counter_counts_empty_intersection():
    env = HoppingLegEnv(mode="baseline", seed=0)
    env.reset(seed=0, options=dict(soc=0.5))
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a6_empty"] == 0
    env.data.qvel[1:] = [60.0, -60.0]                      # back-EMF far above the bus voltage
    assert env.drive.circle_box_empty(env.data.qvel[1:], env.bus.v).all()
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a6_empty"] >= 1


def test_a4_counter_counts_a_root_below_the_turning_point():
    env = HoppingLegEnv(mode="baseline", seed=0)
    env.reset(seed=0, options=dict(soc=0.5))
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a4_fail"] == 0
    bus_step = env.bus.step

    def wrong_branch(p_load, dt):                          # emulate a solver on the lower branch
        x_star = env.bus.turning_point(dt)
        s = bus_step(p_load, dt)
        env.bus.v = s["v"] = 0.5 * x_star
        return s

    env.bus.step = wrong_branch
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a4_fail"] == env.substeps


def test_turning_point_is_the_solvers_lower_bracket():
    """In collapse the plant returns V' = x_lo = max(beta/(2 alpha), 1e-9), and turning_point()
    gives the same number bit for bit, so collapse substeps are not counted as A4 failures."""
    env = HoppingLegEnv(mode="baseline", seed=0)
    env.reset(seed=0, options=dict(soc=0.2))
    x_star = env.bus.turning_point(env.dt_sim)
    s = env.bus.step(10.0 * env.bus.max_power(env.dt_sim), env.dt_sim)
    assert s["collapse"] and s["v"] == x_star


def test_episode_metrics_report_all_proof_counters():
    env = HoppingLegEnv(mode="baseline", seed=0, episode_s=0.2)
    r = rollout(env, lambda obs: np.ones(2), soc=0.2, seed=0)
    m = episode_metrics(r, env.dt_sim, env.substeps, env.max_steps)
    assert PROOF_COUNTERS == ("a3_fail", "rem1_fail", "a4_fail", "a6_empty")
    for k in PROOF_COUNTERS:
        assert m[f"{k}_n"] == int(r[k].sum())


def test_a6_timing_diagnostics_are_subsets_of_a6_and_collapse():
    """a6_collapse (A6 in a collapse substep), a6_prev_collapse (A6 right after one) and a6_uv (A6
    at a start voltage below v_min) count subsets of the A6 substeps (CHANGELOG #31)."""
    from busaware.leg_env import HoppingLegEnv
    env = HoppingLegEnv(mode="baseline", seed=0)
    rng = np.random.default_rng(4)
    env.reset(seed=1000, options=dict(soc=0.2))
    tot = dict(a6_empty=0, a6_collapse=0, a6_prev_collapse=0, a6_uv=0, collapse=0)
    for _ in range(300):
        _, _, term, _, info = env.step(rng.choice([-1.0, 1.0], 2))
        for k in tot:
            tot[k] += info[k]
        assert info["a6_collapse"] <= min(info["a6_empty"], info["collapse"])
        assert info["a6_prev_collapse"] <= info["a6_empty"] and info["a6_uv"] <= info["a6_empty"]
        if term:
            break
    assert tot["collapse"] > 0
