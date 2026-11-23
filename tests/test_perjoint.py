"""Joint-level comparison limiter (test mode "perjoint", CHANGELOG #38) and the margin-ablation
file names. Evaluation-only controller; the plant is the same as in every other mode."""
import importlib.util
import pathlib

import numpy as np
import pytest

from busaware.leg_env import HoppingLegEnv
from busaware.runs import eval_filename, env_mode
from busaware.safety_filter import project_currents, project_per_joint

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _random_case(rng):
    a = rng.uniform(-60.0, 60.0, 2)
    b = rng.uniform(0.05, 0.3)
    lo = rng.uniform(-40.0, 0.0, 2)
    hi = lo + rng.uniform(0.0, 60.0, 2)
    d = rng.uniform(-50.0, 50.0, 2)
    budget = rng.uniform(-50.0, 400.0)
    return d, a, b, budget, lo, hi


def test_each_joint_stays_within_its_share_and_box():
    rng = np.random.default_rng(0)
    n_infeasible = 0
    for _ in range(5000):
        d, a, b, budget, lo, hi = _random_case(rng)
        iq, status = project_per_joint(d, a, b, budget, lo, hi)
        assert np.all(iq >= lo - 1e-12) and np.all(iq <= hi + 1e-12)
        p = a * iq + b * iq ** 2
        if status == "infeasible":
            n_infeasible += 1
            continue
        tol = 1e-9 * max(1.0, abs(budget))
        assert np.all(p <= budget / 2 + tol)
        assert p.sum() <= budget + 2 * tol
    assert 0 < n_infeasible < 5000                      # both branches were exercised


def test_each_joint_gets_the_nearest_admissible_current_for_itself():
    rng = np.random.default_rng(1)
    for _ in range(500):
        d, a, b, budget, lo, hi = _random_case(rng)
        iq, status = project_per_joint(d, a, b, budget, lo, hi)
        if status == "infeasible":
            continue
        for i in range(2):
            grid = np.linspace(lo[i], hi[i], 20001)
            ok = grid[a[i] * grid + b * grid ** 2 <= budget / 2]
            if ok.size:
                best = ok[np.argmin(np.abs(ok - d[i]))]
                assert abs(iq[i] - d[i]) <= abs(best - d[i]) + (hi[i] - lo[i]) / 20000 + 1e-9


def test_inactive_when_the_command_already_fits_and_never_closer_than_the_joint_projection():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        d, a, b, budget, lo, hi = _random_case(rng)
        iq, status = project_per_joint(d, a, b, budget, lo, hi)
        box = np.clip(d, lo, hi)
        if status == "inactive":
            assert np.array_equal(iq, box)
        joint, jstatus = project_currents(d, a, b, budget, lo, hi)
        if status != "infeasible" and jstatus != "infeasible":
            # the joint projection is the nearest point satisfying the total budget; the per-joint
            # result satisfies it too, so it cannot be nearer
            assert np.sum((joint - d) ** 2) <= np.sum((iq - d) ** 2) + 1e-6


def test_empty_share_interval_reports_infeasible_with_the_box_minimiser():
    a, b = np.array([10.0, 0.0]), 0.1
    lo, hi = np.array([5.0, -1.0]), np.array([10.0, 1.0])    # joint 0 draws > 0 W anywhere in its box
    iq, status = project_per_joint(np.array([8.0, 0.5]), a, b, 0.0, lo, hi)
    assert status == "infeasible"
    assert iq[0] == pytest.approx(5.0)                        # box minimiser of 10 x + 0.1 x^2 on [5, 10]


def test_perjoint_env_runs_with_the_same_plant():
    pj, f = HoppingLegEnv(mode="perjoint", seed=0), HoppingLegEnv(mode="filter", seed=0)
    assert pj.p_plant.__dict__.keys() == f.p_plant.__dict__.keys()
    assert all(np.array_equal(np.asarray(pj.p_plant.__dict__[k]), np.asarray(f.p_plant.__dict__[k]))
               for k in pj.p_plant.__dict__)
    obs, _ = pj.reset(seed=1000, options=dict(soc=0.2))
    rng = np.random.default_rng(3)
    active = 0
    for _ in range(150):
        obs, r, term, trunc, info = pj.step(rng.uniform(-1, 1, 2))
        active += info["filt"]
        assert np.isfinite(r)
        if term or trunc:
            break
    assert active > 0                                         # the limiter acted at least once
    assert env_mode("perjoint") == "perjoint"
    with pytest.raises(AssertionError):
        HoppingLegEnv(mode="perjoint", estimator="ekf")


def test_new_eval_files_have_their_own_names_and_parse_into_their_groups():
    assert eval_filename(controller="perjoint") == "eval_ctl-perjoint.csv"
    assert eval_filename(margin_i=0) == "eval_mi0.0.csv" and eval_filename(margin_i=1.0) == "eval_mi1.0.csv"
    assert eval_filename(margin_v=0) == "eval_mv0.0.csv" and eval_filename(margin_v=0.5) == "eval_mv0.5.csv"
    assert eval_filename() == "eval.csv"                      # old names unchanged
    with pytest.raises(ValueError):
        eval_filename(controller="vra")
    spec = importlib.util.spec_from_file_location("aggabl", ROOT / "scripts" / "aggregate_ablations.py")
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    assert agg.setting_of("eval_ctl-perjoint.csv") == ("controller", "perjoint")
    assert agg.setting_of("eval_mi1.0.csv") == ("margin", "delta_I=1.0")
    assert agg.setting_of("eval_mv0.0.csv") == ("margin", "delta_V=0.0")
    assert eval_filename(margin_v=0, margin_i=0) == "eval_mv0.0_mi0.0.csv"
    assert agg.setting_of("eval_mv0.0_mi0.0.csv") == ("margin", "delta_V=0.0, delta_I=0.0")
    assert agg.setting_of("eval_est-ekf_r2.0_n0.0.csv") == ("estimator", "ekf, r0_scale=2.0")   # unchanged
    assert agg.setting_of("eval_g0.6.csv") == ("gamma", "gamma=0.6")
