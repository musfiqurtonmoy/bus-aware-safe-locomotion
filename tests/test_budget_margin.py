"""The plant's recomputed load never exceeds the exact budget P_bar when the filter's step is
inactive or active (assumption A7 of the proof; budget epsilon, CHANGELOG #23)."""
import numpy as np

import busaware.leg_env as le
from busaware.drive import Drive
from busaware.leg_env import HoppingLegEnv
from busaware.params import Params
from busaware.safety_filter import power_budget, project_currents


def test_recomputed_drive_power_within_exact_budget_random_states():
    p, drive, rng = Params(), Drive(Params()), np.random.default_rng(7)
    dt, chop = 5e-4, (p.v_chop_off, p.v_chop, p.r_chop)
    n_active = 0
    for _ in range(20000):
        v, e = rng.uniform(8.0, 27.5), rng.uniform(18.0, 25.2)
        i_bat, w = rng.uniform(-p.i_chg_max, 25.0), rng.uniform(-40.0, 40.0, 2)
        args = (v, e, p.r0, p.c_bus, dt, p.v_min, p.i_bat_max, i_bat, 0.3, 0.2, 0.5, chop)
        p_bar = power_budget(*args, eps_rel=0.0)
        budget = power_budget(*args) - 2 * p.p_idle - p.p_aux
        lo, hi = drive.bounds(w, v)
        iq_des = np.clip(rng.uniform(-p.i_lim, p.i_lim, 2), lo, hi)
        a, b = drive.power_coeffs(w)
        iq, status = project_currents(iq_des, a, b, budget, lo, hi)
        if status == "infeasible":
            continue
        n_active += status == "active"
        p_load = float(np.sum(drive.power(iq, w))) + p.p_aux        # as leg_env computes it
        assert p_load <= p_bar, (status, p_load - p_bar)
    assert n_active > 1000


def test_plant_load_within_exact_budget_in_the_filter_env(monkeypatch):
    rec = {}
    orig_budget, orig_project = le.power_budget, le.project_currents

    def budget(*args, **kw):
        rec["p_bar"] = orig_budget(*args, **{**kw, "eps_rel": 0.0})
        return orig_budget(*args, **kw)

    def project(*args, **kw):
        out = orig_project(*args, **kw)
        rec["status"] = out[1]
        return out

    monkeypatch.setattr(le, "power_budget", budget)
    monkeypatch.setattr(le, "project_currents", project)
    env = HoppingLegEnv(mode="filter", seed=0)
    rng = np.random.default_rng(1)
    checked = 0
    for soc in (1.0, 0.2):
        env.reset(seed=0, options=dict(soc=soc))
        bus_step = env.bus.step

        def step(p_load, dt):
            nonlocal checked
            if rec["status"] != "infeasible":
                assert p_load <= rec["p_bar"], (rec["status"], p_load - rec["p_bar"])
                checked += 1
            return bus_step(p_load, dt)

        env.bus.step = step
        for _ in range(250):
            _, _, term, trunc, info = env.step(rng.uniform(-1.0, 1.0, 2))
            if term or trunc:
                break
    assert checked > 1000
