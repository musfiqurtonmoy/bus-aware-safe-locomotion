import numpy as np
import pytest
from busaware import Params, Battery, Bus, Drive, power_budget, project_currents


def test_bus_rest_and_sag():
    p = Params(); bat = Battery(p, 0.8); bus = Bus(p, bat)
    for _ in range(200):
        s = bus.step(0.0, 5e-4)
    assert abs(s["v"] - bat.emf()) < 1e-3 and abs(s["i_bat"]) < 1e-2
    for _ in range(200):
        s = bus.step(200.0, 5e-4)
    # quasi-static check: V = E - R0 I, P = V I
    assert abs(s["v"] * s["i_bat"] - 200.0) < 1.0
    assert s["v"] < bat.ocv()


def test_energy_balance():
    p = Params(); bat = Battery(p, 0.9); bus = Bus(p, bat)
    dt, e_load, e_bat = 5e-4, 0.0, 0.0
    v0 = bus.v
    for k in range(4000):
        pl = 150.0 + 100.0 * np.sin(0.01 * k)
        s = bus.step(pl, dt)
        e_load += pl * dt; e_bat += s["v"] * s["i_bat"] * dt
    e_cap = 0.5 * p.c_bus * (bus.v ** 2 - v0 ** 2)
    assert abs(e_bat - e_load - e_cap) / e_load < 0.01


def test_regeneration_energy_balance_and_charge_clamp():
    """Negative (regenerative) load: battery charge limit and chopper both engage.

    Bus-side energy must balance: E_bat = E_load + E_chop + dE_cap, and on every step
    (clamped or not, chopper on or not) the backward-Euler capacitor equation, with all
    terms at the end-of-step voltage, must hold exactly.
    """
    p = Params(); bat = Battery(p, 0.9); bus = Bus(p, bat)
    dt, e_load, e_bat, e_chop = 5e-4, 0.0, 0.0, 0.0
    v0, n_clamped, worst = bus.v, 0, 0.0
    for k in range(4000):
        pl = -(120.0 + 80.0 * np.sin(0.01 * k))           # always regenerating
        v_old = bus.v
        s = bus.step(pl, dt)
        e_load += pl * dt; e_bat += s["v"] * s["i_bat"] * dt; e_chop += s["p_chop"] * dt
        n_clamped += s["i_bat"] == -p.i_chg_max
        lhs = p.c_bus * (s["v"] - v_old) / dt
        rhs = s["i_bat"] - pl / s["v"] - s["p_chop"] / s["v"]
        worst = max(worst, abs(lhs - rhs))
    assert n_clamped > 100                                 # the clamp branch was exercised
    assert e_chop > 0.0                                    # and so was the chopper
    e_cap = 0.5 * p.c_bus * (bus.v ** 2 - v0 ** 2)
    assert abs(e_bat - e_load - e_chop - e_cap) / abs(e_load) < 0.01
    assert worst < 1e-9


def test_chopper_steady_regen_settles():
    """Steady 250 W regeneration at the battery charge limit: the averaged chopper, solved
    implicitly at the end-of-step voltage, must settle with no step-to-step oscillation
    above 0.05 V, and its power must equal g(V) V^2 at the solved voltage."""
    p = Params(); bat = Battery(p, 0.9); bus = Bus(p, bat); dt = 5e-4
    vs = []
    for _ in range(4000):
        s = bus.step(-250.0, dt); vs.append(s["v"])
    tail = np.array(vs[-1000:])
    assert s["i_bat"] == -p.i_chg_max                       # at the charge limit
    assert 0.0 < bus.g(s["v"]) < 1.0 / p.r_chop             # chopper inside its band
    assert abs(s["p_chop"] - bus.g(s["v"]) * s["v"] ** 2) < 1e-9
    assert np.abs(np.diff(tail)).max() < 0.05


def test_collapse_threshold_and_current_scaling():
    """max_power is exactly the collapse threshold, and scale_to_power brings the
    drive demand down to it along the commanded current direction."""
    p = Params(); d = Drive(p); rng = np.random.default_rng(5); dt = 5e-4
    n_scaled = 0
    for _ in range(300):
        bat = Battery(p, rng.uniform(0.05, 1.0)); bus = Bus(p, bat)
        bus.v = rng.uniform(16.0, 27.4)
        pm = bus.max_power(dt)
        v_turn = (p.c_bus * bus.v / dt + bat.emf() / p.r0) / (2 * (p.c_bus / dt + 1 / p.r0))
        for frac, expect in ((0.999, False), (1.001, True)):
            b2 = Bus(p, Battery(p, bat.soc)); b2.v = bus.v
            assert b2.step(frac * pm, dt)["collapse"] is expect
        b2 = Bus(p, Battery(p, bat.soc)); b2.v = bus.v
        s = b2.step(pm, dt)                               # exactly at the threshold
        assert not s["collapse"] and abs(s["v"] - v_turn) < 1e-4
        # current scaling against a random power limit between idle power and 1.2x demand
        w = rng.uniform(-30, 30, 2); iq = rng.uniform(-40, 40, 2)
        demand, idle = float(np.sum(d.power(iq, w))), 2 * p.p_idle
        p_lim = idle + rng.uniform(0.01, 1.2) * max(demand - idle, 1.0)
        y, scaled = d.scale_to_power(iq, w, p_lim)
        if scaled:
            n_scaled += 1
            assert abs(float(np.sum(d.power(y, w))) - p_lim) < 1e-9 * max(1.0, abs(p_lim))
            k = y / iq
            assert 0.0 <= k[0] < 1.0 and abs(k[0] - k[1]) < 1e-12   # one common factor
        else:
            assert np.array_equal(y, iq) and float(np.sum(d.power(iq, w))) <= p_lim
    assert n_scaled > 50
    # limit below idle power: best effort is zero current
    y, scaled = d.scale_to_power(np.array([30.0, -20.0]), np.array([10.0, 5.0]), p.p_idle)
    assert scaled and np.array_equal(y, np.zeros(2))


def test_projection_meets_budget_and_is_minimal():
    rng = np.random.default_rng(0); d = Drive(Params())
    for _ in range(200):
        w = rng.uniform(-30, 30, 2); iq = rng.uniform(-40, 40, 2)
        lo, hi = d.bounds(w, rng.uniform(18.0, 27.5))
        a, b = d.power_coeffs(w); budget = rng.uniform(20, 400)
        x, status = project_currents(iq, a, b, budget, lo, hi)
        assert np.all(x >= lo) and np.all(x <= hi)
        x_min = np.clip(-a / (2 * b), lo, hi)
        assert (status == "infeasible") == (np.sum(a * x_min + b * x_min ** 2) > budget)
        if status != "infeasible":
            assert np.sum(a * x + b * x ** 2) <= budget + 1e-6
        if status == "inactive":
            assert np.allclose(x, np.clip(iq, lo, hi))


def test_projection_matches_grid_search():
    """2 joints: the projection must be at least as good as every feasible grid point,
    and (strong convexity of ||x - d||^2) the grid optimum must lie next to it:
    ||x_grid - x*||^2 <= J(x_grid) - J(x*).
    Half the cases use drive-derived bounds; the other half use random narrow boxes
    so that the box and the power constraint bind together (where "project, then
    clip" would be wrong)."""
    rng = np.random.default_rng(2); d = Drive(Params())
    n_compared, n_box_active, statuses = 0, 0, set()
    for case in range(80):
        w = rng.uniform(-35, 35, 2)
        a, b = d.power_coeffs(w); budget = rng.uniform(-100, 400)
        if case % 2 == 0:
            iq = rng.uniform(-40, 40, 2); lo, hi = d.bounds(w, rng.uniform(18.0, 27.5))
        else:   # narrow box, budget strictly between box-min power and power at clip(d)
            iq = rng.uniform(-60, 60, 2)
            lo = rng.uniform(-40, 20, 2); hi = lo + rng.uniform(2, 25, 2)
            pw = lambda z: float(np.sum(a * z + b * z ** 2))
            p_lo, p_hi = pw(np.clip(-a / (2 * b), lo, hi)), pw(np.clip(iq, lo, hi))
            budget = p_lo + rng.uniform(0.05, 0.95) * (p_hi - p_lo)
        x, status = project_currents(iq, a, b, budget, lo, hi)
        statuses.add(status)
        X0, X1 = np.meshgrid(np.linspace(lo[0], hi[0], 801),
                             np.linspace(lo[1], hi[1], 801), indexing="ij")
        P = a[0] * X0 + b * X0 ** 2 + a[1] * X1 + b * X1 ** 2
        px = float(np.sum(a * x + b * x ** 2))
        feas = P <= budget
        if not feas.any():
            assert px <= P.min() + 1e-9          # infeasible: x has the least power on the box
            continue
        assert status != "infeasible"
        assert px <= budget + 1e-9
        J = np.where(feas, (X0 - iq[0]) ** 2 + (X1 - iq[1]) ** 2, np.inf)
        k = np.argmin(J); xg = np.array([X0.flat[k], X1.flat[k]]); jg = J.flat[k]
        jp = float(np.sum((x - iq) ** 2))
        assert jp <= jg + 1e-6
        assert np.sum((xg - x) ** 2) <= (jg - jp) + 1e-6
        n_compared += 1
        n_box_active += status == "active" and bool(np.any((x == lo) | (x == hi)))
    assert n_compared >= 40 and n_box_active >= 10
    assert {"inactive", "active", "infeasible"} <= statuses


def test_budget_is_exact_one_step_condition():
    """Apply exactly the budget to the plant from random states (inside and outside the
    limits, including bus voltages in and above the chopper band): both discrete
    barrier conditions must hold,
    h(k+1) >= (1 - gamma) h(k). At least one must be tight unless the budget is the
    bus's maximum transferable power, or the plant's battery charge clamp engaged
    (the filter's model has no charge limit; the clamp only raises V', so the
    conditions then hold with slack)."""
    p = Params(); rng = np.random.default_rng(4); dt, gamma = 5e-4, 0.3
    chop = (p.v_chop_off, p.v_chop, p.r_chop)
    n_chop, n_viol = 0, 0
    for k in range(2400):
        if k < 2000:            # broad states
            soc, vrc, v0 = rng.uniform(0.05, 1.0), rng.uniform(-1.5, 1.5), rng.uniform(16.0, 34.0)
        else:                   # high E: V_req in the chopper band without hitting the charge limit
            soc, vrc, v0 = rng.uniform(0.9, 1.0), rng.uniform(-1.5, -0.5), rng.uniform(26.0, 34.0)
        bat = Battery(p, soc); bat.v_rc = vrc
        bus = Bus(p, bat)
        bus.v = v0; bus.i_bat = rng.uniform(-p.i_chg_max, 30.0)
        e, v, i = bat.emf(), bus.v, bus.i_bat
        bud = power_budget(v, e, p.r0, p.c_bus, dt, p.v_min, p.i_bat_max, i, gamma, chopper=chop)
        s = bus.step(bud, dt)
        h1, h2 = v - p.v_min, p.i_bat_max - i
        h1n, h2n = s["v"] - p.v_min, p.i_bat_max - s["i_bat"]
        assert h1n >= (1 - gamma) * h1 - 1e-6
        assert h2n >= (1 - gamma) * h2 - 1e-6
        alpha = p.c_bus / dt + 1 / p.r0; beta = p.c_bus * v / dt + e / p.r0
        tight = min(h1n - (1 - gamma) * h1, h2n - (1 - gamma) * h2) < 1e-6
        clamped = s["i_bat"] == -p.i_chg_max
        assert tight or clamped or abs(bud - beta ** 2 / (4 * alpha)) < 1e-6 * beta ** 2 / alpha
        n_chop += tight and s["p_chop"] > 0; n_viol += (h1 < 0) or (h2 < 0)
    assert n_chop > 50 and n_viol > 100     # tight with the chopper on, and violated states, covered


@pytest.mark.parametrize("p", [Params(), Params.variant("paux_10w")], ids=["paux_5w", "paux_10w"])
def test_filter_keeps_bus_inside_limits_with_aux_load(p):
    """Constant computer/sensor load p_aux on the bus; the budget subtracts it like the
    drives' idle power. V >= v_min and I_bat <= i_bat_max must still hold."""
    assert p.p_aux > 0
    d = Drive(p); rng = np.random.default_rng(6)
    for soc in (0.2, 0.5, 1.0):
        bat = Battery(p, soc); bus = Bus(p, bat); dt = 5e-4
        for k in range(8000):
            w = 25 * np.sin(0.002 * k + np.array([0, 1.0])); iq_des = rng.uniform(-40, 40, 2)
            bud = power_budget(bus.v, bat.emf(), p.r0, p.c_bus, dt, p.v_min, p.i_bat_max,
                               bus.i_bat, chopper=(p.v_chop_off, p.v_chop, p.r_chop)) - 2 * p.p_idle - p.p_aux
            a, b = d.power_coeffs(w); lo, hi = d.bounds(w, bus.v)
            iq, _ = project_currents(iq_des, a, b, bud, lo, hi)
            s = bus.step(float(np.sum(d.power(iq, w))) + p.p_aux, dt)
            assert s["v"] >= p.v_min - 1e-6 and s["i_bat"] <= p.i_bat_max + 1e-6


def test_filter_keeps_bus_inside_limits():
    p = Params(); d = Drive(p); rng = np.random.default_rng(1)
    for soc in (0.2, 0.5, 1.0):
        bat = Battery(p, soc); bus = Bus(p, bat); dt = 5e-4
        for k in range(8000):
            w = 25 * np.sin(0.002 * k + np.array([0, 1.0])); iq_des = rng.uniform(-40, 40, 2)
            bud = power_budget(bus.v, bat.emf(), p.r0, p.c_bus, dt, p.v_min, p.i_bat_max,
                               bus.i_bat, chopper=(p.v_chop_off, p.v_chop, p.r_chop)) - 2 * p.p_idle
            a, b = d.power_coeffs(w); lo, hi = d.bounds(w, bus.v)
            iq, _ = project_currents(iq_des, a, b, bud, lo, hi)
            s = bus.step(float(np.sum(d.power(iq, w))), dt)
            assert s["v"] >= p.v_min - 1e-6 and s["i_bat"] <= p.i_bat_max + 1e-6
