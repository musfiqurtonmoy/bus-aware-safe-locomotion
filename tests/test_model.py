"""MuJoCo leg model: Go1 quarter masses and static stand torques."""
import mujoco
import numpy as np
import pytest

from busaware import Params
from busaware.leg_env import ASSET, stand_static_torques

# MuJoCo Menagerie unitree_go1/go1.xml @ 370c43965323c0944b433b202aac5058b5f907a9 (FR leg)
GO1 = dict(trunk=5.204, hip=0.68, thigh=1.009, calf=0.195862)


def test_masses_are_go1_quarter():
    m = mujoco.MjModel.from_xml_path(str(ASSET))
    mass = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): m.body_mass[i] for i in range(1, m.nbody)}
    assert mass["torso"] == pytest.approx(GO1["trunk"] / 4 + GO1["hip"], abs=1e-9)
    assert mass["thigh"] == pytest.approx(GO1["thigh"], abs=1e-9)
    assert mass["shank"] == pytest.approx(GO1["calf"], abs=1e-9)
    assert sum(mass.values()) == pytest.approx(sum(GO1.values()) - 3 * GO1["trunk"] / 4, abs=1e-9)


def test_stand_static_torques_below_half_peak():
    """Static stand torques below 50 % of the drive's peak output torque."""
    p = Params()
    tau, f_z = stand_static_torques()
    m = mujoco.MjModel.from_xml_path(str(ASSET))
    assert f_z == pytest.approx(m.body_mass[1:].sum() * 9.81, rel=1e-6)   # foot carries the weight
    peak = p.tau_peak                      # min(i_max x kt_out, 17 N m) = 17 N m
    assert np.all(np.abs(tau) < 0.5 * peak), (tau, peak)


def test_current_lag_is_first_order_and_off_by_default():
    from busaware.drive import current_lag
    assert Params().i_loop_bw == 0.0
    np.testing.assert_array_equal(current_lag([1.0, -2.0], [5.0, 3.0], 0.0, 5e-4), [5.0, 3.0])
    bw, dt = 1000.0, 1e-6
    tau = 1.0 / (2 * np.pi * bw)
    x = np.zeros(1)
    for _ in range(int(round(tau / dt))):
        x = current_lag(x, [1.0], bw, dt)
    assert x[0] == pytest.approx(1.0 - np.exp(-1.0), abs=2e-3)       # 63.2 % after one time constant


def test_current_lag_variant_acts_in_the_plant():
    """With the current_lag variant the plant's actual currents differ from the ideal loop;
    the filter is unchanged (it never reads i_loop_bw)."""
    from busaware.leg_env import HoppingLegEnv
    acts = np.array([[0.8, -0.6], [-0.5, 0.9], [0.3, 0.3]])
    out = {}
    for name, prm in (("ideal", Params()), ("lag", Params.variant("current_lag"))):
        env = HoppingLegEnv(mode="filter", params=prm, seed=0)
        env.reset(seed=0, options=dict(soc=0.6))
        for a in acts:
            env.step(a)
        out[name] = env.iq_act.copy()
    assert np.all(np.isfinite(out["lag"])) and not np.allclose(out["lag"], out["ideal"])
    src = __import__("inspect").getsource(__import__("busaware.safety_filter", fromlist=["x"]))
    assert "i_loop_bw" not in src


def test_gear_efficiency_scales_motoring_torque_only():
    from busaware.drive import output_torque
    assert Params().gear_eff == 1.0
    tau, w = np.array([5.0, 5.0, -5.0, -5.0, 5.0]), np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    np.testing.assert_array_equal(output_torque(tau, w, 1.0), tau)
    # motoring (tau * w > 0) scaled; generating and standstill unchanged
    np.testing.assert_allclose(output_torque(tau, w, 0.9), [4.5, 5.0, -5.0, -4.5, 5.0])


def test_bus_capacitance_from_thesis():
    """1.5 J at 24 V across 12 controllers (Katz 2018 p. 72), two drives per leg."""
    assert Params().c_bus == pytest.approx(2 * (2 * 1.5 / 24.0 ** 2) / 12, rel=1e-12)
    assert Params().c_bus == pytest.approx(0.868e-3, rel=1e-3)


def test_joint_limits_are_radians_and_contain_the_stand_pose():
    """leg.xml ranges are in radians (model_provenance.md): hip +-1.6, knee -2.818..-0.888 (Go1).
    MJCF defaults to degrees, which would pin the leg nearly straight."""
    m = mujoco.MjModel.from_xml_path(str(ASSET))
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    np.testing.assert_allclose(m.jnt_range[jid("hip")], [-1.6, 1.6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.jnt_range[jid("knee")], [-2.818, -0.888], rtol=0, atol=1e-12)
    assert m.jnt_limited[jid("hip")] and m.jnt_limited[jid("knee")]
    key = m.key_qpos[0]
    for name, q in (("hip", key[1]), ("knee", key[2])):
        lo, hi = m.jnt_range[jid(name)]
        assert lo < q < hi, name
