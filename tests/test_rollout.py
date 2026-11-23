"""Rollout helper and hop counting (Step 3), without a trained model."""
import numpy as np

from busaware.leg_env import HoppingLegEnv
from busaware.rollout import rollout, count_hops


def test_count_hops_on_a_known_contact_pattern():
    """Hop rule (CHANGELOG #18): flight >= 20 ms, ended by contact within the episode,
    apex above the start height."""
    dt = 0.004
    t = dt * np.arange(1, 61)
    contact = np.ones(60, int)
    z = np.zeros(60)
    contact[0:8] = 0;   z[0:8] = -0.05     # 32 ms at the start, never above the start: no
    contact[12:19] = 0; z[15] = 0.12       # 28 ms, apex 0.12: hop
    contact[22:26] = 0; z[24] = 0.30       # 16 ms: too short
    contact[30:37] = 0; z[33] = 0.001      # 28 ms, apex just above the start: hop
    contact[40:47] = 0; z[40:47] = 0.0     # 28 ms, apex equal to the start: no
    contact[50:60] = 0; z[55] = 0.20       # 40 ms, cut off by the episode end: no
    hops = count_hops(t, contact, z, z_start=0.0)
    assert [round(h["apex"], 3) for h in hops] == [0.12, 0.001]
    assert np.isclose(hops[0]["duration"], 0.028)
    assert np.isclose(hops[0]["start"], t[12]) and np.isclose(hops[0]["end"], t[18])
    # the same flights, with contact in the last step: the final flight now counts
    contact[59] = 1
    assert [round(h["apex"], 3) for h in count_hops(t, contact, z, z_start=0.0)] == [0.12, 0.001, 0.2]
    # a higher start height removes the low hop
    assert len(count_hops(t, contact, z, z_start=0.05)) == 2


def test_rollout_runs_with_a_dummy_policy():
    env = HoppingLegEnv(mode="baseline", seed=0, episode_s=0.2)
    r = rollout(env, lambda obs: np.zeros(2), soc=0.6, seed=0)
    n = len(r["t"])
    assert n == env.max_steps
    assert all(len(r[k]) == n for k in r if k not in ("terminated", "z0"))
    assert r["z0"] == 0.0                     # stand keyframe: rail at 0 at the start
    assert np.all(np.isfinite(r["v_bus"])) and r["contact"].max() == 1   # standing on the floor


def test_reset_starts_the_rail_at_the_keyframe_for_every_seed():
    """aggregate.py relies on this for eval.csv files written before z0 was recorded:
    reset perturbs only the joint angles, so the start height z0 is 0 (CHANGELOG #20)."""
    env = HoppingLegEnv(mode="baseline", seed=123)
    for seed in range(1000, 1020):
        for soc in (1.0, 0.5, 0.2):
            env.reset(seed=seed, options=dict(soc=soc))
            assert env.data.qpos[0] == 0.0


def test_episode_metrics_times_and_hops():
    from busaware.rollout import episode_metrics
    env = HoppingLegEnv(mode="baseline", seed=0, episode_s=0.2)
    r = rollout(env, lambda obs: np.ones(2), soc=0.2, seed=0)
    m = episode_metrics(r, env.dt_sim, env.substeps, env.max_steps)
    n_sub = env.max_steps * env.substeps
    for k in ("uv", "oc", "ov", "collapse", "filt", "infeas"):
        assert np.isclose(m[f"{k}_s"], r[k].sum() * env.dt_sim)
        assert np.isclose(m[f"{k}_frac"], r[k].sum() / n_sub)
    assert m["n_hops"] == len(count_hops(r["t"], r["contact"], r["z"], r["z0"]))
    assert np.isclose(m["ret"], r["reward"].sum()) and m["v_min"] == r["v_bus_min"].min()
    assert m["height_max"] == r["z"].max() - r["z0"]


def test_proof_assumption_counters():
    """A3 (max(V, E) <= 2 V_c,off) and the Remark 1 bound (V <= 2E + 2 R0 I_chg + I_chg dt/C)
    are counted per substep (CHANGELOG #23): zero in normal operation, counted when violated."""
    env = HoppingLegEnv(mode="filter", seed=0)
    env.reset(seed=0, options=dict(soc=0.5))
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a3_fail"] == 0 and info["rem1_fail"] == 0
    p, e = env.p_plant, env.bat.emf()
    env.bus.v = 2.0 * p.v_chop_off + 1.0          # above 2 V_c,off (and above the Remark 1 bound)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["a3_fail"] >= 1
    assert 2.0 * p.v_chop_off + 1.0 > 2.0 * e + 2.0 * p.r0 * p.i_chg_max + p.i_chg_max * env.dt_sim / p.c_bus
    assert info["rem1_fail"] >= 1
    r = rollout(HoppingLegEnv(mode="baseline", seed=0, episode_s=0.2), lambda o: np.zeros(2), 0.5, 0)
    from busaware.rollout import episode_metrics
    m = episode_metrics(r, 5e-4, 8, len(r["t"]))
    assert m["a3_fail_n"] == 0 and m["rem1_fail_n"] == 0
