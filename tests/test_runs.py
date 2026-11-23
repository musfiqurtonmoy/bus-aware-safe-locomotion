"""Penalty weight, run names/configuration, warm-start action sampling (CHANGELOG #19)."""
import json

import numpy as np
import pytest

from busaware.leg_env import HoppingLegEnv
from busaware.runs import run_name, run_config


def _rewards(mode, penalty_w, n=60):
    env = HoppingLegEnv(mode=mode, seed=0, penalty_w=penalty_w)
    env.reset(seed=0, options=dict(soc=0.2))
    out, viol = [], 0
    for _ in range(n):                       # hard extension at low SOC: bus violations
        _, r, term, _, info = env.step(np.array([-1.0, 1.0]))
        out.append(r); viol += info["uv"] + info["oc"]
        if term:
            break
    return np.array(out), viol


def test_penalty_weight_scales_only_the_penalty_term():
    r0, viol = _rewards("penalty", 0.0)
    assert viol > 0
    r05, _ = _rewards("penalty", 0.5)
    r1, _ = _rewards("penalty", 1.0)
    assert np.any(r1 < r0)
    np.testing.assert_allclose(r0 - r05, 0.5 * (r0 - r1), rtol=0, atol=1e-12)
    # default 0.5 (the value the from-scratch penalty runs were trained with)
    assert HoppingLegEnv(mode="penalty").penalty_w == 0.5
    # baseline mode ignores it
    np.testing.assert_array_equal(_rewards("baseline", 0.0)[0], _rewards("baseline", 1.0)[0])


def test_run_names_and_config(tmp_path):
    assert run_name("baseline", 0) == "baseline_s0"
    assert run_name("penalty", 2) == "penalty_s2"
    assert run_name("filter", 0, init="baseline_s0") == "filter_ws_s0"
    assert run_name("penalty", 1, 0.05, "baseline_s1") == "penalty_w0.05_ws_s1"
    assert run_name("penalty", 1, 0.1, "baseline_s1") == "penalty_w0.1_ws_s1"
    assert run_name("baseline", 2, init="baseline_s2") == "baseline_ws_s2"
    assert run_name("penalty", 0, 0.2, "baseline_s0") == "penalty_w0.2_ws_s0"
    assert run_name("penalty", 0, 0.5, "baseline_s0") == "penalty_w0.5_ws_s0"   # default weight, warm start
    assert run_name("penalty", 0, 0.5) == "penalty_s0"                         # from scratch: unchanged
    assert run_name("penalty", 0, 2.0, "baseline_s0") == "penalty_w2.0_ws_s0"
    assert run_name("lagrangian", 1, init="baseline_s1") == "lagrangian_ws_s1"
    # names of runs that already exist on disk are unchanged by the repr formatting
    for w in (0.05, 0.1, 0.2, 0.5):
        assert run_name("penalty", 0, w, "baseline_s0") == f"penalty_w{w:g}_ws_s0"
    c = run_config(tmp_path / "penalty_s1")                  # from-scratch run, no config.json
    assert (c["mode"], c["seed"], c["penalty_w"], c["w_energy"], c["init"]) == ("penalty", 1, 0.5, 0.0, None)
    d = tmp_path / "penalty_w0.05_ws_s0"; d.mkdir()
    (d / "config.json").write_text(json.dumps(dict(mode="penalty", seed=0, penalty_w=0.05, init="baseline_s0")))
    c = run_config(d)
    assert (c["mode"], c["penalty_w"], c["init"]) == ("penalty", 0.05, "baseline_s0")
    with pytest.raises(ValueError):
        run_config(tmp_path / "filter_ws_s0")                # warm start without config.json


def _warm_start_class():
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    src = path.read_text(encoding="utf-8")
    ns = {"__file__": str(path), "__name__": "train_head"}
    exec(src[:src.index("ap = argparse.ArgumentParser()")], ns)   # imports + WarmStartSAC only
    return ns["WarmStartSAC"]


def test_warm_start_warmup_uses_the_policy_not_random_actions():
    from stable_baselines3 import SAC
    env = HoppingLegEnv(seed=0)
    marker = np.array([[0.3, -0.7]], np.float32)
    for cls, from_policy in ((_warm_start_class(), True), (SAC, False)):
        m = cls("MlpPolicy", env, seed=0, device="cpu", learning_starts=5000, policy_kwargs=dict(net_arch=[8]))
        m._last_obs = np.zeros((1, 11), np.float32)
        m.num_timesteps = 0                                   # inside the warm-up phase
        m.predict = lambda obs, state=None, episode_start=None, deterministic=False: (marker.copy(), None)
        acts = np.array([m._sample_action(m.learning_starts)[0][0] for _ in range(50)])
        if from_policy:
            np.testing.assert_allclose(acts, np.repeat(marker, 50, 0), atol=1e-6)
        else:                                                 # SB3 default: uniform random actions
            assert not np.allclose(acts, marker) and acts.std(0).min() > 0.3


def test_eval_filenames_keep_old_names_and_separate_settings():
    from busaware.runs import eval_filename
    assert eval_filename() == "eval.csv"
    assert eval_filename(r0_scale=1.5) == "eval_r1.5_n0.0.csv"              # name used before Step 9
    assert eval_filename(est_noise=0.02) == "eval_r1.0_n0.02.csv"
    assert eval_filename(variant="cold") == "eval_var-cold.csv"
    assert eval_filename(gamma=0.6) == "eval_g0.6.csv" and eval_filename(gamma=1) == "eval_g1.0.csv"
    names = {eval_filename(v) for v in __import__("busaware.runs", fromlist=["x"]).EVAL_VARIANTS}
    names |= {eval_filename(gamma=g) for g in (0.1, 0.6, 1.0)}
    names |= {eval_filename(r0_scale=r) for r in (1.5, 2.0)} | {eval_filename(est_noise=n) for n in (0.02, 0.05)}
    assert len(names) == 10 + 3 + 4 and "eval.csv" not in names


def test_variant_params_reach_the_env():
    from busaware.params import Params
    from busaware.runs import EVAL_VARIANTS, variant_params
    assert variant_params(None) is None and variant_params("base") is None
    assert set(EVAL_VARIANTS) == {"cold", *Params.VARIANTS}
    same = lambda a, b: a.__dict__.keys() == b.__dict__.keys() and all(
        np.array_equal(a.__dict__[k], b.__dict__[k]) for k in a.__dict__)
    assert same(variant_params("cold"), Params.cold())
    for name in Params.VARIANTS:
        env = HoppingLegEnv(mode="filter", seed=0, params=variant_params(name), r0_scale=1.5)
        assert same(env.p, Params.variant(name)) and not same(env.p, Params())
        assert env.p_plant.r0 == 1.5 * env.p.r0                     # r0_scale on top of the variant
    with pytest.raises(KeyError):
        variant_params("no_such_variant")


def test_eval_done_requires_counters_and_every_requested_episode(tmp_path):
    import csv
    from busaware.runs import eval_done
    f = tmp_path / "eval.csv"
    assert not eval_done(f, ["filter"], [1.0], 2)

    def write(cols, rows):
        with open(f, "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=cols); w.writeheader(); w.writerows(rows)

    cnt = ["a3_fail_n", "rem1_fail_n", "a4_fail_n", "a6_empty_n"]
    rows = [dict(test_mode=tm, soc=s, episode=e, **{c: 0 for c in cnt})
            for tm in ("baseline", "filter") for s in (1.0, 0.2) for e in range(2)]
    write(["test_mode", "soc", "episode"], [{k: r[k] for k in ("test_mode", "soc", "episode")} for r in rows])
    assert not eval_done(f, ["baseline", "filter"], [1.0, 0.2], 2)       # old file: no counters
    write(["test_mode", "soc", "episode"] + cnt, rows)
    assert eval_done(f, ["baseline", "filter"], [1.0, 0.2], 2)
    assert eval_done(f, ["filter"], [0.2], 2)
    assert not eval_done(f, ["baseline", "filter"], [1.0, 0.5, 0.2], 2)   # SOC missing
    assert not eval_done(f, ["filter"], [1.0], 3)                          # fewer episodes
    assert not eval_done(f, ["lagrangian"], [1.0], 2)                      # mode missing
    write(["test_mode", "soc", "episode"] + cnt, rows[:-1])
    assert not eval_done(f, ["filter"], [0.2], 2)                          # incomplete


def test_existing_current_code_evaluations_count_as_done():
    """Files written by the current code count as done; a file written before the counters does
    not. Since the final batch (#29) every eval.csv has the counters (filter_ws_s1/eval.csv was
    the pre-counter example until then); eval_pre_eps.csv (filter_ws_s0, before #23) still lacks
    them."""
    import pathlib
    from busaware.runs import eval_done
    res = pathlib.Path(__file__).resolve().parent.parent / "results"
    if not (res / "lagrangian_ws_s0" / "eval.csv").exists():
        pytest.skip("results not present")
    assert eval_done(res / "lagrangian_ws_s0" / "eval.csv", ["lagrangian", "filter"], [1.0, 0.5, 0.2], 20)
    assert eval_done(res / "filter_ws_s0" / "eval.csv", ["filter"], [1.0, 0.5, 0.2], 20)
    assert eval_done(res / "filter_ws_s1" / "eval.csv", ["filter"], [1.0, 0.5, 0.2], 20)
    assert (res / "filter_ws_s0" / "eval_pre_eps.csv").exists()
    assert not eval_done(res / "filter_ws_s0" / "eval_pre_eps.csv", ["filter"], [1.0, 0.5, 0.2], 20)


def test_observation_scaling_is_the_trained_one_in_every_variant():
    """The policy's bus-voltage observation is (V - 16.8)/6 in every evaluation variant: v_min may
    move (vmin_datasheet), the observation reference does not (CHANGELOG #28). In baseline mode,
    where v_min acts only on the violation counters, vmin_datasheet then gives bit-identical
    observations, rewards and dynamics to the base parameters."""
    from busaware.runs import EVAL_VARIANTS, variant_params
    for name in EVAL_VARIANTS:
        env = HoppingLegEnv(mode="filter", seed=0, params=variant_params(name))
        obs, _ = env.reset(seed=0, options=dict(soc=0.5))
        assert env.p.v_obs_ref == 16.8
        assert obs[-2] == np.float32((env.bus.v - 16.8) / 6.0), name
    rng = np.random.default_rng(5)
    acts = rng.uniform(-1.0, 1.0, (200, 2))
    out = []
    for prm in (None, variant_params("vmin_datasheet")):
        env = HoppingLegEnv(mode="baseline", seed=0, params=prm)
        o, _ = env.reset(seed=0, options=dict(soc=0.2))
        rec = [o]
        for a in acts:
            o, r, term, trunc, info = env.step(a)
            rec += [o, np.array([r])]
            if term or trunc:
                break
        out.append(np.concatenate(rec))
    np.testing.assert_array_equal(out[0], out[1])
