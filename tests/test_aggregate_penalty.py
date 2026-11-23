"""CHANGELOG #56: scripts/aggregate_ablations.py reads the warm-started penalty runs with the filter at
test time under the variants and barrier rates (load_extra), on a small synthetic results folder
(tmp_path; the real results/ is not read)."""
import importlib.util
import json
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggabl_test", ROOT / "scripts" / "aggregate_ablations.py")
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

RUN = "penalty_w0.5_ws"
SOCS = (1.0, 0.5, 0.2)
OFFSET = {"base": 0.0, "cold": -0.05, "gamma": 0.02, "other": 7.0}   # height offset per setting


def _episodes(run, seed, test_mode, setting, n_ep):
    rows = []
    for soc in SOCS:
        for ep in range(n_ep):
            h = 0.1 + 0.01 * seed + 0.001 * ep + OFFSET[setting]
            if ep >= 10:                     # episodes the base reference must not use
                h = 5.0
            if test_mode != "filter":        # penalty-mode rows must not enter the extra rows
                h = 9.0
            fell = int(setting == "cold" and seed == 0 and ep == 0)
            rows.append(dict(run=f"{run}_s{seed}", train_mode="penalty", test_mode=test_mode, soc=soc,
                             episode=ep, reset_seed=1000 + ep, fell=fell, ret=1.0, n_hops=6 + seed,
                             height_max=h, v_min=20.0, i_max=9.5, uv_s=0.0, oc_s=0.0, ov_s=0.0,
                             collapse_s=0.0, infeas_s=0.001 * seed, filt_frac=0.3 + 0.01 * seed,
                             **{c: 0 for c in agg.COUNTERS}))
    return pd.DataFrame(rows)


def _make_results(tmp, n_ep_settings=10):
    for s in agg.SEEDS:
        d = tmp / f"{RUN}_s{s}"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"run": f"{RUN}_s{s}", "mode": "penalty", "seed": s}))
        pd.concat([_episodes(RUN, s, "penalty", "base", 12), _episodes(RUN, s, "filter", "base", 12)]) \
            .to_csv(d / "eval.csv", index=False)
        _episodes(RUN, s, "filter", "cold", n_ep_settings).to_csv(d / "eval_var-cold.csv", index=False)
        _episodes(RUN, s, "filter", "gamma", n_ep_settings).to_csv(d / "eval_g0.6.csv", index=False)
        # groups the extra runs must ignore: model error and the joint-level limiter
        _episodes(RUN, s, "filter", "other", n_ep_settings).to_csv(d / "eval_r1.5_n0.0.csv", index=False)
        _episodes(RUN, s, "filter", "other", n_ep_settings).to_csv(d / "eval_ctl-perjoint.csv", index=False)
    # a baseline_ws run in the same folder is not an extra run
    b = tmp / "baseline_ws_s0"
    b.mkdir()
    (b / "config.json").write_text(json.dumps({"run": "baseline_ws_s0", "mode": "baseline", "seed": 0}))
    _episodes("baseline_ws", 0, "filter", "other", 10).to_csv(b / "eval_var-cold.csv", index=False)


def test_penalty_rows_metrics_and_selection(tmp_path):
    _make_results(tmp_path)
    df, n = agg.load_extra(results=tmp_path)
    assert n == 10
    out = agg.summarise(df)
    assert set(out["method"]) == {f"{RUN}+filter@test"}
    assert set(zip(out["ablation"], out["setting"])) == {("base", "base"), ("variant", "cold"), ("gamma", "gamma=0.6")}
    assert len(out) == 3 * len(SOCS)
    assert (out["n_seeds"] == 3).all() and (out["n_episodes"] == 30).all()
    # the same metric columns as the baseline_ws+filter@test rows of ablations.csv
    for k in agg.MEAN_METRICS:
        assert f"{k}_mean" in out and f"{k}_std" in out
    for c in ("n_fell", "i_max_peak", *agg.COUNTERS):
        assert c in out
    mean_ep = 0.001 * 4.5                    # mean of 0.001 * ep over episodes 0..9
    for (a, st, off) in (("base", "base", OFFSET["base"]), ("variant", "cold", OFFSET["cold"]),
                         ("gamma", "gamma=0.6", OFFSET["gamma"])):
        r = out[(out["ablation"] == a) & (out["setting"] == st)]
        # mean over seeds of the per-seed episode mean (seed means 0.1 + 0.01 s + ...): no
        # penalty-mode rows, no base episode >= 10
        assert r["height_max_mean"].to_numpy() == pytest.approx([0.11 + mean_ep + off] * 3)
        assert r["height_max_std"].to_numpy() == pytest.approx([0.01] * 3)
        assert r["n_hops_mean"].to_numpy() == pytest.approx([7.0] * 3)
        assert r["filt_frac_mean"].to_numpy() == pytest.approx([0.31] * 3)
        assert r["infeas_s_mean"].to_numpy() == pytest.approx([0.001] * 3)
        assert list(r["n_fell"]) == ([1, 1, 1] if st == "cold" else [0, 0, 0])
        assert list(r["soc"]) == list(SOCS)


def test_no_penalty_files_gives_none(tmp_path):
    assert agg.load_extra(results=tmp_path) is None


def test_episode_count_mismatch_raises(tmp_path):
    _make_results(tmp_path)
    _episodes(RUN, 1, "filter", "gamma", 5).to_csv(tmp_path / f"{RUN}_s1" / "eval_g0.1.csv", index=False)
    with pytest.raises(ValueError):
        agg.load_extra(results=tmp_path)
