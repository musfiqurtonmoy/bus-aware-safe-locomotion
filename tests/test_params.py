"""params.py must agree with the derived battery values (results/pack_proposal/per_leg.json)."""
import json
import pathlib
from dataclasses import fields

import numpy as np
import pytest

from busaware import Params

PER_LEG = pathlib.Path(__file__).resolve().parent.parent / "results" / "pack_proposal" / "per_leg.json"
KEYS = ("q_ah", "r0", "r1", "c1", "v_min", "i_bat_max", "i_chg_max", "p_aux")


def test_base_matches_per_leg_json():
    d = json.loads(PER_LEG.read_text())["base"]
    p = Params()
    assert p.n_series == d["n_series"]
    for k in KEYS:
        assert getattr(p, k) == pytest.approx(d[k], rel=1e-8), k
    np.testing.assert_allclose(p.ocv_soc, d["ocv_soc"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(p.ocv_v, d["ocv_v"], rtol=0, atol=1e-12)


def test_cold_matches_per_leg_json_and_changes_only_battery():
    d = json.loads(PER_LEG.read_text())["cold"]
    p, base = Params.cold(), Params()
    for k in ("q_ah", "r0", "r1", "c1", "i_chg_max"):
        assert getattr(p, k) == pytest.approx(d[k], rel=1e-8), k
    np.testing.assert_allclose(p.ocv_v, d["ocv_v"], rtol=0, atol=1e-12)
    changed = {f.name for f in fields(Params)
               if not np.array_equal(np.asarray(getattr(p, f.name)), np.asarray(getattr(base, f.name)))}
    assert changed == {"q_ah", "r0", "r1", "c1", "ocv_v", "i_chg_max"}


NON_BATTERY = {"rphase_x0.5": dict(r_phase=0.0525), "rphase_x1.5": dict(r_phase=0.1575),
               "current_lag": dict(i_loop_bw=1000.0), "gear_eff_0.9": dict(gear_eff=0.9)}


@pytest.mark.parametrize("name", Params.VARIANTS)
def test_variants_match_per_leg_json_and_change_one_field(name):
    """Battery variants must match per_leg.json; the others change only their own field."""
    d = NON_BATTERY[name] if name in NON_BATTERY else json.loads(PER_LEG.read_text())["variants"][name]
    p, base = Params.variant(name), Params()
    for k, v in d.items():
        assert getattr(p, k) == pytest.approx(v, rel=1e-8), k
    changed = {f.name for f in fields(Params)
               if not np.array_equal(np.asarray(getattr(p, f.name)), np.asarray(getattr(base, f.name)))}
    assert changed == set(d)


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        Params.variant("no_such_variant")


def test_mini_cheetah_battery_check():
    """Per-leg continuous battery power, i_bat_max x nominal pack voltage, against the Mini
    Cheetah's 6S2P pack (960 W continuous / 4 legs, Katz 2018 Sec. 3.2.3): computed from the
    current Params, and not above the real robot's per-leg capability."""
    c = json.loads(PER_LEG.read_text())["check_mini_cheetah"]
    p = Params()
    assert c["ours_i_bat_max_x_v_nominal_per_leg_W"] == pytest.approx(p.i_bat_max * p.n_series * 3.6, rel=1e-12)
    assert c["mini_cheetah_p_cont_per_leg_W"] == pytest.approx(960.0 / 4)
    assert c["ours_i_bat_max_x_v_nominal_per_leg_W"] <= c["mini_cheetah_p_cont_per_leg_W"]
