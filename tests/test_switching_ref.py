"""Switching-level drive reference (busaware/switching_ref.py): internal consistency."""
import numpy as np

from busaware import Params
from busaware.switching_ref import simulate, F_PWM

P = Params.variant("paux_0w")


def test_current_loop_tracks_at_standstill():
    out = simulate(P, 0.0, lambda t: 10.0, t_end=0.02, soc=0.5)
    tail = slice(-int(0.002 * F_PWM), None)                       # last 2 ms
    assert abs(out["i_q"][tail].mean() - 10.0) < 0.2              # 2 % of command
    assert abs(out["i_d"][tail].mean()) < 0.2


def test_dc_energy_equals_copper_mechanical_and_magnetic():
    """Energy from the DC link = copper loss + mechanical work + change in magnetic energy,
    for a torque step at speed (motoring)."""
    out = simulate(P, 20.0, lambda t: 0.0 if t < 0.005 else 15.0, t_end=0.02, soc=0.5)
    ts = 1.0 / F_PWM
    e_dc = np.sum(out["p_elec"] - P.p_idle) * ts
    e_cu, e_mech = np.sum(out["p_cu"]) * ts, np.sum(out["p_mech"]) * ts
    f = out["final"]
    e_mag = 0.75 * P.l_phase * (f["i_d"] ** 2 + f["i_q"] ** 2)  # starts from zero current
    assert abs(e_dc - (e_cu + e_mech + e_mag)) / abs(e_dc) < 0.01
