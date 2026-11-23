"""Print the leg's body masses and the static stand torques as a share of drive peak torque.

    python scripts/check_model.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import mujoco
from busaware import Params
from busaware.leg_env import ASSET, stand_static_torques

m = mujoco.MjModel.from_xml_path(str(ASSET))
for i in range(1, m.nbody):
    print(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i):6s} {m.body_mass[i]:.6f} kg")
print(f"total moving mass {m.body_mass[1:].sum():.6f} kg")
p = Params(); peak = p.tau_peak
tau, f_z = stand_static_torques(m)
print(f"stand keyframe: vertical foot force {f_z:.3f} N; drive peak output torque {peak:.3f} N m "
      f"(min(i_max*kt_out = {p.i_max * p.kt_out:.1f}, tau_max_out = {p.tau_max_out:.1f}))")
for name, t in zip(("hip", "knee"), tau):
    print(f"  {name}: {t:+.4f} N m = {100 * abs(t) / peak:.2f} % of peak")
