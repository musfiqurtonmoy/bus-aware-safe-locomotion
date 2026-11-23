# MuJoCo model provenance (`assets/leg.xml`)

The platform is **one leg of a Go1-scale pronking quadruped, with Mini Cheetah-class
drives**, on a vertical rail. In pronking all four legs move in phase, so each leg carries
a quarter of the trunk; the rail removes trunk pitch, roll and horizontal motion. The leg
keeps a single hip (pitch) and knee joint; the Go1's abduction joint is not modelled, and
its hip-abduction body rides with the torso.

Changed 2026-11-18 (CHANGELOG #11), before any training run.

## Source

| Key | Source |
|---|---|
| Go1-model | MuJoCo Menagerie, `unitree_go1/go1.xml`, commit `370c43965323c0944b433b202aac5058b5f907a9` (BSD-3-Clause), https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/unitree_go1/go1.xml. Per its README, derived from Unitree's public URDF (unitreerobotics/unitree_ros, `go1_description`). Cloned read-only to `data/raw/menagerie/repo`. Citation: `CITATION.cff` of the repository (Zakka, Tassa and the MuJoCo Menagerie Contributors, 2022). |

**Battery inclusion unknown.** Whether the Go1 trunk mass (5.204 kg) includes the battery
is not stated in the model or its README, and Unitree's product page could not be opened
(HTTP 403). The trunk share below may therefore include or exclude the battery.

## Values taken from Go1-model (FR leg)

| `leg.xml` item | Value | Go1-model source |
|---|---|---|
| torso mass | 1.981 kg | trunk/4 + hip = 5.204/4 + 0.68 (`trunk` and `FR_hip` `<inertial mass>`) |
| thigh mass | 1.009 kg | `FR_thigh` `<inertial mass>` |
| thigh inertial | pos (-0.00304722, 0.019315, -0.0305004) m; quat (0.65243, -0.0272313, 0.0775126, 0.753383); diaginertia (0.00478717, 0.00460903, 0.000709268) kg m² | `FR_thigh` `<inertial>`, copied verbatim |
| thigh length (hip to knee) | 0.213 m | `FR_calf` body `pos="0 0 -0.213"` |
| calf (shank) mass | 0.195862 kg | `FR_calf` `<inertial mass>`; the foot geom sits in the calf body, whose `<inertial>` is its only mass (whether the URDF's foot link mass is included is not stated) |
| calf inertial | pos (0.00429862, 0.000976676, -0.146197) m; quat (0.691246, 0.00357467, 0.00511118, 0.722592); diaginertia (0.00149767, 0.00148468, 3.58427e-05) kg m² | `FR_calf` `<inertial>`, copied verbatim |
| calf length (knee to foot centre) | 0.213 m | `foot` default class, `pos="0 0 -0.213"` |
| foot radius | 0.023 m | `foot` default class, `size="0.023"` |
| knee range | −2.818 to −0.888 rad | `knee` default class `range="-2.818 -0.888"` (changed 2026-11-18, CHANGELOG #13) |
| foot friction | 0.8 / 0.02 / 0.01 | `foot` default class `friction="0.8 0.02 0.01"` (changed 2026-11-18, CHANGELOG #13) |
| hip and knee joint axis | (0, 1, 0) | `go1` default class `<joint axis="0 1 0">` (same sign convention: hip positive, knee negative) |

Total moving mass on the rail: **3.185862 kg** = trunk/4 + hip + thigh + calf
(`tests/test_model.py` checks every mass).

## Values not from Go1-model

| `leg.xml` item | Value | Status | Reason |
|---|---|---|---|
| torso position | z = 0.3488 m | DERIVED | 2 × 0.213 cos(0.7) + 0.023: foot just touches the floor at the stand keyframe (0.7, −1.4). Reward and termination use the rail displacement from this start. |
| torso geometry | box 0.16 × 0.12 × 0.08 m | ASSUMED | The rail blocks rotation, so the torso's inertia has no effect; the box only carries the mass. |
| joint armature | 0.0023 kg m² | VERIFIED | Actuator output inertia, Katz 2018 Table 2.2 p. 25 (changed from 0.0026, CHANGELOG #14). Note: §2.2.3 p. 33 gives rotor inertia J1 = 0.000072 kg m² and n = 6, i.e. n²·J1 = 0.002592 kg m² (≈ 0.0026) reflected to the output; the Table 2.2 value 0.0023 is used (user decision, 2026-11-18). |
| joint frictionloss | 0.09 N m | VERIFIED | Static gear and bearing friction at the actuator output, Katz 2018 §2.5.2 Eq. 2.27 pp. 51–52 (ω in Eq. 2.27 is the output speed). The torque-dependent term 0.04 N m/N m is not modelled. Rail joint: frictionloss 0. |
| joint damping | 0 | VERIFIED | Gear and bearing friction has "negligible speed dependence within the operating range of the actuator" (Katz 2018 §2.5.2 p. 51); changed from 0.05 N m s/rad (PLACEHOLDER), CHANGELOG #15. The Go1 model's own damping 2, armature 0.01, frictionloss 0.2 describe Unitree's motors and are not used. |
| hip range | −1.6 to 1.6 rad | ASSUMED (kept) | Unchanged from the scaffold. Go1-model hip: −0.686 to 4.501 rad. |
| collision capsules | thigh r = 0.02 m, calf r = 0.01 m | ASSUMED | Only the foot touches the floor in normal use; capsules carry no mass. |
| stand keyframe | rail 0, hip 0.7, knee −1.4 rad | ASSUMED (kept) | Unchanged. |
| timestep, integrator | 0.5 ms, implicitfast | ASSUMED (kept) | Unchanged. |
| angle unit | `<compiler angle="radian"/>` | FIX | All joint ranges in this file are in radians. Without this element MuJoCo reads MJCF angles in degrees; until CHANGELOG #16 the ranges were compiled as ±1.6° (hip) and −2.818° to −0.888° (knee), which pinned the leg nearly straight. `tests/test_model.py` checks the compiled ranges. |

## Static check at the stand keyframe

`scripts/check_model.py` and `tests/test_model.py` (`stand_static_torques` in
`busaware/leg_env.py`): with the foot carrying the full weight through a purely vertical
force (31.253 N), the holding torques are hip +0.2939 N m and knee +4.1012 N m, i.e.
**1.63 %** and **22.78 %** of the drive peak torque `i_max × kt_out` = 18 N m. The peak
torque uses the PROVISIONAL `kt_out` (UNVERIFIED, see params_provenance.md). The test
requires both to be below 50 %.
