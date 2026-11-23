"""Step 1(c): derive one leg's battery parameters from a 6S2P pack of LG HG2 cells.

    python -I scripts/derive_pack.py

Reads the cell fit (results/battery_fit/fit_summary.json, from scripts/fit_battery.py) and
writes results/pack_proposal/per_leg.json and per_leg.md. busaware/params.py holds the same
numbers as literals; tests/test_params.py checks that the two agree.

Pack (approved 2026-11-18, CHANGELOG #9): 6S2P LG HG2.
  6S: Unitree Go1 battery GO1-BT03, https://www.unitree.com/en/go1battery (accessed
      2026-11-18; not opened directly by the tooling, page returned HTTP 403; values per the
      user and the indexed page): polymer Li-ion, charge limit 25.2 V, 6000 mAh, 133.2 Wh.
      133.2 Wh / 6 Ah = 22.2 V nominal = 6 x 3.7 V, which confirms 6S (and 25.2 V = 6 x 4.2 V).
  2P: our capacity match, 6000 mAh / 3000 mAh nominal HG2 cell. Not a Unitree statement.
  Second basis: the Mini Cheetah's battery is also 6S2P (Kobalt 24 V 5 Ah, Samsung INR18650-25R,
      108 Wh, 960 W continuous; Katz 2018 Sec. 3.2.3 pp. 70-71). Check: 960 W / 4 per leg against
      i_bat_max x nominal pack voltage (6 x 3.6 V, HG2-tech p. 1) per leg.
The leg is one leg of a pronking quadruped and gets a quarter share of the pack:
    capacity / 4,   R0, R1 x 4,   C1 / 4,   current limits / 4,   voltages unchanged,
with pack R0 = N_S/N_P * R0_cell + R_WIRING (ASSUMED), R1 = N_S/N_P * R1_cell,
C1 = N_P/N_S * C1_cell (tau unchanged).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIT = ROOT / "results" / "battery_fit" / "fit_summary.json"
OUT = ROOT / "results" / "pack_proposal"
N_S, N_P, LEGS = 6, 2, 4

# ---- cell limits: LG HG2, HG2-tech p. 1, section 1 "Battery Main Specifications"
#      (PDF in the HG2 dataset, DOI 10.17632/b5mj79w5w9.2)
CELL = dict(v_end_discharge=2.0,     # V, "Discharge: 2V End Voltage"
            i_dis_max_cont=20.0,     # A, "20A MAX Continuous Current"
            i_chg_fast=4.0,          # A, "4A, 4.2V, 100mA End-Current (CC-CV) Fast"
            i_chg_normal=1.5,        # A, "1.5A, 4.2, 50mA End-Current (CC-CV) Normal"
            v_chg=4.2,               # V, CC-CV charge voltage
            v_nominal=3.6)           # V, "Nominal Voltage 3.6 V"
# Lower bound of the validated battery model: the HG2 dataset's C/20 and HPPC tests stop at
# a 2.8 V terminal voltage (minimum logged 2.79993 V in 549_C20DisCh.csv and 549_HPPC.csv).
V_CELL_VALIDATED_MIN = 2.8

# ---- Mini Cheetah battery (Katz 2018 Sec. 3.2.3 pp. 70-71), for the cross-check only
KOBALT = dict(layout="6S2P", cells="Samsung INR18650-25R", q_ah=5.0, e_wh=108.0, p_cont_W=960.0)

# ---- ASSUMED allowances (no source)
R_WIRING_PACK = 0.010     # ohm, cell interconnects + main leads + connector + fuse
P_AUX_PACK = 20.0         # W, on-board computer, IMU, communication
# Step 9 sensitivity values (not run yet)
R_WIRING_PACK_SENS = (0.0, 0.020)    # ohm, pack level
P_AUX_LEG_SENS = (0.0, 10.0)         # W per leg


def per_leg(t, r_wiring_pack):
    k = N_S / N_P
    return dict(q_ah=N_P * t["q_c20_discharge_Ah"] / LEGS,
                r0=LEGS * (k * t["r0_ohm"] + r_wiring_pack),
                r1=LEGS * k * t["r1_ohm"],
                c1=(N_P / N_S) * t["c1_F"] / LEGS)


def main():
    fit = json.loads(FIT.read_text())
    w, c = fit["temps"]["25degC"], fit["temps"]["0degC"]
    base = dict(n_series=N_S, **per_leg(w, R_WIRING_PACK),
                r0_cells=LEGS * N_S / N_P * w["r0_ohm"], r0_wiring=LEGS * R_WIRING_PACK,
                ocv_soc=fit["ocv_grid_soc"], ocv_v=w["ocv_V"],
                v_min=N_S * V_CELL_VALIDATED_MIN,
                i_bat_max=N_P * CELL["i_dis_max_cont"] / LEGS,
                i_chg_max=N_P * CELL["i_chg_fast"] / LEGS,
                p_aux=P_AUX_PACK / LEGS)
    base["tau_s"] = base["r1"] * base["c1"]
    cold = dict(**per_leg(c, R_WIRING_PACK), ocv_v=c["ocv_V"],
                i_chg_max=N_P * CELL["i_chg_normal"] / LEGS)          # ASSUMED reduction, see md
    cold["tau_s"] = cold["r1"] * cold["c1"]
    variants = {
        "vmin_datasheet": dict(v_min=N_S * CELL["v_end_discharge"]),
        **{f"wiring_{1e3 * r:g}mohm": dict(r0=per_leg(w, r)["r0"]) for r in R_WIRING_PACK_SENS},
        **{f"paux_{p:g}w": dict(p_aux=p) for p in P_AUX_LEG_SENS},
    }
    p_leg_ours = base["i_bat_max"] * N_S * CELL["v_nominal"]
    check = dict(mini_cheetah_battery=KOBALT, mini_cheetah_p_cont_per_leg_W=KOBALT["p_cont_W"] / LEGS,
                 ours_i_bat_max_x_v_nominal_per_leg_W=p_leg_ours,
                 ratio_ours_to_mini_cheetah=p_leg_ours / (KOBALT["p_cont_W"] / LEGS))
    out = dict(pack=f"{N_S}S{N_P}P LG HG2", legs=LEGS, r_wiring_pack_ohm=R_WIRING_PACK,
               p_aux_pack_W=P_AUX_PACK, cell_limits=CELL, v_cell_validated_min=V_CELL_VALIDATED_MIN,
               base=base, cold=cold, variants=variants, check_mini_cheetah=check)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_leg.json").write_text(json.dumps(out, indent=2))

    f = lambda x, d=4: f"{x:.{d}f}"
    b = base
    md = [f"# Per-leg battery parameters ({out['pack']}, quarter share): APPROVED 2026-11-18", "",
          "Written by `scripts/derive_pack.py` from `results/battery_fit/fit_summary.json`.", "",
          "| Field | Base, 25 °C | `Params.cold()`, 0 °C (stress test) | Unit | Derivation | Status / source |",
          "|---|---|---|---|---|---|",
          f"| `n_series` | {N_S} | {N_S} | - | pack | Unitree Go1 GO1-BT03 (6S); 2P is our capacity match |",
          f"| `q_ah` | {f(b['q_ah'])} | {f(cold['q_ah'])} | Ah | {N_P} × Q_cell(C/20) / {LEGS} | cell FITTED |",
          f"| `r0` | {f(b['r0'])} | {f(cold['r0'])} | Ω | {LEGS} × ({N_S}/{N_P} × R0_cell + R_wiring) | cell FITTED + wiring ASSUMED |",
          f"| ↳ cells / wiring | {f(b['r0_cells'])} / {f(b['r0_wiring'])} | | Ω | | wiring {1e3 * R_WIRING_PACK:g} mΩ per pack, ASSUMED |",
          f"| `r1` | {f(b['r1'])} | {f(cold['r1'])} | Ω | {LEGS} × {N_S}/{N_P} × R1_cell | cell FITTED |",
          f"| `c1` | {f(b['c1'], 3)} | {f(cold['c1'], 3)} | F | {N_P}/{N_S} × C1_cell / {LEGS} | cell FITTED (τ = {f(b['tau_s'], 3)} / {f(cold['tau_s'], 3)} s) |",
          f"| `ocv_soc`, `ocv_v` | `ocv_25degC.csv` | `ocv_0degC.csv` | V/cell | 23-point grid | cell FITTED |",
          f"| `v_min` | {f(b['v_min'], 1)} | {f(b['v_min'], 1)} | V | {N_S} × {V_CELL_VALIDATED_MIN} V | lower bound of the validated battery model (HG2 tests stop at 2.8 V) |",
          f"| `i_bat_max` | {f(b['i_bat_max'], 2)} | {f(b['i_bat_max'], 2)} | A | {N_P} × {CELL['i_dis_max_cont']:g} A / {LEGS} | HG2-tech p. 1, §1 |",
          f"| `i_chg_max` | {f(b['i_chg_max'], 2)} | {f(cold['i_chg_max'], 2)} | A | {N_P} × {CELL['i_chg_fast']:g} A (25 °C) or {CELL['i_chg_normal']:g} A (0 °C) / {LEGS} | HG2-tech p. 1, §1; 0 °C value ASSUMED (dataset reduced regeneration below 10 °C) |",
          f"| `p_aux` | {f(b['p_aux'], 2)} | {f(b['p_aux'], 2)} | W | {P_AUX_PACK:g} W / {LEGS} | ASSUMED |",
          "", "## Sensitivity variants (Step 9, not run yet)", "",
          "| Variant | Changes | Value |", "|---|---|---|"]
    for name, ch in variants.items():
        for k, v in ch.items():
            md.append(f"| `{name}` | `{k}` | {f(v)} |")
    md += ["", "## Cross-check against the Mini Cheetah battery (Katz 2018 Sec. 3.2.3 pp. 70-71)", "",
           f"Kobalt {KOBALT['layout']} ({KOBALT['cells']}), {KOBALT['p_cont_W']:g} W continuous: "
           f"{check['mini_cheetah_p_cont_per_leg_W']:.1f} W per leg. Ours: i_bat_max x {N_S} x {CELL['v_nominal']} V = "
           f"{check['ours_i_bat_max_x_v_nominal_per_leg_W']:.1f} W per leg (ratio {check['ratio_ours_to_mini_cheetah']:.3f})."]
    md.append("")
    (OUT / "per_leg.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
