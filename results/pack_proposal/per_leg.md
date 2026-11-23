# Per-leg battery parameters (6S2P LG HG2, quarter share): APPROVED 2026-11-18

Written by `scripts/derive_pack.py` from `results/battery_fit/fit_summary.json`.

| Field | Base, 25 °C | `Params.cold()`, 0 °C (stress test) | Unit | Derivation | Status / source |
|---|---|---|---|---|---|
| `n_series` | 6 | 6 | - | pack | Unitree Go1 GO1-BT03 (6S); 2P is our capacity match |
| `q_ah` | 1.3894 | 1.2978 | Ah | 2 × Q_cell(C/20) / 4 | cell FITTED |
| `r0` | 0.2887 | 0.5458 | Ω | 4 × (6/2 × R0_cell + R_wiring) | cell FITTED + wiring ASSUMED |
| ↳ cells / wiring | 0.2487 / 0.0400 | | Ω | | wiring 10 mΩ per pack, ASSUMED |
| `r1` | 0.1925 | 0.3218 | Ω | 4 × 6/2 × R1_cell | cell FITTED |
| `c1` | 129.871 | 79.483 | F | 2/6 × C1_cell / 4 | cell FITTED (τ = 24.995 / 25.580 s) |
| `ocv_soc`, `ocv_v` | `ocv_25degC.csv` | `ocv_0degC.csv` | V/cell | 23-point grid | cell FITTED |
| `v_min` | 16.8 | 16.8 | V | 6 × 2.8 V | lower bound of the validated battery model (HG2 tests stop at 2.8 V) |
| `i_bat_max` | 10.00 | 10.00 | A | 2 × 20 A / 4 | HG2-tech p. 1, §1 |
| `i_chg_max` | 2.00 | 0.75 | A | 2 × 4 A (25 °C) or 1.5 A (0 °C) / 4 | HG2-tech p. 1, §1; 0 °C value ASSUMED (dataset reduced regeneration below 10 °C) |
| `p_aux` | 5.00 | 5.00 | W | 20 W / 4 | ASSUMED |

## Sensitivity variants (Step 9, not run yet)

| Variant | Changes | Value |
|---|---|---|
| `vmin_datasheet` | `v_min` | 12.0000 |
| `wiring_0mohm` | `r0` | 0.2487 |
| `wiring_20mohm` | `r0` | 0.3287 |
| `paux_0w` | `p_aux` | 0.0000 |
| `paux_10w` | `p_aux` | 10.0000 |

## Cross-check against the Mini Cheetah battery (Katz 2018 Sec. 3.2.3 pp. 70-71)

Kobalt 6S2P (Samsung INR18650-25R), 960 W continuous: 240.0 W per leg. Ours: i_bat_max x 6 x 3.6 V = 216.0 W per leg (ratio 0.900).
