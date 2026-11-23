# Parameter provenance

One row per field of `busaware.params.Params` (CLAUDE.md rule 3). The **value**
column is what `params.py` holds. Battery values were fixed on 2026-11-18, before any
training run (CHANGELOG #9). Every number is copied from a file written by a script
(`results/battery_fit/fit_summary.json`, `results/pack_proposal/per_leg.json`) or quoted
from the cited document; `tests/test_params.py` checks `params.py` against `per_leg.json`.

Status key:
- **SOURCED**: value taken from the cited source.
- **FITTED**: value fitted by a script in this repository from cited data.
- **DERIVED**: computed by a script from SOURCED/FITTED values with a stated rule.
- **VERIFIED**: confirmed in the cited source (section and printed page given).
- **PLACEHOLDER**: engineering value without a source, still to be replaced.
- **ASSUMED**: no source; reason given.
- **UNVERIFIED**: a value with a claimed source that could not be opened.

Last updated 2026-11-18, after Step 1(a), (b) and (c) (CHANGELOG #14 for 1(a)).

## Sources

| Key | Source |
|---|---|
| HG2-data | Kollmeyer, Naguib, Skells, *LG 18650HG2 Li-ion Battery Data*, Mendeley Data v2, DOI [10.17632/b5mj79w5w9.2](https://data.mendeley.com/datasets/b5mj79w5w9/2), CC BY 4.0. Files used and SHA-256: `data/raw/lg_hg2/MANIFEST.csv`, repeated in `fit_summary.json`. |
| HG2-tech | *Technical Information and Experimental Test Results for LG 18650HG2*, PDF in the same dataset (January 2020). Page numbers below are the printed page numbers. |
| HG2-fit | `scripts/fit_battery.py` -> `results/battery_fit/` (`fit_summary.json`, `ocv_25degC.csv`, `ocv_0degC.csv`, `pulse_fits.csv`, `ocv_check.csv`, `fig_*.png/pdf`). |
| Pack | `scripts/derive_pack.py` -> `results/pack_proposal/per_leg.json`, `per_leg.md`. |
| Go1-bat | Unitree, Go1 battery, https://www.unitree.com/en/go1battery (accessed 2026-11-18): model GO1-BT03, polymer Li-ion, charge limit 25.2 V, 6000 mAh, 133.2 Wh. **Not opened by the tooling** (HTTP 403 on every variant); values per the user and the search-engine index of that page. **Verified by the author on 2026-11-20**: the author opened the page and confirmed model GO1-BT03, polymer lithium-ion, charge limit voltage DC 25.2 V, rated capacity 6000 mAh and 133.2 Wh. Cited in the paper as `unitree2026go1battery` (paper/refs.bib). 133.2 Wh / 6 Ah = 22.2 V nominal = 6 × 3.7 V, which confirms 6S (and 25.2 V = 6 × 4.2 V). The page states neither the cell type nor the parallel count. |
| Katz-thesis | B. G. Katz, *A low cost modular actuator for dynamic robots*, S.M. thesis, MIT, June 2018 (submitted May 11, 2018), [dspace.mit.edu/handle/1721.1/118671](https://dspace.mit.edu/handle/1721.1/118671). Placed by hand at `data/raw/katz2018/katz2018_thesis.pdf` (SHA-256 `be6bf4780645b138c1a32fda45fb207ce1546ded4b8f7e6d3d7f326ea47ad8d1`). Pages below are printed page numbers (equal to PDF page numbers). |
| Katz-fw | `bgkatz/motorcontrol` (MIT licence), commit `93bbc60676ba4c2c0c64466c2b921eeb9d9fb1df`, files and lines as cited per row; cloned read-only to `data/raw/bgkatz_motorcontrol/repo`. Gear ratio and torque constant are runtime flash settings there (defaults 1.0), so the repo does **not** source `gear` or `kt_out`. The older mbed `motor_config.h` (rev. 51647c6c500d) that held 6.0 and 0.45 is a dead link. |

## Drive

| Field | Value | Unit | Status | Source / reason | Notes |
|---|---|---|---|---|---|
| `gear` | 6.0 | - | VERIFIED | Katz-thesis Ch. 2 intro p. 23 ("a 6:1 single-stage planetary gear reduction") and §2.2.2 p. 27 ("a 6:1 planetary gear reduction"). | Katz-fw holds no value (`GR` flash setting, default 1.0, `Core/Src/main.c` L203). |
| `kt_out` | 0.45 | N·m/A | DERIVED (from VERIFIED) | 0.45 = gear × 0.075 N·m/A. The thesis states no single kt: the motor torque constant falls from about 0.082 to about 0.0685 N·m/A over 5–40 A, dropping "by 12% at maximum" (§2.2.1 p. 27, Fig. 2-3 p. 27; values read from the plot); 0.075 lies inside that range. | Linear kt; the 12 % saturation is not modelled. Peak torque is capped by `tau_max_out`. |
| `pole_pairs` | 21 | - | VERIFIED | Katz-thesis §2.2.1 p. 27 ("a large number of pole-pairs (21)"); also §2.4.2 p. 45. | Katz-fw default `PPAIRS = 21` (`Core/Src/main.c` L202) agrees. |
| `r_phase` | 0.105 | Ω | ASSUMED | Not given in Katz-thesis (Ch. 2 read; Appendix C pp. 97–104 uses only a generic example, R = 0.5 Ω, L = 1 mH). | Phase resistance incl. inverter on-resistance. Step 9 variants ×0.5 / ×1.5. |
| `l_phase` | 30e-6 | H | VERIFIED | Katz-thesis §2.3.1 p. 38 ("very low inductance, around 30 µH"). | Katz-fw `hw_config.h` L53-54 uses 40 µH for its own default motor. |
| `i_max` | 40.0 | A | VERIFIED | Katz-thesis §2.3 p. 37 ("designed for 24V nominal input voltage and 40 amp peak phase current"). | Katz-fw default `I_MAX = 40` (`Core/Src/main.c` L193) agrees. |
| `tau_max_out` | 17.0 | N·m | VERIFIED | Katz-thesis Table 2.2 p. 25 ("Maximum Torque 17 N m"); also §2.2.4 p. 36. | Enforced as a current clip `i_lim = min(i_max, tau_max_out/kt_out)` = 37.78 A in `Drive.bounds` (plant, filter and switching reference). Torque alone is not clipped, because the electrical power model is linear in kt. |
| `p_idle` | 2.0 | W | ASSUMED | No source: Katz-thesis gives no idle consumption of the drive electronics (gate driver, logic, current sensing), and no other source was found. 2 W per drive is a round assumption. It enters only as a constant load (2 × `p_idle` per leg) beside `p_aux`; the Step 9 variants `paux_0w` and `paux_10w` change that constant load by 5 W either way, more than the whole 4 W of idle power, without a filter-mode violation. | Gate driver + logic per drive. Status was PLACEHOLDER until 2026-11-20 (CHANGELOG #36); value unchanged. |
| `gear_eff` | 1.0 (off) | - | ASSUMED | Plant-only gear efficiency applied to motoring output torque (τ·ω > 0); generating torque unchanged; electrical power unchanged. Variant `gear_eff_0.9` uses 0.9, the bottom of the "transmission efficiency of 90-95%" (Katz-thesis §2.5.2 p. 52). CHANGELOG #15. |
| `i_loop_bw` | 0.0 (off) | Hz | ASSUMED | Plant-only first-order current-loop lag; 0 = ideal current loop, as the averaged model assumes. Variant `current_lag` uses 1000 Hz: current bandwidth "was set to 1 kHz" for most experiments (Katz-thesis §2.4.1 p. 44; the hardware reaches over 4.5 kHz, p. 41; Katz-fw `Core/Src/main.c` L192 default). The filter does not see it (CHANGELOG #13). |
| `kt` (derived) | `kt_out/gear` | N·m/A | derived | - | Property, no own value. |
| `psi` (derived) | `kt/(1.5·pole_pairs)` | Wb | derived | - | Property, no own value. |

## Battery: one leg's quarter share of a 6S2P LG HG2 pack

Pack: **6S2P** LG HG2. Second basis for 6S2P: the Mini Cheetah's Kobalt 24 V Max 5 Ah pack is 6S2P of
Samsung INR18650-25R cells (2.5 Ah, 3.6 V nominal, 20 A continuous), 108 Wh, 960 W continuous
(Katz-thesis §3.2.3 pp. 70–71). Check (`per_leg.json` `check_mini_cheetah`): 960 W / 4 = 240 W per
leg against `i_bat_max` × 6 × 3.6 V = 216 W per leg (ratio 0.900). 6S from Go1-bat; **2P is our capacity match** (6000 mAh / 3000 mAh
nominal cell, HG2-tech p. 1, §1), not a Unitree statement. The Go1 pack's own cells are
polymer Li-ion; only the series/parallel layout is taken from it. The leg is one leg of
a pronking quadruped and gets a quarter share: capacity / 4; R0, R1 × 4; C1 / 4; current
limits / 4; voltages unchanged. Base values are at 25 °C (cell test 549); `Params.cold()`
uses 0 °C (cell test 585) and is a **stress test only**.

| Field | Value (base) | `cold()` | Unit | Status | Derivation and source |
|---|---|---|---|---|---|
| `n_series` | 6 | 6 | - | SOURCED | Go1-bat (6S). |
| `q_ah` | 1.389382094 | 1.297832363 | Ah | DERIVED from FITTED | 2 × Q_cell / 4; Q_cell = C/20 discharge to 2.8 V, **2.778764 Ah** (25 °C), **2.595665 Ah** (0 °C), HG2-fit `q_c20_discharge_Ah`. Nominal 3.0 Ah (HG2-tech p. 1, §1). The two temperatures are different cell specimens. |
| `r0` | 0.288681778 | 0.545835528 | Ω | DERIVED from FITTED + ASSUMED | 4 × (6/2 × R0_cell + R_wiring). R0_cell **20.723 mΩ** (25 °C, 36 pulses 3–18 A) and **42.153 mΩ** (0 °C, 16 pulses 12 and 18 A only, CHANGELOG #8), HG2-fit `r0_ohm`. Cells alone: 0.248681778 / 0.505835528 Ω. R_wiring = 10 mΩ per pack (0.040 Ω per leg), **ASSUMED**: of order 1 mΩ per cell-to-cell link plus a few mΩ for main leads, connector and fuse. Cross-check: HG2-tech p. 4, §8 gives 27.2 mΩ 1C discharge resistance at 25 °C, 50 % SOC; the fitted 10 s apparent resistance is **26.012 mΩ** (HG2-fit `r_10s_ohm`). |
| `r1` | 0.192460965 | 0.321832531 | Ω | DERIVED from FITTED | 4 × 6/2 × R1_cell; R1_cell **16.038 mΩ** / **26.819 mΩ** (HG2-fit `r1_ohm`). |
| `c1` | 129.870715 | 79.4827132 | F | DERIVED from FITTED | 2/6 × C1_cell / 4; C1_cell **1558 F** (τ = 24.995 s) / **954 F** (τ = 25.580 s) (HG2-fit `c1_F`, `tau_s`). τ is unchanged by the scaling. |
| `ocv_soc` | 23-point grid, 0 to 1 | same | - | FITTED | HG2-fit `ocv_grid_soc`. |
| `ocv_v` | `OCV_V_25C` | `OCV_V_0C` | V per cell | FITTED | `results/battery_fit/ocv_25degC.csv`, `ocv_0degC.csv`: mean of C/20 discharge and charge branches. Checked against the relaxed voltage after 1 h HPPC rests: RMSE **15.28 mV** (25 °C, 12 points), **19.51 mV** (0 °C, 10 points); rests sit below the table on average (−10.7 / −11.6 mV) and by up to 39 / 55 mV below SOC 0.2 (discharge-side hysteresis). `Battery.ocv()` multiplies by `n_series`. |

**Fit quality** (HG2-fit): RMSE over all fitted pulse windows **8.40 mV** at 25 °C and
**29.34 mV** at 0 °C (12 and 18 A pulses only). Running the fitted model open-loop through
the whole HPPC test (SOC ≥ 0.15) gives **14.21 mV** at 25 °C and **26.57 mV** at 0 °C
(45.84 mV under load). At 0 °C the cell's resistance falls with current (per-pulse median
R0 44.9 → 39.5 mΩ, R1 35.3 → 24.4 mΩ from 3 A to 18 A, `pulse_fits.csv`); a linear 1-RC
model cannot follow this. At 25 °C, R0 does not depend on current (20.5 to 20.7 mΩ).

**0 °C refit on ≥ 12 A pulses** (HG2-fit `by_pulse_current_soc_ge_lo`, with the
all-currents fit kept as `fit_all_currents`), pooled RMSE per pulse current, SOC ≥ 0.15:
3 A **21.32 mV** (all-currents fit: 20.70 mV), 18 A **30.14 mV** (32.74 mV). Median
error at the end of the 10 s pulse, model − measured: 3 A **+16.96 mV** (+14.58 mV),
18 A **−62.81 mV** (−77.10 mV); negative means the model predicts more sag than
measured. The remaining error comes mostly from the rise of R0 above SOC 0.85 and from
multi-time-constant recovery, which a single linear RC pair cannot represent.

## DC bus, protection and loads

| Field | Value (base) | `cold()` | Unit | Status | Derivation and source |
|---|---|---|---|---|---|
| `c_bus` | 8.680556e-4 (0.868 mF) | same | F | VERIFIED (derived) | Katz-thesis §3.2.3 p. 72: the input capacitance of the 12 motor controllers stores "around 1.5 J at 24 V". C_total = 2E/V² = 2 × 1.5 J / (24 V)² = 5.2083 mF; per controller C_total/12 = 0.43403 mF; this leg has two drives: 2 × 0.43403 = 0.86806 mF. Computed in `params.py` as `2 * (2 * 1.5 / 24.0 ** 2) / 12`. Changed from 4 mF (PLACEHOLDER), CHANGELOG #15. |
| `v_min` | 16.8 | same | V | SOURCED (data range) | 6 × 2.8 V: lower bound of the validated battery model. The HG2-data C/20 and HPPC tests stop at a 2.8 V terminal voltage (minimum logged 2.79993 V in `549_C20DisCh.csv` and `549_HPPC.csv`). The cell datasheet allows 2.0 V (HG2-tech p. 1, §1): see variant `vmin_datasheet`. |
| `v_obs_ref` | 16.8 | same | V | FIXED (training) | Not a physical parameter: the reference of the policy's bus-voltage observation (V − `v_obs_ref`)/6. Equal to the `v_min` with which every policy was trained, and kept there in every variant (including `vmin_datasheet`), so a variant changes the limit the plant and filter enforce but not the policy's input scaling (CHANGELOG #28). |
| `v_max` | 27.5 | same | V | ASSUMED | Overvoltage trip, a drive/BMS choice. Full charge is 6 × 4.2 V = 25.2 V (HG2-tech p. 1, §1). Katz-fw gives 40 V soft / 43 V hard limits for its own controller (`hw_config.h` L49-50). |
| `i_bat_max` | 10.0 | same | A | DERIVED from SOURCED | 2 × 20 A / 4; 20 A = cell max continuous discharge (HG2-tech p. 1, §1). |
| `i_chg_max` | 2.0 | 0.75 | A | DERIVED from SOURCED; `cold()` ASSUMED | Base: 2 × 4 A / 4, 4 A = cell fast-charge current (HG2-tech p. 1, §1). `cold()`: 2 × 1.5 A / 4, 1.5 A = cell standard charge current (same source); **ASSUMED** choice because the HG2 dataset reduced the regeneration current limit below 10 °C (dataset readme). |
| `v_chop` | 26.0 | same | V | ASSUMED | Averaged brake chopper: full duty at and above this voltage. Design choice between the 25.2 V full-charge voltage and `v_max`. |
| `v_chop_off` | 25.5 | same | V | ASSUMED | Averaged brake chopper: duty 0 below, rising linearly to 1 at `v_chop` (CHANGELOG #10; was a hysteresis turn-off, #5). Must stay above the 25.2 V full-charge voltage so the chopper cannot discharge a resting pack. |
| `r_chop` | 2.0 | same | Ω | ASSUMED | Brake resistor: design choice, no source. |
| `p_aux` | 5.0 | same | W per leg | ASSUMED | 20 W per robot / 4: on-board computer, IMU and communication. For comparison: the Mini Cheetah's UP board draws "roughly 5 W peak" (Katz-thesis §3.2.1 p. 69), and its logic supply is a 5 V, 30 W converter (§3.2.3 p. 72); 20 W is kept (user decision, 2026-11-18). Constant load on the bus in the plant; subtracted in the filter's budget like the drives' idle power (CHANGELOG #9). |
| `sigma_v_meas` | 0.02 | same | V (1σ) | ASSUMED | Bus-voltage measurement noise seen by the battery EKF (Step 8), white, per 0.5 ms sample. No source: about 3 LSB of a 12-bit ADC behind a divider with a ~30 V full scale (30 V / 4096 = 7.3 mV). Used only in EKF evaluations (`HoppingLegEnv(estimator="ekf")`); the plant and the true-state filter do not use it (CHANGELOG #30). |
| `sigma_i_meas` | 0.05 | same | A (1σ) | ASSUMED | Battery-current measurement noise seen by the battery EKF (Step 8), white, per 0.5 ms sample. No source: 0.5 % of `i_bat_max`, about 5 LSB of a 12-bit reading over ±20 A. Used only in EKF evaluations, as above (CHANGELOG #30). |

## Sensitivity variants (`Params.variant(name)`, for Step 9; not run before then)

| Variant | Field | Value | Reason |
|---|---|---|---|
| `vmin_datasheet` | `v_min` | 12.0 V | 6 × 2.0 V, HG2 discharge end voltage (HG2-tech p. 1, §1). Below 16.8 V the battery model is extrapolated. |
| `wiring_0mohm` | `r0` | 0.248681778 Ω | No wiring allowance. |
| `wiring_20mohm` | `r0` | 0.328681778 Ω | 20 mΩ per pack. |
| `paux_0w` | `p_aux` | 0.0 W | No auxiliary load. |
| `paux_10w` | `p_aux` | 10.0 W per leg | 40 W per robot. |
| `rphase_x0.5` | `r_phase` | 0.0525 Ω | Phase resistance is not in the thesis (user, 2026-11-18); ×0.5 of the ASSUMED value. |
| `rphase_x1.5` | `r_phase` | 0.1575 Ω | ×1.5 of the ASSUMED value. |
| `gear_eff_0.9` | `gear_eff` | 0.9 | Motoring output torque × 0.9 (Katz-thesis §2.5.2 p. 52: 90–95 % transmission efficiency); generating torque unchanged. |
| `current_lag` | `i_loop_bw` | 1000 Hz | Plant-only current-loop lag at the bandwidth used in the thesis experiments (§2.4.1 p. 44); the filter assumes an ideal loop. |

## Notes from Katz-thesis (recorded, not applied)
1. **Torque-dependent friction**: Eq. 2.27 (p. 52) has a 0.04 N·m/N·m term; only the 0.09 N·m static part is modelled (`frictionloss`). The Step 9 variant `gear_eff_0.9` bounds the effect of transmission losses on motoring torque.
2. **Reflected rotor inertia**: §2.2.3 p. 33 gives J1 = 0.000072 kg·m² and n = 6, so n²·J1 = 0.002592 kg·m² (≈ 0.0026), against the Table 2.2 "Output Inertia" 0.0023 kg·m² used for `armature` (user decision). See model_provenance.md.
3. **Computer load**: the UP board draws "roughly 5 W peak" (§3.2.1 p. 69); `p_aux` keeps 20 W per robot (5 W per leg), user decision.

## Switching-level reference (Step 2; `busaware/switching_ref.py`, not `Params` fields)

Outputs in `results/validation/` (no longer PROVISIONAL since CHANGELOG #14). `r_phase` is
ASSUMED and `p_idle` was PLACEHOLDER when these outputs were written; every output header says so. `p_idle` is now labelled ASSUMED (same value, CHANGELOG #36), so the outputs stay valid; their headers keep the earlier label.

| Setting | Value | Status | Source / reason |
|---|---|---|---|
| integration step | 1 µs | SOURCED (spec) | PROMPTS.md Step 2: "1 microsecond or finer". |
| PWM frequency | 40 kHz | VERIFIED | Katz-thesis §2.3.1 p. 38 ("switching frequency and control loop frequency of 40 kHz") and §2.3.2 p. 40; Katz-fw `hw_config.h` L47 (25 µs). Was 20 kHz until CHANGELOG #13. |
| dead time | 100 ns | ASSUMED | Katz-fw `hw_config.h` L46 comment "deadtime compensation (100 ns / 25 us)". |
| current-loop bandwidth | 1000 Hz | VERIFIED | Katz-thesis §2.4.1 p. 44 (bandwidth set to 1 kHz in most experiments). PI zero at the RL pole (thesis Eq. 2.25–2.26, pp. 43–44); here in continuous-time form, kp = L·2π·BW, ki = R·2π·BW. |
| diode forward drop | 0 V | ASSUMED | Not modelled. |
| switch on-resistance | inside `r_phase` | as averaged model | `params.py`: `r_phase` includes inverter on-resistance. |
| scenario speeds, currents, SOC | see `scripts/validate_drive.py` docstring | ASSUMED (rule stated before results) | S1 30 A at standstill; S2 20 A at the speed using 95 % of V/√3; S3 −20 A at the S2 speed; SOC 0.5. |
