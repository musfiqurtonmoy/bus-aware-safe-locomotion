"""All physical parameters in one place.

Every field has a row in params_provenance.md (CLAUDE.md rule 3).
- Battery values (one leg's quarter share of a 6S2P LG HG2 pack) were fixed on 2026-11-18,
  before any training run (CHANGELOG #9). They are copied from
  results/pack_proposal/per_leg.json (scripts/derive_pack.py, from the cell fit in
  results/battery_fit/fit_summary.json); tests/test_params.py checks that they agree.
- Drive values were confirmed in the Katz thesis (Step 1(a), 2026-11-18, CHANGELOG #14);
  "Katz 2018 p. N" refers to its printed page N (data/raw/katz2018/katz2018_thesis.pdf).
- Values still marked PLACEHOLDER or ASSUMED are explained in params_provenance.md.
"""
from dataclasses import dataclass, field, replace
import numpy as np

# Cell OCV tables (V per cell) on a 23-point SOC grid, from scripts/fit_battery.py
# (mean of the C/20 discharge and charge branches; results/battery_fit/ocv_<T>.csv).
OCV_SOC = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
OCV_V_25C = (2.87928, 3.12512, 3.23294, 3.29848, 3.35863, 3.43851, 3.5022, 3.54451, 3.59568,
             3.6311, 3.66134, 3.69654, 3.74009, 3.78914, 3.84438, 3.89309, 3.93315, 3.97837,
             4.03, 4.07237, 4.09121, 4.11511, 4.18976)
OCV_V_0C = (2.90567, 3.17951, 3.2929, 3.36822, 3.42142, 3.48693, 3.53151, 3.58532, 3.62081,
            3.64807, 3.68074, 3.72193, 3.76735, 3.81388, 3.86364, 3.90598, 3.94672, 3.99393,
            4.04296, 4.07691, 4.0927, 4.11631, 4.19007)

# One leg's series resistance = cells + wiring allowance (ASSUMED 10 mOhm per pack, x 4).
R0_CELLS_25C = 0.248681778          # ohm, 4 x 6/2 x R0_cell(25 degC)
R0_CELLS_0C = 0.505835528           # ohm, 4 x 6/2 x R0_cell(0 degC, >= 12 A pulses)
R_WIRING_LEG = 0.040                # ohm, 4 x 0.010 (pack level). ASSUMED


@dataclass
class Params:
    # ---- drive (Mini-Cheetah-class quasi-direct-drive actuator) ----
    gear: float = 6.0            # 6:1 planetary. VERIFIED: Katz 2018 Ch. 2 p. 23, Sec. 2.2.2 p. 27
    kt_out: float = 0.45         # N m/A at the output. DERIVED: gear x 0.075 N m/A; motor kt spans
                                 # about 0.082 -> 0.0685 N m/A over 5-40 A (Katz 2018 Fig. 2-3, p. 27)
    pole_pairs: int = 21         # VERIFIED: Katz 2018 Sec. 2.2.1 p. 27
    r_phase: float = 0.105       # ohm, incl. inverter on-resistance. ASSUMED (not given in Katz 2018)
    l_phase: float = 30e-6       # H, "around 30 uH". VERIFIED: Katz 2018 Sec. 2.3.1 p. 38
    i_max: float = 40.0          # A, peak phase current. VERIFIED: Katz 2018 Sec. 2.3 p. 37
    tau_max_out: float = 17.0    # N m, maximum output torque. VERIFIED: Katz 2018 Table 2.2 p. 25.
                                 # Enforced as a current clip at tau_max_out / kt_out (see i_lim).
    p_idle: float = 2.0          # W per drive, gate driver + logic. ASSUMED (no source; params_provenance.md)
    gear_eff: float = 1.0        # plant-only gear efficiency on MOTORING output torque (tau*w > 0);
                                 # 1 = lossless. Variant gear_eff_0.9: bottom of 90-95 % (Katz 2018 p. 52).
    i_loop_bw: float = 0.0       # Hz, plant-only first-order current-loop lag; 0 = ideal (off). Variant
                                 # current_lag: 1000 Hz (Katz 2018 Sec. 2.4.1 p. 44). The filter does not see it.

    # ---- battery: one leg's quarter share of a 6S2P LG HG2 pack, 25 degC ----
    n_series: int = 6                            # Unitree Go1 GO1-BT03 is 6S; 2P is our match
    q_ah: float = 1.389382094                    # Ah, 2 x Q_cell(C/20) / 4. FITTED
    r0: float = R0_CELLS_25C + R_WIRING_LEG      # ohm, 0.288681778. FITTED + ASSUMED wiring
    r1: float = 0.192460965                      # ohm, 4 x 6/2 x R1_cell. FITTED
    c1: float = 129.870715                       # F, 2/6 x C1_cell / 4 (tau 24.995 s). FITTED
    ocv_soc: np.ndarray = field(default_factory=lambda: np.array(OCV_SOC))
    ocv_v: np.ndarray = field(default_factory=lambda: np.array(OCV_V_25C))   # V per cell

    # ---- DC bus and loads ----
    # F, DC-link capacitance of the leg's two drives. VERIFIED (derived): the 12 Mini Cheetah motor
    # controllers store about 1.5 J at 24 V (Katz 2018 Sec. 3.2.3 p. 72): C = 2E/V^2 / 12 per drive.
    c_bus: float = 2 * (2 * 1.5 / 24.0 ** 2) / 12               # = 0.868 mF
    v_min: float = 16.8          # V, 6 x 2.8 V: lower bound of the validated battery model
    # V, reference of the policy's bus-voltage observation (V - v_obs_ref)/6. Not a physical limit:
    # fixed at the v_min every policy was trained with, so variants that move v_min (vmin_datasheet)
    # do not shift the observation (CHANGELOG #28).
    v_obs_ref: float = 16.8
    v_max: float = 27.5          # V, overvoltage trip. PLACEHOLDER
    i_bat_max: float = 10.0      # A, 2 x 20 A / 4 (HG2 max continuous discharge)
    i_chg_max: float = 2.0       # A, 2 x 4 A / 4 (HG2 fast-charge current)
    v_chop: float = 26.0         # V, averaged brake chopper at full duty from here up. ASSUMED
    v_chop_off: float = 25.5     # V, chopper duty 0 below, rising linearly to 1 at v_chop. ASSUMED
    r_chop: float = 2.0          # ohm, brake resistor. ASSUMED
    p_aux: float = 5.0           # W per leg, computer + sensors (20 W per robot / 4). ASSUMED

    # ---- sensors seen by the battery EKF (Step 8; busaware/ekf.py), 1 sigma, white ----
    sigma_v_meas: float = 0.02   # V, bus-voltage measurement noise. ASSUMED
    sigma_i_meas: float = 0.05   # A, battery-current measurement noise. ASSUMED

    @property
    def i_lim(self) -> float:
        """Effective q-axis current limit (A): min(i_max, tau_max_out / kt_out). The drive model
        is linear in kt (the measured 12 % kt drop at 40 A is not modelled), so the 17 N m torque
        cap is applied as a current clip; plant, filter bounds and switching reference share it."""
        return min(self.i_max, self.tau_max_out / self.kt_out)

    @property
    def tau_peak(self) -> float:
        """Peak output torque of the drive model (N m)."""
        return self.kt_out * self.i_lim

    @property
    def kt(self) -> float:
        """Motor-side torque constant, Nm/A."""
        return self.kt_out / self.gear

    @property
    def psi(self) -> float:
        """Flux linkage from kt = 1.5 * p * psi."""
        return self.kt / (1.5 * self.pole_pairs)

    @classmethod
    def cold(cls) -> "Params":
        """0 degC battery (cell test 585, 1-RC fit on >= 12 A pulses). Stress test only.
        i_chg_max: 2 x 1.5 A (HG2 standard charge) / 4. ASSUMED, because the HG2 dataset
        reduced regeneration current below 10 degC."""
        return replace(cls(), q_ah=1.297832363, r0=R0_CELLS_0C + R_WIRING_LEG,
                       r1=0.321832531, c1=79.4827132, ocv_v=np.array(OCV_V_0C), i_chg_max=0.75)

    VARIANTS = ("vmin_datasheet", "wiring_0mohm", "wiring_20mohm", "paux_0w", "paux_10w",
                "rphase_x0.5", "rphase_x1.5", "current_lag", "gear_eff_0.9")

    @classmethod
    def variant(cls, name: str) -> "Params":
        """Step 9 sensitivity variants of the 25 degC base (not run before Step 9)."""
        changes = {
            "vmin_datasheet": dict(v_min=12.0),                 # 6 x 2.0 V, HG2 discharge end voltage
            "wiring_0mohm": dict(r0=R0_CELLS_25C),              # no wiring allowance
            "wiring_20mohm": dict(r0=R0_CELLS_25C + 4 * 0.020),  # 20 mOhm per pack
            "paux_0w": dict(p_aux=0.0),
            "paux_10w": dict(p_aux=10.0),
            "rphase_x0.5": dict(r_phase=0.5 * cls.r_phase),     # phase resistance is ASSUMED
            "rphase_x1.5": dict(r_phase=1.5 * cls.r_phase),
            "current_lag": dict(i_loop_bw=1000.0),             # plant-only current-loop lag
            "gear_eff_0.9": dict(gear_eff=0.9),                 # motoring torque x 0.9, generating unchanged
        }
        if name not in changes:
            raise KeyError(f"unknown variant {name!r}; choose from {cls.VARIANTS}")
        return replace(cls(), **changes[name])
