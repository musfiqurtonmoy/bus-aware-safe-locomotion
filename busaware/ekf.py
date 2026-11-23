"""Extended Kalman filter for one leg's battery share (Step 8), and its confidence score.

State x = [SOC, V_rc, R0]. Input: measured battery current I (A, > 0 discharges).
Measurement: bus voltage V (V). The model is the plant's Thevenin battery (busaware/battery.py,
busaware/bus.py) with the filter's Params (OCV table, Q, R1, C1), and R0 as a random walk:
    SOC'  = SOC - I dt / (3600 Q)
    V_rc' = a V_rc + (1 - a) R1 I,          a = exp(-dt / (R1 C1))   (exact, as the plant)
    R0'   = R0 + w,                         w ~ N(0, Q_R0 dt)
    V     = n_s ocv(SOC) - V_rc - R0 I + v, v ~ N(0, sigma_v^2 + R0^2 sigma_i^2)
The output equation is the battery's terminal relation I_bat = (E - V)/R0 of the bus model. It
does not hold while the battery's charge-current limit clamps I_bat (bus.py, i_chg_max); the
measurement update is skipped when the measured current is within 3 sigma_i of that limit.
Current noise enters the prediction through the input (G G^T sigma_i^2) and the measurement
through the R0 I term.

Timing within one plant substep k: the bus step computes V_k and I_k from the battery state before
its own update, then the battery steps with I_k. step(v_k, i_k) does the same: measurement update
on the prior state, then prediction with i_k. Afterwards emf() and r0 are the estimates of the
battery state the plant holds at the start of substep k+1, where the barrier filter uses them.

Confidence (for the confidence-scaled margin): the normalised innovation squared
NIS = nu^2 / S of each update, averaged over the last `window` updates. For a consistent filter
its expected value is 1. confidence = clip((nis_hi - mean NIS) / (nis_hi - nis_lo), 0, 1); 0
before the first update. Tuning (P0, Q, window, nis_lo, nis_hi) is the controller's choice, not a
physical parameter; the values are below (CHANGELOG #30).
"""
from collections import deque

import numpy as np

from .params import Params

# ---- tuning (controller side; CHANGELOG #30) ----
SD_SOC0 = 0.05            # initial SOC standard deviation (SOC initialised from the rest voltage)
SD_VRC0 = 0.005           # V, initial V_rc standard deviation (episodes start at rest: V_rc = 0)
SD_R0_REL0 = 1.0          # initial R0 standard deviation, relative to the model's R0 (a cold or
                          # aged pack can have about twice the R0: Params.cold() has 1.9 x)
Q_SOC = 1e-12             # 1/s, extra SOC process noise (beyond the current-sensor noise)
Q_VRC = 1e-8              # V^2/s, extra V_rc process noise
Q_R0 = 1e-4               # ohm^2/s, R0 random walk
R0_FLOOR_REL = 0.1        # R0 estimate kept >= 0.1 x model R0 (state constraint by clipping)
WINDOW = 200              # updates (0.1 s at 0.5 ms) in the NIS average
NIS_LO, NIS_HI = 1.5, 6.0  # mean NIS at or below NIS_LO -> confidence 1; at or above NIS_HI -> 0


class BatteryEKF:
    def __init__(self, p: Params, dt: float, soc0: float = 1.0, r0_0: float = None,
                 sigma_v: float = None, sigma_i: float = None, window: int = WINDOW,
                 nis_lo: float = NIS_LO, nis_hi: float = NIS_HI):
        self.p, self.dt = p, float(dt)
        self.sigma_v = p.sigma_v_meas if sigma_v is None else float(sigma_v)
        self.sigma_i = p.sigma_i_meas if sigma_i is None else float(sigma_i)
        self.a = float(np.exp(-dt / (p.r1 * p.c1)))
        self.k_soc = dt / (3600.0 * p.q_ah)
        self.r0_floor = R0_FLOOR_REL * p.r0
        r0_0 = p.r0 if r0_0 is None else float(r0_0)
        self.x = np.array([float(soc0), 0.0, r0_0])
        self.P = np.diag([SD_SOC0 ** 2, SD_VRC0 ** 2, (SD_R0_REL0 * p.r0) ** 2])
        g = np.array([-self.k_soc, (1.0 - self.a) * p.r1, 0.0])
        self.Q = (np.outer(g, g) * self.sigma_i ** 2
                  + np.diag([Q_SOC * dt, Q_VRC * dt, Q_R0 * dt]))
        self.F = np.diag([1.0, self.a, 1.0])
        self.nis = deque(maxlen=int(window))
        self.nis_lo, self.nis_hi = float(nis_lo), float(nis_hi)
        self.last_nis = float("nan")

    # ---- model ----
    def ocv(self, soc):
        p = self.p
        return p.n_series * float(np.interp(min(max(soc, 0.0), 1.0), p.ocv_soc, p.ocv_v))

    def ocv_slope(self, soc):
        """d(pack OCV)/dSOC of the piecewise-linear table at soc (end segments outside [0, 1])."""
        s, v = self.p.ocv_soc, self.p.ocv_v
        j = int(np.clip(np.searchsorted(s, soc, side="right") - 1, 0, len(s) - 2))
        return self.p.n_series * (v[j + 1] - v[j]) / (s[j + 1] - s[j])

    def soc_from_ocv(self, v_pack):
        """Inverse of the (increasing) OCV table, clipped to [0, 1]."""
        p = self.p
        return float(np.interp(v_pack / p.n_series, p.ocv_v, p.ocv_soc))

    def init_from_rest(self, v_meas):
        """SOC from a measured rest voltage (V_rc = 0 at rest), as a BMS does at power-up."""
        self.x[0] = self.soc_from_ocv(v_meas)
        self.x[1] = 0.0

    # ---- estimates handed to the barrier filter ----
    @property
    def soc(self):
        return float(self.x[0])

    @property
    def v_rc(self):
        return float(self.x[1])

    @property
    def r0(self):
        return float(self.x[2])

    def emf(self):
        """Estimated EMF behind R0: pack OCV(SOC) - V_rc."""
        return self.ocv(self.x[0]) - float(self.x[1])

    @property
    def confidence(self):
        if not self.nis:
            return 0.0
        m = sum(self.nis) / len(self.nis)
        return float(min(max((self.nis_hi - m) / (self.nis_hi - self.nis_lo), 0.0), 1.0))

    # ---- one substep ----
    def step(self, v_meas, i_meas):
        """Measurement update with (v_meas, i_meas) on the prior state, then prediction with
        i_meas. Returns the NIS of the update, or None if the update was skipped (current at the
        battery's charge limit)."""
        nis = None
        x, P = self.x, self.P
        if i_meas > -self.p.i_chg_max + 3.0 * self.sigma_i:
            h = np.array([self.ocv_slope(x[0]), -1.0, -i_meas])
            nu = v_meas - (self.ocv(x[0]) - x[1] - x[2] * i_meas)
            r = self.sigma_v ** 2 + (x[2] * self.sigma_i) ** 2
            ph = P @ h
            s = float(h @ ph) + r
            k = ph / s
            x = x + k * nu
            ikh = np.eye(3) - np.outer(k, h)
            P = ikh @ P @ ikh.T + np.outer(k, k) * r          # Joseph form
            nis = nu * nu / s
            self.nis.append(nis)
            self.last_nis = nis
        # prediction (state constraints by clipping)
        x = np.array([min(max(x[0] - self.k_soc * i_meas, 0.0), 1.0),
                      self.a * x[1] + (1.0 - self.a) * self.p.r1 * i_meas,
                      max(x[2], self.r0_floor)])
        self.x = x
        self.P = self.F @ P @ self.F.T + self.Q
        return nis
