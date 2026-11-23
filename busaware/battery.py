"""Thevenin battery: OCV(SOC) - R0 - (R1 || C1)."""
import numpy as np
from .params import Params


class Battery:
    def __init__(self, p: Params, soc: float = 1.0):
        self.p = p
        self.soc = float(soc)
        self.v_rc = 0.0

    def ocv(self) -> float:
        return self.p.n_series * float(np.interp(self.soc, self.p.ocv_soc, self.p.ocv_v))

    def emf(self) -> float:
        """Voltage behind R0: pack OCV minus the RC-pair voltage."""
        return self.ocv() - self.v_rc

    def step(self, i_bat: float, dt: float) -> None:
        """i_bat > 0 discharges."""
        p = self.p
        self.soc = float(np.clip(self.soc - i_bat * dt / (3600.0 * p.q_ah), 0.0, 1.0))
        a = np.exp(-dt / (p.r1 * p.c1))          # exact update of the RC pair
        self.v_rc = a * self.v_rc + (1.0 - a) * p.r1 * i_bat
