"""Shared DC bus: battery + DC-link capacitor + averaged brake chopper, constant-power load.

C dV/dt = I_bat(V) - P_load / V - g(V) V
I_bat(V) = max((E - V)/R0, -i_chg_max)                    (battery charge limit)
g(V)     = clip((V - v_chop_off)/(v_chop - v_chop_off), 0, 1) / r_chop
                                                          (averaged chopper duty / resistor)
Integrated with backward Euler, all terms at the end-of-step voltage V'. Multiplying the
step equation by V' gives
    H(V') = C/dt V'(V' - V) - V' I_bat(V') + P + g(V') V'^2 = 0.
-V' I_bat(V') is -V'(E - V')/R0 below x_c = E + R0 i_chg_max and V' i_chg_max above it
(continuous at x_c), and g(V') V'^2 is non-decreasing, so H is strictly increasing above
x_lo = beta/(2 alpha), the chopper-free turning point (for physical states, V <= 2E; see
step()). The stable root is the unique root above x_lo; it is found by Newton's method
safeguarded by bisection. If H(x_lo) > 0 there is no root: the load exceeds what the bus
can deliver (collapse), and V' = x_lo.
"""
import numpy as np
from .params import Params
from .battery import Battery


def chopper_conductance(v, v_off, v_on, r_chop):
    """Averaged brake-chopper conductance g(V) (S): duty rises linearly from 0 at v_off to 1 at v_on."""
    return min(max((v - v_off) / (v_on - v_off), 0.0), 1.0) / r_chop


class Bus:
    def __init__(self, p: Params, battery: Battery):
        self.p, self.bat = p, battery
        self.v = battery.ocv()
        self.i_bat = 0.0

    def g(self, v: float) -> float:
        p = self.p
        return chopper_conductance(v, p.v_chop_off, p.v_chop, p.r_chop)

    def max_power(self, dt: float) -> float:
        """Largest load power the bus can deliver next step: beta^2 / (4 alpha), chopper-free.
        Valid because at the turning point beta/(2 alpha), about half the bus voltage, the
        chopper is off and the battery is discharging (charge limit inactive)."""
        p = self.p
        alpha = p.c_bus / dt + 1.0 / p.r0
        beta = p.c_bus * self.v / dt + self.bat.emf() / p.r0
        return beta * beta / (4.0 * alpha)

    def turning_point(self, dt: float) -> float:
        """x* = beta/(2 alpha) for the next step from the present state, the chopper-free turning
        point of the step equation (paper Lemma 1). Same arithmetic as step(); used only by the
        run-time A4 counter (end-of-step root below x*), never by control or plant."""
        p = self.p
        e, v, cd = self.bat.emf(), self.v, p.c_bus / dt
        alpha, beta = cd + 1.0 / p.r0, cd * v + e / p.r0
        return beta / (2.0 * alpha)

    def _h(self, x, v, e, p_load, cd):
        """H(x), dH/dx, I_bat(x), g(x) for the step equation (see module docstring)."""
        p = self.p
        ib_u = (e - x) / p.r0
        if ib_u >= -p.i_chg_max:
            t_i, d_i, ib = (x * x - e * x) / p.r0, (2.0 * x - e) / p.r0, ib_u
        else:
            t_i, d_i, ib = p.i_chg_max * x, p.i_chg_max, -p.i_chg_max
        band = p.v_chop - p.v_chop_off
        u = (x - p.v_chop_off) / band
        if u <= 0.0:
            g, dg = 0.0, 0.0
        elif u >= 1.0:
            g, dg = 1.0 / p.r_chop, 0.0
        else:
            g, dg = u / p.r_chop, 1.0 / (band * p.r_chop)
        h = cd * x * (x - v) + t_i + p_load + g * x * x
        dh = cd * (2.0 * x - v) + d_i + dg * x * x + 2.0 * g * x
        return h, dh, ib, g

    def step(self, p_load: float, dt: float) -> dict:
        p, bat = self.p, self.bat
        e, v, cd = bat.emf(), self.v, p.c_bus / dt
        alpha, beta = cd + 1.0 / p.r0, cd * v + e / p.r0
        # H is the unclamped piece for x <= x_c = E + R0 i_chg_max and the clamped piece above.
        # The unclamped piece increases above beta/(2 alpha); the clamped piece increases above
        # t_c, and t_c <= x_c for any physical state (V <= 2E). Guard the other case anyway.
        x_lo = max(beta / (2.0 * alpha), 1e-9)
        t_c = (cd * v - p.i_chg_max) / (2.0 * cd)
        if t_c > e + p.r0 * p.i_chg_max:
            x_lo = max(x_lo, t_c)
        h_lo, _, _, _ = self._h(x_lo, v, e, p_load, cd)
        tol = 1e-9 * beta * beta / (4.0 * alpha)          # p_load == max_power to rounding
        collapse = h_lo > tol
        if h_lo >= -tol:
            x = x_lo
        else:
            lo, hi = x_lo, max(v, e, p.v_chop) + 1.0
            while self._h(hi, v, e, p_load, cd)[0] <= 0.0:
                hi = lo + 2.0 * (hi - lo)
            x = min(max(v, lo), hi)
            for _ in range(100):
                h, dh, _, _ = self._h(x, v, e, p_load, cd)
                if h > 0.0:
                    hi = x
                else:
                    lo = x
                x_new = x - h / dh if dh > 0.0 else 0.5 * (lo + hi)
                if not (lo < x_new < hi):
                    x_new = 0.5 * (lo + hi)                # safeguard: bisect
                if abs(x_new - x) <= 1e-13 * max(1.0, x):
                    x = x_new
                    break
                x = x_new
        _, _, ib, g = self._h(x, v, e, p_load, cd)
        self.v, self.i_bat = float(x), float(ib)
        bat.step(ib, dt)
        return dict(
            v=self.v, i_bat=self.i_bat, p_chop=g * x * x,
            uv=bool(self.v < p.v_min), ov=bool(self.v > p.v_max),
            oc=bool(self.i_bat > p.i_bat_max), collapse=bool(collapse))
