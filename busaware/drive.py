"""Averaged PMSM + inverter model (q-axis control, i_d = 0, current loop assumed fast).

tau_out = gear * kt * iq
v_q = R iq + w_e psi,  v_d = -w_e L iq,  v_d^2 + v_q^2 <= (V_bus / sqrt(3))^2
P_elec = 1.5 R iq^2 + kt iq w_m + P_idle     (kt = 1.5 p psi)
"""
import numpy as np
from .params import Params


def current_lag(iq_prev, iq_cmd, bw_hz, dt):
    """Plant-only first-order current-loop lag with bandwidth bw_hz (exact for a command held
    over the step): iq <- iq_prev + (1 - exp(-2 pi bw dt)) (iq_cmd - iq_prev). bw_hz <= 0
    means an ideal current loop (iq = iq_cmd)."""
    iq_cmd = np.asarray(iq_cmd, float)
    if bw_hz <= 0.0:
        return iq_cmd.copy()
    k = 1.0 - np.exp(-2.0 * np.pi * bw_hz * dt)
    return np.asarray(iq_prev, float) + k * (iq_cmd - np.asarray(iq_prev, float))


def output_torque(tau, w_joint, gear_eff):
    """Plant-only gear loss: scale the output torque by gear_eff where the drive does positive
    work on the load (tau * w > 0, motoring); generating torque (tau * w <= 0) is unchanged.
    Electrical power is not changed (the loss is in the gear, after the motor)."""
    tau = np.asarray(tau, float)
    if gear_eff >= 1.0:
        return tau.copy()
    return np.where(tau * np.asarray(w_joint, float) > 0.0, gear_eff * tau, tau)


class Drive:
    def __init__(self, p: Params):
        self.p = p

    def iq_from_torque(self, tau_out):
        return np.asarray(tau_out) / self.p.kt_out

    def torque(self, iq):
        return self.p.kt_out * np.asarray(iq)

    def bounds(self, w_joint, v_bus):
        """Per-joint interval [lo, hi] for iq from the current limit i_lim (i_max and the
        17 N m output-torque cap, Params.i_lim) and the voltage circle.

        Voltage circle: (R iq + w_e psi)^2 + (w_e L iq)^2 <= (V_bus/sqrt(3))^2, a
        quadratic in iq whose roots give the interval. If that interval does not
        overlap [-i_max, i_max] (or is empty: back-EMF alone exceeds the limit),
        no current satisfies both; the bound collapses to the point of
        [-i_max, i_max] closest to the circle's centre (best effort).
        """
        p = self.p
        lo, hi, vertex, _ = self._circle_box(w_joint, v_bus)
        empty = lo > hi
        best = np.clip(vertex, -p.i_lim, p.i_lim)
        return np.where(empty, best, lo), np.where(empty, best, hi)

    def circle_box_empty(self, w_joint, v_bus):
        """Per joint: True where no current satisfies both the voltage circle and the current
        limit, i.e. where bounds() returns a single best-effort point that the inverter cannot
        realise (paper assumption A6). Either the circle is empty (back-EMF alone exceeds the
        voltage limit) or its interval does not overlap [-i_lim, i_lim]. Same arithmetic as
        bounds(); used only by the run-time counter, never by control or plant."""
        lo, hi, _, disc = self._circle_box(w_joint, v_bus)
        return (disc < 0) | (lo > hi)

    def _circle_box(self, w_joint, v_bus):
        """Voltage-circle interval intersected with [-i_lim, i_lim], before the empty case is
        handled. Returns (lo, hi, vertex, disc); with disc < 0 the circle is empty and
        lo = hi = vertex clipped to the limit."""
        p = self.p
        we = p.pole_pairs * p.gear * np.asarray(w_joint, float)
        vlim2 = (max(v_bus, 0.0) / np.sqrt(3.0)) ** 2
        qa = p.r_phase ** 2 + (we * p.l_phase) ** 2
        qb = 2.0 * p.r_phase * we * p.psi
        qc = (we * p.psi) ** 2 - vlim2
        disc = qb * qb - 4.0 * qa * qc
        vertex = -qb / (2 * qa)
        root = np.sqrt(np.maximum(disc, 0.0)) / (2 * qa)
        lo = np.maximum(np.where(disc >= 0, vertex - root, vertex), -p.i_lim)
        hi = np.minimum(np.where(disc >= 0, vertex + root, vertex), p.i_lim)
        return lo, hi, vertex, disc

    def limit(self, iq_cmd, w_joint, v_bus):
        """Clip commanded iq to the current limit and the bus-voltage circle."""
        lo, hi = self.bounds(w_joint, v_bus)
        return np.clip(np.asarray(iq_cmd, float), lo, hi)

    def power_coeffs(self, w_joint):
        """P_i = a_i iq_i + b iq_i^2 (+ idle). Used by the safety filter."""
        a = self.p.kt * self.p.gear * np.asarray(w_joint, float)
        b = 1.5 * self.p.r_phase
        return a, b

    def scale_to_power(self, iq, w_joint, p_max):
        """Bus collapse model: if total power (incl. idle) exceeds p_max, scale all
        currents by one factor s in [0, 1) so that total power equals p_max.

        Total power at s: A s + B s^2 + P_idle_total, A = sum a_i iq_i, B = b sum iq_i^2.
        With c = P_idle_total - p_max < 0 the roots have opposite signs; the positive
        one is s = -2c / (A + sqrt(A^2 - 4 B c)) (cancellation-free form).
        Returns (iq_scaled, scaled).
        """
        iq = np.asarray(iq, float)
        if float(np.sum(self.power(iq, w_joint))) <= p_max:
            return iq, False
        a, b = self.power_coeffs(w_joint)
        A, B = float(np.sum(a * iq)), b * float(np.sum(iq ** 2))
        c = self.p.p_idle * iq.size - p_max
        if c >= 0.0:                                  # idle alone exceeds p_max
            return np.zeros_like(iq), True
        s = -2.0 * c / (A + np.sqrt(A * A - 4.0 * B * c))
        return float(np.clip(s, 0.0, 1.0)) * iq, True

    def power(self, iq, w_joint):
        a, b = self.power_coeffs(w_joint)
        return a * iq + b * np.asarray(iq) ** 2 + self.p.p_idle
