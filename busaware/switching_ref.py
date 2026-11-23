"""Switching-level reference: ONE drive on the battery-fed bus (Step 2).

Drive values from params.py (gear, kt_out, pole pairs, L, current and torque limits confirmed
in the Katz thesis; c_bus derived from it; r_phase ASSUMED; p_idle PLACEHOLDER; see params_provenance.md).

Time-domain model, integrated with a fixed step DT (1 us), numpy/math only:
- Battery, charge limit, averaged brake chopper and drive logic power P_idle: the same
  models and parameters as the plant (busaware/battery.py, bus.py), so that only the drive
  differs from the averaged model. Bus: C dV/dt = I_bat(V) - i_dc - P_idle/V - g(V) V.
  The battery state is advanced once per PWM period with the period-mean current.
- Inverter: three-phase two-level, centre-aligned space-vector PWM (min-max zero-sequence
  injection) at F_PWM, linear range |v_dq| <= V/sqrt(3) (no overmodulation). Dead time
  DEAD_TIME: during it both switches are off and the pole follows the current direction
  (i > 0: lower diode, pole at 0; i < 0: upper diode, pole at V), so the effective high
  interval of a pole is shortened or lengthened by DEAD_TIME at one edge. Within each
  integration step the pole voltage is its exact time-average over that step. Diode
  forward drop: 0 (ASSUMED). Switch on-resistance is lumped into the phase resistance
  r_phase, exactly as in the averaged model (params.py: "incl. inverter on-resistance").
- PMSM in the dq frame (amplitude-invariant Park, kt = 1.5 p psi), L_d = L_q = l_phase,
  integrated with the trapezoidal rule; DC current, copper loss and mechanical power use the
  step-midpoint current, so the energy balance holds exactly:
      L di_d/dt = v_d - R i_d + w_e L i_q
      L di_q/dt = v_q - R i_q - w_e L i_d - w_e psi
  The mechanical speed is imposed (dynamometer-style scenarios).
- Current loop: PI in dq, i_d* = 0, sampled at the start of every PWM period and applied in
  the same period. Gains by pole-zero cancellation for bandwidth I_BW: kp = L w_c,
  ki = R w_c. Voltage limited to V/sqrt(3) with conditional-integration anti-windup.
DC-side current i_dc = sum_x f_x i_x (f_x: fraction of the step the pole is high), so
V i_dc = 1.5 (v_d i_d + v_q i_q) exactly.
"""
import math

import numpy as np

from .battery import Battery
from .bus import chopper_conductance
from .params import Params

DT = 1e-6            # s, integration step (PROMPTS Step 2: 1 us or finer)
F_PWM = 40e3         # Hz, switching and loop rate: Katz 2018 Sec. 2.3.1 p. 38, Sec. 2.3.2 p. 40 (Katz-fw hw_config.h L47)
DEAD_TIME = 100e-9   # s, ASSUMED from Katz-fw hw_config.h comment "deadtime compensation (100 ns / 25 us)"
I_BW = 1000.0        # Hz, current-loop bandwidth: Katz 2018 Sec. 2.4.1 p. 44 (Katz-fw main.c L192 default)
SQ3 = math.sqrt(3.0)


def _overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def _high_interval(duty, ts, current):
    """Effective [start, end] of the pole-high interval within one centre-aligned period."""
    if duty >= 1.0:
        return 0.0, ts
    if duty <= 0.0:
        return 0.0, 0.0
    t_on, t_off = 0.5 * ts * (1.0 - duty), 0.5 * ts * (1.0 + duty)
    if current > 0.0:                      # lower diode conducts in the dead time
        return t_on + DEAD_TIME, t_off
    return t_on, min(t_off + DEAD_TIME, ts)  # upper diode conducts in the dead time


def simulate(p: Params, w_joint, iq_cmd, t_end, soc, dt=DT, f_pwm=F_PWM, i_bw=I_BW, steady_start=True):
    """Run the switching model with imposed joint speed w_joint (rad/s) and q-axis current
    command iq_cmd(t) (A, motor side). steady_start: start in the steady state for zero
    current at the imposed speed, i.e. the PI integrator holds the back-EMF voltage
    (v_q = w_e psi, v_d = 0); otherwise it starts at zero. Returns per-PWM-period means:
    t (period end), v_bus, i_bat, p_elec (= V i_dc + P_idle, power drawn from the bus),
    i_d, i_q, iq_ref, v_d, v_q, p_cu (1.5 R |i|^2), p_mech (1.5 w_e psi i_q); and the
    final dq currents as attributes in "final"."""
    ts = 1.0 / f_pwm
    n_sub = int(round(ts / dt))
    if abs(n_sub * dt - ts) > 1e-12:
        raise ValueError("PWM period must be a whole number of integration steps")
    n_per = int(round(t_end / ts))
    r, ell, psi = p.r_phase, p.l_phase, p.psi
    we = p.pole_pairs * p.gear * w_joint
    wc = 2.0 * math.pi * i_bw
    kp, ki = ell * wc, r * wc
    # trapezoidal-rule matrices for d/dt i = A i + b, A = [[-R/L, w_e], [-w_e, -R/L]]
    ha, hw = 0.5 * dt * r / ell, 0.5 * dt * we
    ka, kw, kb = 1.0 - ha, hw, 1.0 + ha                 # (I + dt/2 A) = [[ka, kw], [-kw, ka]]
    det = kb * kb + hw * hw                              # (I - dt/2 A) = [[kb, -kw], [kw, kb]]
    bat = Battery(p, soc)
    v = bat.ocv()
    i_d = i_q = 0.0
    x_d, x_q = 0.0, (we * psi if steady_start else 0.0)
    theta = 0.0
    keys = ("t", "v_bus", "i_bat", "p_elec", "i_d", "i_q", "iq_ref", "v_d", "v_q", "p_cu", "p_mech")
    out = {k: np.zeros(n_per) for k in keys}
    for n in range(n_per):
        t0 = n * ts
        e = bat.emf()
        # ---- current controller, sampled at the period start
        iq_ref = max(-p.i_lim, min(p.i_lim, iq_cmd(t0)))
        ed, eq = -i_d, iq_ref - i_q
        vd_c, vq_c = kp * ed + x_d, kp * eq + x_q
        vlim = max(v, 0.0) / SQ3
        mag = math.hypot(vd_c, vq_c)
        if mag > vlim:
            vd_c, vq_c = vd_c * vlim / mag, vq_c * vlim / mag
        else:
            x_d += ki * ts * ed
            x_q += ki * ts * eq
        th = theta + 0.5 * we * ts                       # rotate the command to mid-period
        c, s = math.cos(th), math.sin(th)
        va, vb_ = vd_c * c - vq_c * s, vd_c * s + vq_c * c
        vabc = (va, -0.5 * va + 0.5 * SQ3 * vb_, -0.5 * va - 0.5 * SQ3 * vb_)
        v0 = -0.5 * (max(vabc) + min(vabc))
        duty = [min(1.0, max(0.0, 0.5 + (vx + v0) / max(v, 1e-9))) for vx in vabc]
        # ---- integrate the period
        sv = sib = spe = sid = siq = svd = svq = scu = smech = 0.0
        for k in range(n_sub):
            tl = k * dt
            c, s = math.cos(theta), math.sin(theta)
            ia = i_d * c - i_q * s
            ib = i_d * s + i_q * c
            ix = (ia, -0.5 * ia + 0.5 * SQ3 * ib, -0.5 * ia - 0.5 * SQ3 * ib)
            f = []
            for x in range(3):
                a0, a1 = _high_interval(duty[x], ts, ix[x])
                f.append(_overlap(a0, a1, tl, tl + dt) / dt)
            vp = (f[0] * v, f[1] * v, f[2] * v)
            vn = (vp[0] + vp[1] + vp[2]) / 3.0
            v_a, v_b, v_c = vp[0] - vn, vp[1] - vn, vp[2] - vn
            v_al = (2.0 / 3.0) * (v_a - 0.5 * v_b - 0.5 * v_c)
            v_be = (v_b - v_c) / SQ3
            thm = theta + 0.5 * we * dt
            c, s = math.cos(thm), math.sin(thm)
            v_d = v_al * c + v_be * s
            v_q = -v_al * s + v_be * c
            # dq currents: trapezoidal (implicit midpoint) rule, closed-form 2x2 solve
            rhs_d = ka * i_d + kw * i_q + dt * v_d / ell
            rhs_q = -kw * i_d + ka * i_q + dt * (v_q - we * psi) / ell
            n_d = (kb * rhs_d + kw * rhs_q) / det
            n_q = (-kw * rhs_d + kb * rhs_q) / det
            md, mq = 0.5 * (i_d + n_d), 0.5 * (i_q + n_q)          # step-midpoint currents
            ma = md * c - mq * s
            mb = md * s + mq * c
            i_dc = f[0] * ma + f[1] * (-0.5 * ma + 0.5 * SQ3 * mb) + f[2] * (-0.5 * ma - 0.5 * SQ3 * mb)
            i_b = max((e - v) / p.r0, -p.i_chg_max)
            g = chopper_conductance(v, p.v_chop_off, p.v_chop, p.r_chop)
            # V i_dc = 1.5 (v_d md + v_q mq) = copper + mechanical + magnetic, exactly
            sv += v; sib += i_b; spe += v * i_dc + p.p_idle; sid += md; siq += mq
            svd += v_d; svq += v_q
            scu += 1.5 * r * (md * md + mq * mq); smech += 1.5 * we * psi * mq
            i_d, i_q = n_d, n_q
            v += dt * (i_b - i_dc - p.p_idle / v - g * v) / p.c_bus
            theta += we * dt
        theta = math.fmod(theta, 2.0 * math.pi)
        bat.step(sib / n_sub, ts)
        for key, val in zip(keys, (t0 + ts, sv / n_sub, sib / n_sub, spe / n_sub, sid / n_sub,
                                   siq / n_sub, iq_ref, svd / n_sub, svq / n_sub,
                                   scu / n_sub, smech / n_sub)):
            out[key][n] = val
    out["final"] = dict(i_d=i_d, i_q=i_q, v_bus=v)
    return out
