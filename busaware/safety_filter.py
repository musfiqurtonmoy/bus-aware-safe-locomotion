"""Discrete-time barrier filter for bus undervoltage and battery overcurrent.

Barriers:  h1 = V - V_min,   h2 = I_max - I_bat.
One backward-Euler bus step gives P = V'[(E - V')/R0 - C (V' - V)/dt] - g(V') V'^2
(g: averaged chopper conductance), which is decreasing in V' on the stable branch. So h(k+1) >= (1 - gamma) h(k) for both
barriers is equivalent to a single upper bound on total load power (the budget).
Joint power is quadratic in iq, so the minimal-change projection onto the budget
and the per-joint drive bounds (current limit, voltage circle) is a convex QCQP
with one quadratic constraint plus a box, solved by bisection on its multiplier.
"""
import numpy as np
from .bus import chopper_conductance


EPS_REL = 1e-9   # rounding margin on the budget (CHANGELOG #23)
MAX_DOUBLINGS = 200   # upper end of the multiplier bracket is doubled at most this often (named, CHANGELOG #37)


def power_budget(v, e_hat, r0_hat, c_bus, dt, v_min, i_max, i_bat,
                 gamma=0.3, margin_v=0.0, margin_i=0.0, chopper=None, eps_rel=EPS_REL):
    """Largest total load power that keeps both barriers satisfied next step.

    e_hat, r0_hat: estimates of battery EMF (OCV - V_rc) and series resistance.
    margin_v, margin_i: extra margins, e.g. scaled by estimator confidence.
    chopper: (v_chop_off, v_chop, r_chop) of the averaged brake chopper, or None.

    Backward-Euler bus step, all terms at the end-of-step voltage V':
        P(V') = V' (beta - alpha V') - g(V') V'^2,
        alpha = C/dt + 1/R0,  beta = C V/dt + E/R0  (chopper-free),
    with g the averaged chopper conductance. The first term is decreasing above the
    chopper-free turning point beta/(2 alpha), and g is non-decreasing, so P(V') is
    decreasing there too. The plant's operating root lies there, so for any
    v_req >= beta/(2 alpha):  V' >= v_req  <=>  P <= P(v_req), i.e. g(v_req) enters alpha.
    Below the turning point every deliverable P already gives V' >= v_req.

    eps_rel: the returned budget is P_bar - eps_rel * max(|P_bar|, beta^2/(4 alpha)), so that the
    plant's separately recomputed load (drive powers + idle + p_aux) does not exceed P_bar by a
    rounding error (assumption A7 of the proof). The scale includes beta^2/(4 alpha), the bus's
    largest deliverable power (the same scale as the plant's collapse tolerance), so the margin
    does not vanish when P_bar is near zero. eps_rel = 0 gives the exact P_bar.
    """
    vm = v_min + margin_v
    im = i_max - margin_i
    v_req_uv = vm + (1.0 - gamma) * (v - vm)                 # h1' >= (1-gamma) h1
    i_allow = im - (1.0 - gamma) * (im - i_bat)              # h2' >= (1-gamma) h2
    v_req_oc = e_hat - r0_hat * i_allow                      # I_bat' <= i_allow
    alpha = c_bus / dt + 1.0 / r0_hat
    beta = c_bus * v / dt + e_hat / r0_hat
    v_req = max(v_req_uv, v_req_oc, beta / (2.0 * alpha))
    g = chopper_conductance(v_req, *chopper) if chopper is not None else 0.0
    p_bar = v_req * (beta - (alpha + g) * v_req)
    return p_bar - eps_rel * max(abs(p_bar), beta * beta / (4.0 * alpha))


def project_currents(iq_des, a, b, budget, lo, hi, iters=40):
    """min ||iq - iq_des||^2  s.t.  sum(a_i iq_i + b iq_i^2) <= budget,  lo <= iq <= hi.

    The Lagrangian separates per joint; for multiplier lam each joint minimises a
    1-D convex quadratic over [lo_i, hi_i], so
        iq_i(lam) = clip((d_i - lam a_i / 2) / (1 + lam b), lo_i, hi_i).
    Power along iq(lam) is non-increasing in lam (derivative of the concave dual),
    so lam is found by bisection; the returned point is always on the feasible side.

    Returns (iq_safe, status):
      "inactive"   power constraint not binding; iq_safe = clip(iq_des, lo, hi)
      "active"     power constraint binding; iq_safe is the box-constrained projection
      "infeasible" minimum power over the box exceeds the budget; iq_safe is the
                   box minimiser of power (best effort, budget NOT met)
    """
    d = np.asarray(iq_des, float)
    a = np.asarray(a, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    f = lambda iq: float(np.sum(a * iq + b * iq ** 2))
    sol = lambda lam: np.clip((d - 0.5 * lam * a) / (1.0 + lam * b), lo, hi)
    x0 = sol(0.0)
    if f(x0) <= budget:
        return x0, "inactive"
    x_min = np.clip(-a / (2.0 * b), lo, hi)                   # = lim_{lam->inf} iq(lam)
    if f(x_min) > budget:
        return x_min, "infeasible"
    lam_lo, lam_hi = 0.0, 1.0
    for _ in range(MAX_DOUBLINGS):
        if f(sol(lam_hi)) <= budget:
            break
        lam_lo, lam_hi = lam_hi, 2.0 * lam_hi
    else:                                                     # budget == box minimum (to rounding)
        return x_min, "active"
    for _ in range(iters):
        mid = 0.5 * (lam_lo + lam_hi)
        lam_lo, lam_hi = (mid, lam_hi) if f(sol(mid)) > budget else (lam_lo, mid)
    return sol(lam_hi), "active"


def project_per_joint(iq_des, a, b, budget, lo, hi):
    """Joint-level comparison limiter (CHANGELOG #38; evaluation only, never trained with).

    The same power budget as project_currents, split equally between the n drives: joint i may
    draw a_i iq_i + b iq_i^2 <= budget / n. Each joint is limited on its own, without knowing the
    other joints' currents: iq_i = clip(iq_des_i, L_i, U_i), where [L_i, U_i] is the intersection
    of [lo_i, hi_i] with the interval on which joint i's power is within its share (the roots of
    b x^2 + a_i x - budget/n). This is the nearest admissible current for that joint alone.

    Returns (iq, status) with the statuses of project_currents:
      "inactive"   no joint is limited by its share; iq = clip(iq_des, lo, hi)
      "active"     some joint is limited by its share; every joint is within its share, so the
                   total is within the budget
      "infeasible" for some joint no current in the box is within its share; that joint gets the
                   box minimiser of its power (best effort), the others are limited as above
    """
    d = np.asarray(iq_des, float)
    a = np.asarray(a, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    share = budget / d.size
    box = np.clip(d, lo, hi)
    iq, status = box.copy(), "inactive"
    for i in range(d.size):
        disc = a[i] * a[i] + 4.0 * b * share
        L = U = None
        if disc >= 0.0:
            r = np.sqrt(disc)
            L, U = max(lo[i], (-a[i] - r) / (2.0 * b)), min(hi[i], (-a[i] + r) / (2.0 * b))
        if L is None or L > U:
            iq[i] = np.clip(-a[i] / (2.0 * b), lo[i], hi[i])
            status = "infeasible"
            continue
        iq[i] = min(max(d[i], L), U)
        if iq[i] != box[i] and status == "inactive":
            status = "active"
    return iq, status
