import numpy as np

def _safe_lmtd(dT1, dT2, eps=1e-9):
    """
    Numerically robust LMTD:
    - Smooth when dT1 ≈ dT2 (series expansion)
    - Stable sign, avoids log singularities
    """
    m = 0.5 * (dT1 + dT2)
    r = 0.5 * (dT1 - dT2)
    # Near-equal case: use 2nd-order series
    if abs(r) <= 0.05 * (abs(m) + eps):
        return m * (1.0 - (r*r) / (3.0 * (m*m + eps)))
    # General case
    return (dT1 - dT2) / np.log((dT1 + eps) / (dT2 + eps))

def _soft_saturate(Q, Qmax, k=3.0):
    """Smoothly limit |Q| ≤ |Qmax| without sharp corners."""
    if Qmax == 0:
        return 0.0
    return Qmax * np.tanh(k * Q / (abs(Qmax) + 1e-12))

def fhx_predict_Q(
    Tf_in, Tw_in, mdot_f, cp_f, mdot_w, cp_w,
    AU_fhx, F=1.0, tol_rel=1e-4, max_iter=100, under_relax=0.5, Q_prev=None
):
    """
    Fixed-point LMTD-based HX solver with warm-start and smooth numerics.

    Returns dict: {"Q": heat_transfer, "Tf_out": hot_out, "Tw_out": cold_out}
    """
    # Heat capacity rates (guard against zeros)
    Cf = max(mdot_f * cp_f, 1e-12)
    Cw = max(mdot_w * cp_w, 1e-12)

    # Physical signed max heat transfer
    Cmin = min(Cf, Cw)
    Qmax = Cmin * (Tf_in - Tw_in)  # can be negative if Tw_in > Tf_in

    # Warm start: previous Q if available, else ~50% of |Qmax|
    Q = np.clip(Q_prev, -abs(Qmax), abs(Qmax)) if Q_prev is not None else 0.5 * Qmax

    for _ in range(max_iter):
        # Outlet temps from current Q
        Tf_out = Tf_in - Q / Cf
        Tw_out = Tw_in + Q / Cw

        # Temperature differences at ends
        dT1 = Tf_in - Tw_out
        dT2 = Tf_out - Tw_in

        # Robust LMTD
        LMTD = _safe_lmtd(dT1, dT2)

        # New Q from UA*F*LMTD, softly saturated to |Qmax|
        Q_new = _soft_saturate(AU_fhx * F * LMTD, Qmax)

        # Convergence
        if abs(Q_new - Q) <= tol_rel * max(abs(Q_new), 1e-9):
            Q = Q_new
            break

        # Under-relaxed update
        Q += under_relax * (Q_new - Q)

    # Final outlets
    Tf_out = Tf_in - Q / Cf
    Tw_out = Tw_in + Q / Cw

    return {"Q": Q, "Tf_out": Tf_out, "Tw_out": Tw_out}
