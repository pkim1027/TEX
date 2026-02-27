import math
import numpy as np
from TEX_new_volume import HullDims, r_hull

# ------------------------------------------------------------
# Precompute normalized circular-segment area + centroid tables
# (for a circle of radius r, with horizontal waterline draft d from bottom)
# Returns:
#   A = r^2 * Ahat(u),  zbar_from_bottom = r * zhat(u),  where u = d/r in [0,2]
# ------------------------------------------------------------
def _build_seg_tables(nu=4001, nz=2000):
    u = np.linspace(0.0, 2.0, nu)   # u = d/r
    Ahat = np.zeros_like(u)
    zhat = np.zeros_like(u)

    # normalized integration in t = z/r, circle center at t=1, bottom t=0
    t = np.linspace(0.0, 2.0, nz)
    # half-width at height t (normalized): y/r
    half = np.sqrt(np.maximum(0.0, 1.0 - (t - 1.0) ** 2))
    dA_dt = 2.0 * half  # (area element) / r^2

    # cumulative integrals for fast lookup
    # Ahat(u) = ∫_0^u dA_dt dt
    # zhat(u) = (1/Ahat) ∫_0^u t * dA_dt dt
    cumA = np.cumsum(dA_dt) * (t[1] - t[0])
    cumQ = np.cumsum(t * dA_dt) * (t[1] - t[0])

    # map u to index in t-grid
    # clamp u into [0,2]
    u_cl = np.clip(u, 0.0, 2.0)
    idx = np.minimum((u_cl / 2.0 * (nz - 1)).astype(int), nz - 2)
    frac = (u_cl - t[idx]) / np.maximum(1e-12, (t[idx + 1] - t[idx]))

    A_u = cumA[idx] + frac * (cumA[idx + 1] - cumA[idx])
    Q_u = cumQ[idx] + frac * (cumQ[idx + 1] - cumQ[idx])

    Ahat[:] = A_u
    zhat[:] = 0.0
    mask = A_u > 1e-14
    zhat[mask] = Q_u[mask] / A_u[mask]

    return u, Ahat, zhat

_U_TAB, _AHAT_TAB, _ZHAT_TAB = _build_seg_tables()

def _seg_A_zbar_from_bottom(d, r):
    """Submerged area A and centroid zbar (from bottom) for circle radius r, draft d from bottom."""
    if r <= 0.0:
        return 0.0, 0.0
    if d <= 0.0:
        return 0.0, 0.0
    if d >= 2.0 * r:
        return math.pi * r * r, r  # centroid of full circle from bottom is at r

    u = d / r  # in [0,2]
    Ahat = np.interp(u, _U_TAB, _AHAT_TAB)
    zhat = np.interp(u, _U_TAB, _ZHAT_TAB)
    A = (r * r) * Ahat
    zbar = r * zhat
    return float(A), float(zbar)

# ------------------------------------------------------------
# Heeled hydrostatics (outer geometry only) -> B(θ), KN(θ), GZ(θ)
# ------------------------------------------------------------
def _heel_volume_and_B(p, theta, h, x):
    """
    For a given heel theta [rad] and free-surface intercept h (in rotated coords),
    compute displaced volume V and buoyancy centroid B in BODY coords.
    """
    c = math.cos(theta)
    s = math.sin(theta)

    # integrate along x
    A_list = np.zeros_like(x)
    My_p   = np.zeros_like(x)  # first moment about y' axis: A * y'_cent
    Mz_p   = np.zeros_like(x)  # first moment about z' axis: A * z'_cent

    for i, xi in enumerate(x):
        r = r_hull(float(xi), p)  # OUTER radius at this station
        if r <= 1e-12:
            continue

        # In rotated (y',z') coords:
        # circle center was (y=0,z=r) in body coords
        # rotate by theta about x:
        yC_p = r * s
        zC_p = r * c

        # bottom point in rotated coords
        z_bot_p = zC_p - r  # = r*(c-1)

        # draft from bottom to free surface (horizontal in rotated coords)
        d = h - z_bot_p

        A, zbar_from_bottom = _seg_A_zbar_from_bottom(d, r)
        if A <= 0.0:
            continue

        # centroid in rotated coords
        ybar_p = yC_p
        zbar_p = z_bot_p + zbar_from_bottom

        A_list[i] = A
        My_p[i] = A * ybar_p
        Mz_p[i] = A * zbar_p

    # trapezoid in x
    V  = np.trapezoid(A_list, x)
    My = np.trapezoid(My_p, x)
    Mz = np.trapezoid(Mz_p, x)

    if V <= 1e-12:
        return 0.0, (0.0, 0.0)

    yB_p = My / V
    zB_p = Mz / V

    # rotate back to BODY coords:
    # [y] = [ c -s ] [y']
    # [z]   [ s  c ] [z']
    yB = yB_p * c - zB_p * s
    zB = yB_p * s + zB_p * c

    return float(V), (float(yB), float(zB))

def _solve_h_for_volume(p, theta, V_target, x, h_lo=None, h_hi=None, it=80):
    """Bisection on h so that V(theta,h) = V_target."""
    # safe brackets: h spans from "way above" to "way below"
    # In rotated coords, z' ranges roughly ~[-r, +r] along hull; use global R.
    Rmax = p.R
    if h_lo is None: h_lo = -2.5 * Rmax
    if h_hi is None: h_hi = +2.5 * Rmax

    V_lo, _ = _heel_volume_and_B(p, theta, h_lo, x)
    V_hi, _ = _heel_volume_and_B(p, theta, h_hi, x)

    # ensure bracket (monotone increasing V with h)
    # if not, expand
    k = 0
    while (V_lo - V_target) > 0.0 and k < 20:
        h_lo -= 2.0 * Rmax
        V_lo, _ = _heel_volume_and_B(p, theta, h_lo, x)
        k += 1
    k = 0
    while (V_hi - V_target) < 0.0 and k < 20:
        h_hi += 2.0 * Rmax
        V_hi, _ = _heel_volume_and_B(p, theta, h_hi, x)
        k += 1

    # bisection
    a, b = h_lo, h_hi
    for _ in range(it):
        m = 0.5 * (a + b)
        V_m, _ = _heel_volume_and_B(p, theta, m, x)
        if V_m >= V_target:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

def compute_KN_KGsin_GZ_curve(
    p,
    V_disp_in3,
    KG_in,
    thetas_deg=np.linspace(0.0, 60.0, 61),
    Nx=2500
):
    """
    Returns arrays: theta_deg, KN, KGsin, GZ, plus B_y, B_z for debugging.
    Units: inches (consistent with your geometry).
    """
    x = np.linspace(0.0, p.x_back_end, Nx)

    theta_deg = np.array(thetas_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)

    KN   = np.zeros_like(theta_rad)
    KGsn = np.zeros_like(theta_rad)
    GZ   = np.zeros_like(theta_rad)
    By   = np.zeros_like(theta_rad)
    Bz   = np.zeros_like(theta_rad)

    for i, th in enumerate(theta_rad):
        # solve for h to hold displaced volume constant
        h = _solve_h_for_volume(p, th, V_disp_in3, x)

        V, (yB, zB) = _heel_volume_and_B(p, th, h, x)
        # (V should be ~V_disp_in3)
        By[i], Bz[i] = yB, zB

        c = math.cos(th)
        s = math.sin(th)

        # KN and KG*sin(theta)
        KN[i]   = yB * c + zB * s
        KGsn[i] = KG_in * s

        # GZ = KN - KG*sin(theta)
        GZ[i]   = KN[i] - KGsn[i]

    return theta_deg, KN, KGsn, GZ, By, Bz

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # your hull params (outer)
    p = HullDims(R=12.0, Lf=29.5, Lc=182.0, Lb=31.5, r_tip=3.5)

    # You must provide displacement volume (outer displaced volume) and KG.
    # Example: if you already computed displaced volume in ft^3, convert:
    # V_disp_in3 = V_disp_ft3 * 1728
    V_disp_in3 = 18.28 * 1728  # <-- example (replace with YOUR displaced volume)
    KG_in = 1.0 * 12.0         # <-- example: KG=1 ft -> 12 in (replace with YOUR KG)

    th_deg, KN, KGsn, GZ, By, Bz = compute_KN_KGsin_GZ_curve(
        p,
        V_disp_in3=V_disp_in3,
        KG_in=KG_in,
        thetas_deg=np.linspace(0, 70, 71),
        Nx=2500
    )

    # print a few rows
    print("theta(deg)    KN(in)    KG*sin(in)    GZ(in)")
    for k in range(0, len(th_deg), 10):
        print(f"{th_deg[k]:8.1f}  {KN[k]:8.3f}    {KGsn[k]:10.3f}   {GZ[k]:8.3f}")