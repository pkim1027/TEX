import numpy as np

# ============================================================
# RSV Buoyancy — Motor Intact (sealed), Ball Fill ≤ 50%
# Computes surface equilibrium waterline h and draft for given W, B
# Draft is reported at the straight tube mid-body (positive downward).
# ============================================================

# ----------------------------
# Switches / Print Options
# ----------------------------
np.set_printoptions(suppress=True)
MOTOR_FLOODED = False   # Motor bay sealed/intact
B_MAX = 0.50            # Cap ball fill to 50%

# ----------------------------
# Geometry (inches) and constants
# ----------------------------
R_in   = 12.0         # tube outer radius [in]
L_in   = 240.0 - 51.0 # tube straight length shortened by 51 in -> 189 in
lf_in  = 29.5         # front cone length [in]
lb_in  = 31.5         # back cone nominal length [in]
r_tip_in = 3.5        # back cone cut radius [in]

in_to_ft = 1.0/12.0
R   = R_in*in_to_ft
lf  = lf_in*in_to_ft
lb  = lb_in*in_to_ft
r_tip = r_tip_in*in_to_ft

# Fluids & packing
pw = 62.4   # water weight density [lbf/ft^3]
pd = 0.74   # packing density of balls (solid fraction of filled volume)

# Atmosphere & venting (for trapped air model if vents submerge)
P0_psf = 14.7 * 144.0  # atmospheric pressure [psf]
vent_height_in = R_in  # vents at crown: submerge once h > 12 in

# ----------------------------
# Volumes from geometry
# ----------------------------
# You can keep these calibrated values for cones, or replace with your own if needed:
Vf = 3.86154        # ft^3 (front cone)  - from your calibration
Vb = 4.47411        # ft^3 (back cone)   - from your calibration
Vm_total = np.pi * (R**2) * (L_in*in_to_ft)  # straight tube volume [ft^3]
V_total = Vf + Vm_total + Vb

# Sealed motor compartment volume (use 4*pi ft^3 as in your model)
Vtm = 4.0*np.pi                     # [ft^3]
Lm  = Vtm / (np.pi * R**2)          # sealed motor tube length [ft]

# Back cone paraboloid extension due to cut tip (affects geometry profile; volume pre-calibrated)
lb_extra = ((r_tip**2) / (R**2 - r_tip**2)) * lb
lb_tot   = lb + lb_extra

# Effective free-flooding volume baseline
V_free = V_total if MOTOR_FLOODED else (V_total - Vtm)

# ----------------------------
# Geometry profiles (radius vs axial y; y=0 at each section start)
# ----------------------------
def r_front(y):  # y in [0, lf]
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lf))

def r_back(y):   # y in [0, lb] but shape uses lb_tot
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lb_tot))

def r_tube(y):   # constant radius along straight section
    return R * np.ones_like(y)

# ----------------------------
# Circular segment areas (submerged vs above-water inside the hull)
# h is waterline height from centerline (inches): +R at crown, -R at keel
# ----------------------------
def A_submerged(a, h_w_in):
    a = np.asarray(a, dtype=float)
    h = np.broadcast_to(np.asarray(h_w_in, dtype=float), a.shape)
    A = np.zeros_like(a, dtype=float)
    dry = h <= -a
    wet = h >= a
    mid = (~dry) & (~wet)
    z = np.empty_like(a, dtype=float)
    denom = np.where(a[mid]==0.0, 1.0, a[mid])
    z[mid] = np.clip(h[mid]/denom, -1.0, 1.0)
    A[mid] = (a[mid]**2) * (np.pi - np.arccos(z[mid]) + z[mid]*np.sqrt(1 - z[mid]**2))
    A[wet] = np.pi * a[wet]**2
    A[dry] = 0.0
    return A

def A_above(a, h_w_in):
    return np.pi * np.asarray(a, float)**2 - A_submerged(a, h_w_in)

# ----------------------------
# Displaced volumes (surface equilibrium mode)
# ----------------------------
def Vdisp_motor(h_in):
    """Sealed motor displacement (0 if MOTOR_FLOODED). Integrates over sealed length Lm."""
    if MOTOR_FLOODED:
        return 0.0
    h_ft = h_in * in_to_ft
    Ny  = 6001
    y_m = np.linspace(0, Lm, Ny)
    Am  = A_submerged(r_tube(y_m), h_ft)
    return np.trapz(Am, y_m)

def _V_above_geom(h_in_local):
    """
    Geometric 'air above WL' volume for the whole free-flooding region.
    If MOTOR_FLOODED: the entire straight tube is free-flooding.
    Else: straight tube excludes sealed motor length.
    """
    h_ft = h_in_local * in_to_ft
    Ny = 6001

    # Front cone
    y_f = np.linspace(0, lf, Ny)
    Af_above = A_above(r_front(y_f), h_ft)

    # Straight tube (exclude sealed length when motor intact)
    if MOTOR_FLOODED:
        L_free_len = (L_in*in_to_ft)
    else:
        L_free_len = max(0.0, (L_in*in_to_ft) - Lm)
    y_t = np.linspace(0, L_free_len, Ny)
    At_above = A_above(r_tube(y_t), h_ft)

    # Back cone (shape uses lb_tot, but we integrate physical lb; your Vb was precomputed)
    y_b = np.linspace(0, lb, Ny)
    Ab_above = A_above(r_back(y_b), h_ft)

    return np.trapz(Af_above, y_f) + np.trapz(At_above, y_t) + np.trapz(Ab_above, y_b)

def V_air_free(h_in, B):
    """
    Air volume in free-flooding spaces that displaces water.
    Vents open (h <= vent_height_in):  V_air = v * V_above_geom(h)
    Vents submerged (h > vent_height_in): trapped air compresses isothermally.
    """
    B_eff = min(max(B, 0.0), B_MAX)        # clamp to [0, B_MAX]
    v = max(0.0, 1.0 - pd*B_eff)           # remaining void fraction
    if v <= 0.0:
        return 0.0

    if h_in <= vent_height_in:
        return v * _V_above_geom(h_in)

    # Vents submerged: capture V0 at closure, then compress with depth
    V0 = v * _V_above_geom(vent_height_in)
    if V0 <= 0.0:
        return 0.0
    depth_ft = (h_in - vent_height_in) * in_to_ft  # vent depth below free surface
    dP_psf = pw * depth_ft
    P_abs = P0_psf + dP_psf
    return V0 * (P0_psf / P_abs)

def V_balls_solid(B):
    B_eff = min(max(B, 0.0), B_MAX)
    return pd * B_eff * V_free

def F_total(h_in, B):
    """Total buoyant force at waterline h_in (inches) for ball fill B (≤ B_MAX)."""
    return pw * (Vdisp_motor(h_in) + V_balls_solid(B) + V_air_free(h_in, B))

# ----------------------------
# Surface-mode solvers
# ----------------------------
def solve_h_for_equilibrium(B, W, h_min_in=-(R_in+8.0), h_max_in=(R_in+8.0), tol=1e-3, maxit=60):
    """
    Solve for waterline height h (in) such that F_total(h,B) = W.
    Returns np.nan if no root is bracketed in [h_min_in, h_max_in].
    """
    B_eff = min(max(B, 0.0), B_MAX)
    a, b = h_min_in, h_max_in
    Fa = F_total(a, B_eff) - W
    Fb = F_total(b, B_eff) - W
    if Fa > 0 and Fb > 0:
        return np.nan
    if Fa < 0 and Fb < 0:
        return np.nan
    for _ in range(maxit):
        c = 0.5*(a+b)
        Fc = F_total(c, B_eff) - W
        if abs(Fc) < 1e-2 or (b-a)/2 < tol:
            return c
        if np.sign(Fc) == np.sign(Fa):
            a, Fa = c, Fc
        else:
            b, Fb = c, Fc
    return c

def solve_B_for_target_h(W, h_target_in, B_min=0.0, B_max=B_MAX, tol=1e-4, maxit=60):
    """
    Minimum B in [B_min, B_max] so that F_total(h_target, B) >= W.
    Returns np.nan if even B_max is insufficient.
    """
    a, b = max(0.0, B_min), min(B_MAX, B_max)
    def G(B): return F_total(h_target_in, B) - W
    Ga, Gb = G(a), G(b)
    if Ga < 0 and Gb < 0:
        return np.nan
    if Ga >= 0:
        return a
    for _ in range(maxit):
        c = 0.5*(a+b)
        Gc = G(c)
        if abs(Gc) < 1e-2 or (b-a)/2 < tol:
            return c
        if np.sign(Gc) == np.sign(Ga):
            a, Ga = c, Gc
        else:
            b, Gb = c, Gc
    return c

# ----------------------------
# Convenience helpers
# ----------------------------
def draft_from_h(h_in):
    """Draft (inches, positive downward) at mid-body for given waterline height h."""
    return R_in - h_in

def draft_at_fill(W, B):
    """Draft for a given weight W and ball fill B (≤ 0.5). Returns np.nan if no surface solution."""
    h = solve_h_for_equilibrium(B, W)
    return np.nan if np.isnan(h) else draft_from_h(h)

# ----------------------------
# Example runs (edit as needed)
# ----------------------------
if __name__ == "__main__":
    # Example 1: Single weight & B=0.50 (half volume filled with balls)
    W_example = 750.0   # lbf (edit)
    B_example = 1  
    h_eq = solve_h_for_equilibrium(B_example, W_example)
    if np.isnan(h_eq):
        print(f"[B={B_example:.2f}] No surface equilibrium found for W={W_example:.1f} lbf within bracket.")
    else:
        print(f"[B={B_example:.2f}] W={W_example:.1f} lbf -> h = {h_eq:.2f} in, draft = {draft_from_h(h_eq):.2f} in")

    # Example 2: What B (≤ 0.50) is needed to sit at h_target = 12 in (≈ 12 in draft for 24 in tube)?
    h_target = 12.0
    B_req = solve_B_for_target_h(W_example, h_target)
    if np.isnan(B_req):
        print(f"B_req(≤{B_MAX:.2f}) for W={W_example:.1f} lbf at h={h_target:.2f} in: > {B_MAX:.2f} (not achievable within cap)")
    else:
        print(f"B_req(≤{B_MAX:.2f}) for W={W_example:.1f} lbf at h={h_target:.2f} in: {B_req:.3f}")

    # Example 3 (optional): quick table across weights at B=0.50
    weights = np.arange(570, 1000+1, 50)
    print("\nTable — Draft at B=0.50 (motor intact)")
    print("   W (lbf) |    h (in) |  Draft (in)")
    print("-----------+-----------+------------")
    for W in weights:
        hW = solve_h_for_equilibrium(0.50, W)
        if np.isnan(hW):
            print(f"{W:10.0f} |   (no eq) |   (no eq)")
        else:
            print(f"{W:10.0f} | {hW:9.2f} | {draft_from_h(hW):10.2f}")
