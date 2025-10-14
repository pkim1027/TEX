import numpy as np

# ============================================================
# RSV Buoyancy — Motor Intact (sealed), Wall Thickness = 0.25 in
# Displacement uses OUTER geometry; air/balls use INNER geometry.
# Ball fill B is capped at 0.50 (half the floodable volume).
# ============================================================

np.set_printoptions(suppress=True)

# ----------------------------
# Config / Physical constants
# ----------------------------
MOTOR_FLOODED = False      # Motor bay sealed/intact
B_MAX = 0.50               # cap on ball fill fraction
t_wall_in = 0.25           # wall thickness [in]
pw = 62.4                  # water weight density [lbf/ft^3]
pd = 0.74                  # packing density of balls (solid fraction of filled volume)
P0_psf = 14.7 * 144.0      # atmospheric pressure [psf]
gas_n = 1.0                # polytropic exponent (1.0=isothermal; ~1.2–1.4 fast compression)

# ----------------------------
# Geometry (inches)
# ----------------------------
R_out_in = 12.0            # tube outer radius [in]
R_in_in  = max(0.0, R_out_in - t_wall_in)  # tube inner radius [in]
L_in     = 240.0 - 51.0    # straight tube length [in]
lf_in    = 29.5            # front cone length [in] (outer)
lb_in    = 31.5            # back cone nominal length [in] (outer)
r_tip_in = 3.5             # back cone cut radius (outer) [in]

# Unit conversions
in_to_ft = 1.0/12.0
R_out = R_out_in * in_to_ft
R_inn = R_in_in  * in_to_ft
L     = L_in     * in_to_ft
lf    = lf_in    * in_to_ft
lb    = lb_in    * in_to_ft
r_tip = r_tip_in * in_to_ft

# Vents at crown (outer): submerge when h > R_out_in (inches from CL)
vent_height_in = R_out_in

# ----------------------------
# Back-cone extension model (outer/inner)
# Use same shape but subtract thickness for inner radius.
# ----------------------------
# Outer profile radius vs local y (0 at cone base)
def r_front_out(y):  # y in [0, lf]
    return R_out * np.sqrt(np.maximum(0.0, 1.0 - y/lf))

def r_back_out(y):   # y in [0, lb_tot_out]
    return R_out * np.sqrt(np.maximum(0.0, 1.0 - y/lb_tot_out))

# Paraboloid extension for back cone due to tip cut (outer)
lb_extra_out = ((r_tip**2) / (R_out**2 - r_tip**2)) * lb
lb_tot_out   = lb + lb_extra_out

# Inner profiles: thin-wall offset (clamped at 0)
def r_front_inn(y):
    return np.maximum(0.0, r_front_out(y) - (t_wall_in*in_to_ft))

def r_back_inn(y):
    # inner shape follows the outer extension but with radius reduced by t
    return np.maximum(0.0, r_back_out(y) - (t_wall_in*in_to_ft))

def r_tube_out(y):  # straight tube
    return R_out * np.ones_like(y)

def r_tube_inn(y):
    return R_inn * np.ones_like(y)

# ----------------------------
# Circular-segment areas for a circle of radius a (feet), at WL height h (inches)
# WARNING: A_submerged uses 'a' in FEET and 'h' in INCHES. Internally converts.
# ----------------------------
def A_submerged(a_ft, h_in):
    a = np.asarray(a_ft, dtype=float)
    # convert h to feet to compare on same units as radius
    h_ft = np.broadcast_to(np.asarray(h_in, dtype=float), a.shape) * in_to_ft
    A = np.zeros_like(a, dtype=float)
    dry = h_ft <= -a         # WL below keel -> dry
    wet = h_ft >= a          # WL above crown -> fully wetted
    mid = (~dry) & (~wet)
    z = np.empty_like(a, dtype=float)
    denom = np.where(a[mid]==0.0, 1.0, a[mid])
    z[mid] = np.clip(h_ft[mid]/denom, -1.0, 1.0)
    A[mid] = (a[mid]**2) * (np.pi - np.arccos(z[mid]) + z[mid]*np.sqrt(1 - z[mid]**2))
    A[wet] = np.pi * a[wet]**2
    A[dry] = 0.0
    return A

def A_above(a_ft, h_in):
    return np.pi * np.asarray(a_ft, float)**2 - A_submerged(a_ft, h_in)

# ----------------------------
# Volume helpers (numerical, consistent geometry)
# ----------------------------
def integrate_volume_of_revolution(r_func, y0, y1, Ny=6001):
    y = np.linspace(y0, y1, Ny)
    r = r_func(y)
    return np.trapz(np.pi * r**2, y)  # ∫ π r(y)^2 dy

# External (displacement) volumes
V_front_out = integrate_volume_of_revolution(r_front_out, 0.0, lf)
V_tube_out  = np.pi * (R_out**2) * L
V_back_out  = integrate_volume_of_revolution(r_back_out, 0.0, lb)  # integrate physical lb (not lb_tot_out)
V_total_out = V_front_out + V_tube_out + V_back_out

# Internal (floodable) straight-tube area (for motor bay length, etc.)
A_tube_inn = np.pi * (R_inn**2)

# ----------------------------
# Sealed motor bay: specified by INTERNAL volume Vtm (ft^3)
# Use inner area to compute its sealed length.
# ----------------------------
Vtm = 4.0*np.pi               # given sealed motor internal volume [ft^3]
Lm  = Vtm / max(A_tube_inn, 1e-12)  # sealed motor length (ft) along straight tube (inner)

# ----------------------------
# INTERNAL 'above WL' geometric volume (floodable regions only)
# Uses INNER radii and excludes sealed motor length from the tube
# ----------------------------
def V_above_geom_internal(h_in):
    Ny = 6001

    # Front cone (inner)
    y_f = np.linspace(0, lf, Ny)
    Af_above = A_above(r_front_inn(y_f), h_in)

    # Straight tube inner length available to flood (exclude Lm)
    L_free = L if MOTOR_FLOODED else max(0.0, L - Lm)
    y_t = np.linspace(0, L_free, Ny)
    At_above = A_above(r_tube_inn(y_t), h_in)

    # Back cone (inner)
    y_b = np.linspace(0, lb, Ny)
    Ab_above = A_above(r_back_inn(y_b), h_in)

    return np.trapz(Af_above, y_f) + np.trapz(At_above, y_t) + np.trapz(Ab_above, y_b)

# ----------------------------
# INTERNAL floodable volume (for balls): constant capacity
# (Useful for sanity checks and V_balls_solid; not needed for air calc which is level-dependent.)
# ----------------------------
V_front_inn = integrate_volume_of_revolution(r_front_inn, 0.0, lf)
V_tube_inn_free = (np.pi * R_inn**2) * (L if MOTOR_FLOODED else max(0.0, L - Lm))
V_back_inn  = integrate_volume_of_revolution(r_back_inn, 0.0, lb)
V_floodable_internal = V_front_inn + V_tube_inn_free + V_back_inn  # ft^3

# ----------------------------
# Displacement from sealed motor section (external wetted area integrated over Lm)
# ----------------------------
def Vdisp_motor_external(h_in):
    if MOTOR_FLOODED:
        return 0.0
    Ny = 4001
    y_m = np.linspace(0, Lm, Ny)                  # along the sealed straight length
    Am  = A_submerged(r_tube_out(y_m), h_in)      # OUTER radius for displacement
    return np.trapz(Am, y_m)

# ----------------------------
# Air & balls
# ----------------------------
def V_air_trapped(h_in, B):
    """
    Air volume in floodable spaces that displaces water.
    If h <= vent_height_in: vents open -> air at ambient: v * V_above(h).
    If h > vent_height_in: vents submerged -> capture V0 at closure,
    then compress polytropically: P V^n = const.
    """
    B_eff = min(max(B, 0.0), B_MAX)
    v = max(0.0, 1.0 - pd*B_eff)  # remaining void fraction after packing
    if v <= 0.0:
        return 0.0

    if h_in <= vent_height_in:
        return v * V_above_geom_internal(h_in)

    # capture at closure
    V0 = v * V_above_geom_internal(vent_height_in)
    if V0 <= 0.0:
        return 0.0
    depth_ft = (h_in - vent_height_in) * in_to_ft  # depth of vent below free surface
    P_abs = P0_psf + pw * depth_ft
    # P0 * V0^n = P_abs * V^n  => V = V0 * (P0/P_abs)^(1/n)
    return V0 * (P0_psf / P_abs)**(1.0 / max(gas_n, 1e-9))

def V_balls_solid(B):
    B_eff = min(max(B, 0.0), B_MAX)
    return pd * B_eff * V_floodable_internal

def F_total_surface(h_in, B):
    """
    Total buoyant force at WL h_in (inches) with fill B (≤ B_MAX):
      - External displacement of *sealed motor length* (explicitly integrated)
      - External displacement of *everything else* is captured implicitly through air/balls term:
        We don't need full external volume; we add the *internal* displaced terms (balls + air),
        plus the *sealed* section external displacement.
    """
    # Buoyancy from balls (solid inside) + trapped/open air (internal) is pw * (V_balls + V_air)
    # Add sealed motor external displacement (since it doesn't flood internally)
    return pw * (V_balls_solid(B) + V_air_trapped(h_in, B) + Vdisp_motor_external(h_in))

# ----------------------------
# Surface-mode solvers
# ----------------------------
def solve_h_for_equilibrium(B, W, h_min_in=-(R_out_in+8.0), h_max_in=(R_out_in+8.0), tol=1e-3, maxit=60):
    """Solve for WL height h (in) s.t. F_total_surface(h,B) = W. Returns np.nan if not bracketed."""
    B_eff = min(max(B, 0.0), B_MAX)
    a, b = h_min_in, h_max_in
    Fa = F_total_surface(a, B_eff) - W
    Fb = F_total_surface(b, B_eff) - W
    if Fa > 0 and Fb > 0:
        return np.nan
    if Fa < 0 and Fb < 0:
        return np.nan
    for _ in range(maxit):
        c = 0.5*(a+b)
        Fc = F_total_surface(c, B_eff) - W
        if abs(Fc) < 1e-2 or (b-a)/2 < tol:
            return c
        if np.sign(Fc) == np.sign(Fa):
            a, Fa = c, Fc
        else:
            b, Fb = c, Fc
    return c

def solve_B_for_target_h(W, h_target_in, B_min=0.0, B_max=B_MAX, tol=1e-4, maxit=60):
    """
    Minimum B in [B_min, B_max] to meet/exceed buoyancy at target h.
    Returns np.nan if even B_max is insufficient.
    """
    a, b = max(0.0, B_min), min(B_MAX, B_max)
    def G(B): return F_total_surface(h_target_in, B) - W
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
# Convenience
# ----------------------------
def draft_from_h(h_in):
    """Draft (inches, positive downward) at mid-body for given WL height h."""
    return R_out_in - h_in

def draft_at_fill(W, B):
    h = solve_h_for_equilibrium(B, W)
    return (np.nan, np.nan) if np.isnan(h) else (h, draft_from_h(h))

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example 1: Single weight at B=0.50 (half fill)
    W_example = 750.0
    B_example = 0.50
    h_eq = solve_h_for_equilibrium(B_example, W_example)
    if np.isnan(h_eq):
        print(f"[B={B_example:.2f}] No surface equilibrium for W={W_example:.1f} lbf within bracket.")
    else:
        print(f"[B={B_example:.2f}] W={W_example:.1f} lbf -> h = {h_eq:.2f} in, draft = {draft_from_h(h_eq):.2f} in")

    # Example 2: Minimum B (≤ 0.50) to sit at h = 12 in (≈ 12-in draft for 24-in OD)
    h_target = 12.0
    B_req = solve_B_for_target_h(W_example, h_target)
    if np.isnan(B_req):
        print(f"B_req(≤{B_MAX:.2f}) for W={W_example:.1f} at h={h_target:.2f} in: > {B_MAX:.2f} (not achievable within cap)")
    else:
        print(f"B_req(≤{B_MAX:.2f}) for W={W_example:.1f} at h={h_target:.2f} in: {B_req:.3f}")

    # Example 3: Quick table across weights at B=0.50
    weights = np.arange(570, 1000+1, 50)
    print("\nTable — Draft at B=0.50 (motor intact, t=0.25 in)")
    print("   W (lbf) |    h (in) |  Draft (in)")
    print("-----------+-----------+------------")
    for W in weights:
        hW = solve_h_for_equilibrium(0.50, W)
        if np.isnan(hW):
            print(f"{W:10.0f} |   (no eq) |   (no eq)")
        else:
            print(f"{W:10.0f} | {hW:9.2f} | {draft_from_h(hW):10.2f}")
