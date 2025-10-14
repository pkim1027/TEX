import numpy as np

# ----------------------------
# Switches / Print Options
# ----------------------------
MOTOR_FLOODED = True   # worst-case: motor bay floods
np.set_printoptions(suppress=True)

# ----------------------------
# Geometry (inches) and constants
# ----------------------------
R_in = 12.0         # tube outer radius [in]
L_in = 240.0 - 51.0 # tube length shortened by 51 in = 189 in
lf_in = 29.5        # front cone length [in]
lb_in = 31.5        # back cone nominal length [in]
r_tip_in = 3.5      # back cone cut radius [in]

in_to_ft = 1.0/12.0
R = R_in*in_to_ft
lf = lf_in*in_to_ft
lb = lb_in*in_to_ft
r_tip = r_tip_in*in_to_ft

pw = 62.4           # lbf/ft^3 (water weight density; using weight units)
pd = 0.74           # packing density of balls (solid fraction)

# Atmosphere & hydrostatics for compressed-air modeling
P0_psf = 14.7 * 144.0        # atmospheric pressure [psf]
vent_height_in = R_in        # assume vents at hull crown: submerge when h > vent_height_in

# ----------------------------
# Volumes from geometry
# ----------------------------
Vf = 3.86154        # ft^3 (front cone)
Vm_total = np.pi * (R**2) * (L_in*in_to_ft)  # straight tube volume
Vb = 4.47411        # ft^3 (back cone)
V_total = Vf + Vm_total + Vb

Vtm = 4*np.pi       # sealed motor compartment [ft^3] if sealed
Lm = Vtm / (np.pi * R**2)  # sealed motor tube length

# Back cone paraboloid extension (due to cut tip)
lb_extra = ((r_tip**2) / (R**2 - r_tip**2)) * lb
lb_tot = lb + lb_extra

# Effective free-flooding volume baseline
V_free = V_total if MOTOR_FLOODED else (V_total - Vtm)

# ----------------------------
# Geometry profiles
# ----------------------------
def r_front(y):
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lf))

def r_back(y):
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lb_tot))

def r_tube(y):
    return R * np.ones_like(y)

# ----------------------------
# Circular segment areas
# ----------------------------
def A_submerged(a, h_w):
    a = np.asarray(a, dtype=float)
    h = np.broadcast_to(np.asarray(h_w, dtype=float), a.shape)
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

def A_above(a, h_w):
    return np.pi * np.asarray(a, float)**2 - A_submerged(a, h_w)

# ----------------------------
# Displaced volumes (surface equilibrium mode)
# ----------------------------
def Vdisp_motor(h_in):
    """Sealed motor displacement; zero if MOTOR_FLOODED worst-case."""
    if MOTOR_FLOODED:
        return 0.0
    h_ft = h_in * in_to_ft
    Ny = 12001
    y_m = np.linspace(0, Lm, Ny)
    Am = A_submerged(r_tube(y_m), h_ft)
    return np.trapz(Am, y_m)

def _V_above_geom(h_in_local):
    """
    Geometric 'air above WL' volume for the whole free-flooding region.
    If MOTOR_FLOODED: the entire straight tube is free-flooding.
    Else: straight tube excludes sealed motor length.
    """
    h_ft = h_in_local * in_to_ft
    Ny = 12001

    y_f = np.linspace(0, lf, Ny)
    Af_above = A_above(r_front(y_f), h_ft)

    if MOTOR_FLOODED:
        L_free_len = (L_in*in_to_ft)
    else:
        L_free_len = max(0.0, (L_in*in_to_ft) - Lm)
    y_t = np.linspace(0, L_free_len, Ny)
    At_above = A_above(r_tube(y_t), h_ft)

    y_b = np.linspace(0, lb, Ny)
    Ab_above = A_above(r_back(y_b), h_ft)

    return np.trapz(Af_above, y_f) + np.trapz(At_above, y_t) + np.trapz(Ab_above, y_b)

def V_air_free(h_in, B):
    """
    Air volume in free-flooding spaces that displaces water.
    Vents open (h <= vent_height_in):  V_air = v * V_above_geom(h)
    Vents submerged (h > vent_height_in): trapped air compresses isothermally.
    """
    v = max(0.0, 1.0 - pd*B)
    if v <= 0.0:
        return 0.0

    if h_in <= vent_height_in:
        return v * _V_above_geom(h_in)

    V0 = v * _V_above_geom(vent_height_in)
    if V0 <= 0.0:
        return 0.0

    depth_ft = (h_in - vent_height_in) * in_to_ft  # depth of vent below free surface (>=0)
    dP_psf = pw * depth_ft
    P_abs = P0_psf + dP_psf

    return V0 * (P0_psf / P_abs)

def V_balls_solid(B):
    return pd * B * V_free

def F_total(h_in, B):
    """Total buoyant force at waterline h_in (inches) and balls fraction B."""
    return pw * (Vdisp_motor(h_in) + V_balls_solid(B) + V_air_free(h_in, B))

# ----------------------------
# Surface-mode solvers (for completeness)
# ----------------------------
def solve_h_for_equilibrium(B, W, h_min_in=-(R_in+6.0), h_max_in=(R_in+6.0), tol=1e-3, maxit=60):
    a, b = h_min_in, h_max_in
    Fa = F_total(a, B) - W
    Fb = F_total(b, B) - W
    if Fa > 0 and Fb > 0:
        return np.nan
    if Fa < 0 and Fb < 0:
        return np.nan
    for _ in range(maxit):
        c = 0.5*(a+b)
        Fc = F_total(c, B) - W
        if abs(Fc) < 1e-2 or (b-a)/2 < tol:
            return c
        if np.sign(Fc) == np.sign(Fa):
            a, Fa = c, Fc
        else:
            b, Fb = c, Fc
    return c

def solve_B_for_target_h(W, h_target_in, B_min=0.0, B_max=1.0, tol=1e-4, maxit=60):
    """Minimum B in [B_min, B_max] so that F_total(h_target, B) >= W."""
    def G(B): return F_total(h_target_in, B) - W
    a, b = B_min, B_max
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
# Fully-submerged (motor flooded) calculators
# ----------------------------
def B_required_deep(W):
    """Deep-submerged conservative requirement: trapped air ~ 0."""
    return W / (pw * pd * V_free)

def B_required_at_depth(W, depth_ft, tol=1e-6, maxit=60):
    """
    Required B for neutral buoyancy at depth_ft below the free surface (vents submerged),
    including compressed trapped air that was captured at closure.
    """
    V0_geom = _V_above_geom(vent_height_in)  # geometric trapped-air volume with v=1 at closure

    def F_total_at_depth(B):
        v = max(0.0, 1.0 - pd*B)
        V0 = v * V0_geom
        V_air = 0.0
        if depth_ft >= 0.0 and V0 > 0.0:
            P_abs = P0_psf + pw * depth_ft
            V_air = V0 * (P0_psf / P_abs)
        return pw * (pd * B * V_free + V_air)

    # Feasibility checks
    if F_total_at_depth(1.0) < W:
        return np.nan
    if F_total_at_depth(0.0) >= W:
        return 0.0

    # Bisection on [0, 1]
    a, b = 0.0, 1.0
    Fa, Fb = F_total_at_depth(a) - W, F_total_at_depth(b) - W
    for _ in range(maxit):
        c = 0.5*(a+b)
        Fc = F_total_at_depth(c) - W
        if abs(Fc) < 1e-3 or (b-a) < tol:
            return c
        if np.sign(Fc) == np.sign(Fa):
            a, Fa = c, Fc
        else:
            b, Fb = c, Fc
    return 0.5*(a+b)

# ----------------------------
# RUN: Generate all the tables (no plots)
# ----------------------------
weights = np.arange(570, 1000+1, 25)
B_marks = [0.00, 0.25, 0.50, 0.75, 1.00]
target_h_list = [0.0, R_in]       # surface-mode target drafts (inches)
depth_samples_ft = [0.0, 1.0, 3.0, 6.0, 10.0]  # submerged-mode sample depths

print("=== MODE: MOTOR FLOODED (worst case) ===\n")

# Table A: which weights float with NO balls (B=0) in this mode (surface equilibrium)
print("Table A — Surface Mode: Equilibria with B=0 (no balls)")
no_balls_rows = []
for W in weights:
    h0 = solve_h_for_equilibrium(0.0, W)
    if not np.isnan(h0):
        no_balls_rows.append((W, h0))
if no_balls_rows:
    print("   W (lbf) |  h at B=0 (in)")
    print("-----------+----------------")
    for w, h0 in no_balls_rows:
        print(f"{w:10.0f} | {h0:12.2f}")
else:
    print("  None found within search bracket.")
print()

# Table B: Minimum B to reach target drafts (surface mode)
print("Table B — Surface Mode: Minimum B to reach target drafts")
for h_t in target_h_list:
    print(f"  Target h = {h_t:.2f} in")
    print("   W (lbf) |   B_required  ")
    print("-----------+---------------")
    for W in weights:
        B_req = solve_B_for_target_h(W, h_t)
        if np.isnan(B_req):
            msg = "no solution in [0,1]"
        else:
            msg = f"{B_req:.3f}"
        print(f"{W:10.0f} | {msg:>13}")
    print()

# Table C: Equilibrium waterline h at selected B values (surface mode)
print("Table C — Surface Mode: Equilibrium waterline h [in] at selected B")
header = "   W (lbf) | " + " | ".join([f"B={b:>4.2f}" for b in B_marks])
print(header)
print("-"*len(header))
for W in weights:
    row_vals = []
    for b in B_marks:
        h_val = solve_h_for_equilibrium(b, W)
        row_vals.append("  ---  " if np.isnan(h_val) else f"{h_val:7.2f}")
    print(f"{W:10.0f} | " + " | ".join(row_vals))
print()

# Table D: Fully Submerged — Deep-submerged conservative requirement
print("Table D — Fully Submerged: Deep-submerged conservative B_required (air ~ 0)")
print("   W (lbf) |   B_required  | Feasible?")
print("-----------+---------------+----------")
for W in weights:
    Bdeep = B_required_deep(W)
    feas = "YES" if Bdeep <= 1.0 else "NO"
    print(f"{W:10.0f} | {Bdeep:13.3f} | {feas}")
print()

# Table E: Fully Submerged — Depth-dependent B_required with compressed air
print("Table E — Fully Submerged: B_required at depth (with compressed trapped air)")
for z in depth_samples_ft:
    print(f"  Depth = {z:.1f} ft")
    print("   W (lbf) |   B_required  | Feasible?")
    print("-----------+---------------+----------")
    for W in weights:
        Bz = B_required_at_depth(W, z)
        if np.isnan(Bz):
            print(f"{W:10.0f} | {'>1.000':>13} | NO")
        else:
            feas = "YES" if Bz <= 1.0 else "NO"
            print(f"{W:10.0f} | {Bz:13.3f} | {feas}")
    print()
