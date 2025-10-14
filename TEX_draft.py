import numpy as np
import matplotlib.pyplot as plt

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

pw = 62.4           # lbf/ft^3 (water weight density)
pd = 0.74           # packing density of balls (solid fraction)

# Atmosphere & hydrostatics for compressed air modeling
P0_psf = 14.7 * 144.0        # atmospheric pressure [psf]
vent_height_in = R_in        # assume vents at top (hull crown). Submerge when h > vent_height_in

# Volumes from geometry
Vf = 3.86154        # ft^3 (front cone, from prior calc)
Vm_total = np.pi * (R**2) * (L_in*in_to_ft)  # recomputed straight tube volume
Vb = 4.47411        # ft^3 (back cone, from prior calc)
V_total = Vf + Vm_total + Vb

Vtm = 4*np.pi       # sealed motor compartment [ft^3]
V_free = V_total - Vtm   # free-flooding compartments [ft^3]

# Motor sealed tube length from volume Vtm and radius R
Lm = Vtm / (np.pi * R**2)

# Back cone paraboloid extension (for cut tip)
lb_extra = ((r_tip**2) / (R**2 - r_tip**2)) * lb
lb_tot = lb + lb_extra

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
    # avoid invalid divide where a==0
    denom = np.where(a[mid]==0.0, 1.0, a[mid])
    z[mid] = np.clip(h[mid]/denom, -1.0, 1.0)
    A[mid] = (a[mid]**2) * (np.pi - np.arccos(z[mid]) + z[mid]*np.sqrt(1 - z[mid]**2))
    A[wet] = np.pi * a[wet]**2
    A[dry] = 0.0
    return A

def A_above(a, h_w):
    return np.pi * np.asarray(a, float)**2 - A_submerged(a, h_w)

# ----------------------------
# Displaced volumes
# ----------------------------
def Vdisp_motor(h_in):
    h_ft = h_in * in_to_ft
    Ny = 12001
    y_m = np.linspace(0, Lm, Ny)
    Am = A_submerged(r_tube(y_m), h_ft)
    return np.trapz(Am, y_m)

def _V_above_geom(h_in_local):
    """Geometric 'air above WL' volume for the entire free-flooding geometry (no compression)."""
    h_ft = h_in_local * in_to_ft
    Ny = 12001

    # Front cone
    y_f = np.linspace(0, lf, Ny)
    Af_above = A_above(r_front(y_f), h_ft)

    # Free tube segment (tube minus sealed motor length)
    L_free = max(0.0, (L_in*in_to_ft) - Lm)
    y_t = np.linspace(0, L_free, Ny)
    At_above = A_above(r_tube(y_t), h_ft)

    # Back cone (cut at r_tip; profile parameterized by lb_tot)
    y_b = np.linspace(0, lb, Ny)
    Ab_above = A_above(r_back(y_b), h_ft)

    return np.trapz(Af_above, y_f) + np.trapz(At_above, y_t) + np.trapz(Ab_above, y_b)

def V_air_free(h_in, B):
    """
    Air volume in free-flooding spaces that displaces water.
    - Vents open (h <= vent_height_in): pocket open to atmosphere -> V_air = v * V_above_geom(h)
    - Vents submerged (h > vent_height_in): trapped air compresses isothermally:
        V_air = v * V_above_geom(h_closure) * P0 / (P0 + rho*g*depth_at_vent)
      with h_closure = vent_height_in.
    """
    # Remaining void fraction after adding balls
    v = max(0.0, 1.0 - pd*B)
    if v <= 0.0:
        return 0.0

    if h_in <= vent_height_in:
        return v * _V_above_geom(h_in)

    # Trapped at the instant vents go under
    V0 = v * _V_above_geom(vent_height_in)
    if V0 <= 0.0:
        return 0.0

    depth_ft = (h_in - vent_height_in) * in_to_ft  # depth of vent below external free surface (>=0)
    dP_psf = pw * depth_ft                          # pw is weight density => psf
    P_abs = P0_psf + dP_psf

    return V0 * (P0_psf / P_abs)

def V_balls_solid(B):
    return pd * B * V_free

def F_total(h_in, B):
    """Total buoyant force at waterline h_in (inches) and balls fraction B."""
    return pw * (Vdisp_motor(h_in) + V_balls_solid(B) + V_air_free(h_in, B))

# ----------------------------
# Equilibrium solver: solve for h given (W, B)
# ----------------------------
def solve_h_for_equilibrium(B, W, h_min_in=-(R_in+6.0), h_max_in=(R_in+6.0), tol=1e-3, maxit=60):
    a, b = h_min_in, h_max_in
    Fa = F_total(a, B) - W
    Fb = F_total(b, B) - W
    # If both sides are same sign, no guarantee of a root in [a,b]
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

# ----------------------------
# New: solve for B given (W, target h)
# ----------------------------
def solve_B_for_target_h(W, h_target_in, B_min=0.0, B_max=1.0, tol=1e-4, maxit=60):
    """
    Find the minimum B in [B_min, B_max] such that F_total(h_target, B) >= W.
    Equivalent to root of G(B) = F_total(h_target, B) - W, looking for sign change.
    Returns np.nan if no solution in the interval.
    """
    a, b = B_min, B_max
    def G(B): return F_total(h_target_in, B) - W
    Ga, Gb = G(a), G(b)

    # If even at max B we can't support W, no solution
    if Ga < 0 and Gb < 0:
        return np.nan
    # If even at min B we already support W, solution is B_min (minimum)
    if Ga >= 0:
        return a

    # Otherwise, bisection
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
# Build and plot h(B) curves for multiple weights
# ----------------------------
weights = np.arange(570, 1000+1, 25)
B_grid = np.linspace(0, 1, 401)

h_curves = []
for W in weights:
    hs = np.array([solve_h_for_equilibrium(B, W) for B in B_grid])
    h_curves.append(hs)

# y-limits from available solutions
all_h = np.concatenate([hs[~np.isnan(hs)] for hs in h_curves]) if len(h_curves)>0 else np.array([])
if all_h.size:
    min_h, max_h = np.min(all_h), np.max(all_h)
else:
    min_h, max_h = -R_in, R_in

plt.figure(figsize=(10,6))
for W, hs in zip(weights, h_curves):
    plt.plot(B_grid, hs, label=f"{W} lbf")
plt.axhline(0, linewidth=0.8, linestyle=":")
plt.xlabel("Balls fraction B (0–1)")
plt.ylabel("Equilibrium waterline h (inches)")
plt.title("Equilibrium Draft vs Foam Fraction\n(570–1000 lbf; shortened tube; trapped air included)")
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(ncol=3, fontsize=7)
plt.ylim(min_h-1, max_h+1)
plt.tight_layout()
plt.show()

print(f"Auto-scaled y-axis range: {min_h-1:.2f} to {max_h+1:.2f} inches")

# ----------------------------
# Report: which weights float with NO balls (B=0)
# ----------------------------
no_balls_ok = []
for W in weights:
    h0 = solve_h_for_equilibrium(0.0, W)
    if not np.isnan(h0):
        no_balls_ok.append((W, h0))

if no_balls_ok:
    w_min = min(w for w, _ in no_balls_ok)
    w_max = max(w for w, _ in no_balls_ok)
    print(f"\nWeights that can equilibrate with B=0 (no balls) span ~{w_min}–{w_max} lbf.")
    for w, h0 in no_balls_ok[:8]:
        print(f"  W={w:>4} lbf -> h={h0:6.2f} in at B=0")
else:
    print("\nNo equilibrium found at B=0 within the current h bracket.")

# ----------------------------
# New: Minimum B to hit target drafts
# ----------------------------
target_h_list = [0.0, R_in]  # change or add values as desired
print("\nMinimum B required to reach target drafts:")
for h_t in target_h_list:
    print(f"\n  Target h = {h_t:.2f} in")
    for W in weights:
        B_req = solve_B_for_target_h(W, h_t)
        if np.isnan(B_req):
            msg = "no solution in [0,1]"
        else:
            msg = f"B ≈ {B_req:.3f}"
        print(f"    W={W:>4} lbf: {msg}")

# ----------------------------
# Summary table: h at a few B values
# ----------------------------
B_marks = [0.00, 0.25, 0.50, 0.75, 1.00]
print("\nEquilibrium waterline h [in] at selected B values:")
header = "   W (lbf) | " + " | ".join([f"B={b:>4.2f}" for b in B_marks])
print(header)
print("-"*len(header))
for W in weights:
    row = []
    for b in B_marks:
        h_val = solve_h_for_equilibrium(b, W)
        row.append("  ---  " if np.isnan(h_val) else f"{h_val:7.2f}")
    print(f"{W:9.0f} | " + " | ".join(row))
