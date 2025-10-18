# -------------------------------------------------
# TEX Hull — Center of Buoyancy + Profile Plotter
# -------------------------------------------------
# - Computes center of buoyancy for the sealed motor bay section
# - Accepts either a given draft OR solves draft from total weight
# - Draws the hull profile, section markers, waterline, and CB point
# -------------------------------------------------

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# SEGMENT GEOMETRY (CIRCULAR SEGMENT)
# ----------------------------
EPS = 1e-9

def seg_area_in2(d_in: float, R_in: float) -> float:
    """Submerged circular-segment area (in^2) for radius R_in (in), draft d_in (in) measured up from bottom."""
    d = max(0.0, min(d_in, 2.0 * R_in))
    if d <= EPS:
        return 0.0
    if d >= 2.0 * R_in - EPS:
        return math.pi * R_in * R_in
    u = (R_in - d) / max(EPS, R_in)            # may be slightly outside [-1,1]
    u = max(-1.0, min(1.0, u))
    A1 = (R_in**2) * math.acos(u)
    rad = max(0.0, 2.0 * R_in * d - d * d)
    A2 = (R_in - d) * math.sqrt(rad)
    return A1 - A2

def seg_theta(d_in: float, R_in: float) -> float:
    d = max(0.0, min(d_in, 2.0 * R_in))
    if R_in <= 0.0:
        return 0.0
    u = (R_in - d) / R_in
    u = max(-1.0, min(1.0, u))
    return 2.0 * math.acos(u)

def seg_centroid_below_WL_in(d_in: float, R_in: float) -> float:
    """Centroid depth (in) measured downward from WL to the segment centroid."""
    d = max(0.0, min(d_in, 2.0 * R_in))
    if d <= EPS:
        return 0.0
    if d >= 2.0 * R_in - EPS:
        return R_in  # full circle: centroid is at center (R below WL)
    theta = seg_theta(d, R_in)                 # safe theta
    denom = theta - math.sin(theta)
    if abs(denom) < 1e-14:
        return 0.0
    return (4.0 * R_in * (math.sin(theta / 2.0) ** 3)) / (3.0 * denom)

# ----------------------------
# INPUTS (HYDRO + GEOMETRY)
# ----------------------------
@dataclass
class Inputs:
    # Hull (outer surface dims)
    R_out_in: float = 12.0
    t_wall_in: float = 0.25  # info only; not used in surface-mode buoyancy
    L_front_in: float = 29.5
    L_cyl_in: float = 182.0
    L_back_in: float = 31.5

    # Hydro
    draft_in: float = 12.0        # outside draft at motor-bay section (in)
    gamma: float = 62.4           # lbf/ft^3 (fresh water)

    # Motor bay placement (along the straight cylinder)
    back_offset_from_back_cone_in: float = 61.5
    motor_bay_length_in: float = 50.0

    # Optional solve for draft given total weight
    W_total_lbf: Optional[float] = None
    solve_equilibrium: bool = False
    fully_submerged: bool = False   # keep this; remove the extra solve_equilibrium line


# ----------------------------
# CORE COMPUTATIONS (CB FOR SEALED MOTOR BAY)
# ----------------------------
def bay_span_from_back_offset(L_cyl_in: float, back_offset_in: float, bay_len_in: float):
    """Return (s_front, s_back) in cylinder coordinates [0, L_cyl]."""
    s_back = max(0.0, min(L_cyl_in, L_cyl_in - back_offset_in))
    s_front = max(0.0, min(L_cyl_in, s_back - bay_len_in))
    if s_front > s_back:
        s_front, s_back = s_back, s_front
    return s_front, s_back

def displaced_volume_from_draft_ft3(R_out_in: float, draft_in: float, L_bay_in: float) -> float:
    """Submerged volume (ft^3) of the sealed bay at the given draft."""
    A_in2 = seg_area_in2(draft_in, R_out_in)
    V_in3 = A_in2 * max(0.0, L_bay_in)
    return V_in3 / 1728.0

def buoyancy_lbf_from_draft(R_out_in: float, draft_in: float, L_bay_in: float, gamma: float) -> float:
    return gamma * displaced_volume_from_draft_ft3(R_out_in, draft_in, L_bay_in)

def solve_draft_no_foam(W_lbf: float, R_out_in: float, L_bay_in: float, gamma: float,
                        d_lo_in: float = 0.0, d_hi_in: float = None, iters: int = 60) -> float:
    """Solve draft d such that buoyancy equals W_lbf for the sealed bay only (bisection in [0, 2R])."""
    if d_hi_in is None:
        d_hi_in = 2.0 * R_out_in
    lo, hi = d_lo_in, d_hi_in
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        B_mid = buoyancy_lbf_from_draft(R_out_in, mid, L_bay_in, gamma)
        if B_mid < W_lbf:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def compute_cb_surface(inp: Inputs):
    # bay span along cylinder
    s_front, s_back = bay_span_from_back_offset(
        inp.L_cyl_in, inp.back_offset_from_back_cone_in, inp.motor_bay_length_in
    )
    L_bay_in = max(0.0, s_back - s_front)

    # --- choose the draft once (precedence: fully_submerged > solve_equilibrium > given) ---
    if inp.fully_submerged:
        draft_used_in = 2.0 * inp.R_out_in
    elif inp.solve_equilibrium:
        if inp.W_total_lbf is None or inp.W_total_lbf <= 0:
            raise ValueError("Set W_total_lbf > 0 when solve_equilibrium=True.")
        draft_used_in = solve_draft_no_foam(inp.W_total_lbf, inp.R_out_in, L_bay_in, inp.gamma)
    else:
        draft_used_in = inp.draft_in

    # --- hydro @ chosen draft ---
    V_ft3 = displaced_volume_from_draft_ft3(inp.R_out_in, draft_used_in, L_bay_in)
    B_lbf = inp.gamma * V_ft3

    # longitudinal CB_x from nose: nose→front cone + bay midpoint in cylinder
    x_front_bay_from_nose = inp.L_front_in + s_front
    x_back_bay_from_nose  = inp.L_front_in + s_back
    CBx_in = 0.5 * (x_front_bay_from_nose + x_back_bay_from_nose)

    # vertical CB via circular-segment centroid (depth below WL positive)
    ybar_in = seg_centroid_below_WL_in(draft_used_in, inp.R_out_in)
    z_CB_from_bottom_in = draft_used_in - ybar_in
    z_CB_below_WL_in = ybar_in

    return {
        "draft_used_in": draft_used_in,
        "L_bay_in": L_bay_in,
        "submerged_area_in2": seg_area_in2(draft_used_in, inp.R_out_in),
        "submerged_volume_ft3": V_ft3,
        "buoyant_force_lbf": B_lbf,
        "CBx_in": CBx_in,
        "z_CB_from_bottom_in": z_CB_from_bottom_in,
        "z_CB_below_WL_in": z_CB_below_WL_in,
        "s_front_in": s_front,
        "s_back_in": s_back,
    }

# ----------------------------
# PRETTY PRINT
# ----------------------------
def fmt(x, nd=4):
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"

def print_block(inp: Inputs, r: dict, title="CB (Surface Mode) — Motor Bay Only"):
    print(f"=== {title} ===")
    print("[Inputs]")
    if inp.fully_submerged:
        print("Mode: fully submerged (draft = 2R)")
    elif inp.solve_equilibrium:
        print(f"Equilibrium draft solved for W_total (lbf): {fmt(inp.W_total_lbf,3)}")
    print(f"Draft used (in): {fmt(r['draft_used_in'],3)}")
    print("")
    print("[Hydro @ Draft]")
    print(f"  Submerged area A (in^2):       {fmt(r['submerged_area_in2'],3)}")
    print(f"  Submerged volume (ft^3):       {fmt(r['submerged_volume_ft3'],3)}")
    print(f"  Buoyant force B (lbf):         {fmt(r['buoyant_force_lbf'],3)}")
    print("")
    print("[Center of Buoyancy]")
    print(f"  CB_x from nose (in):           {fmt(r['CBx_in'],3)}")
    print(f"  CB depth below WL (in):        {fmt(r['z_CB_below_WL_in'],3)}")
    print(f"  CB height above bottom (in):   {fmt(r['z_CB_from_bottom_in'],3)}")
    print("")


# ----------------------------
# HULL PROFILE (FOR DRAWING)
# ----------------------------
@dataclass(frozen=True)
class HullDims:
    R: float = 12.0
    Lf: float = 29.5
    Lc: float = 182.0
    Lb: float = 31.5
    r_tip: float = 3.5
    @property
    def x_front_end(self):
        return self.Lf
    @property
    def x_cyl_end(self):
        return self.Lf + self.Lc
    @property
    def x_back_end(self):
        return self.Lf + self.Lc + self.Lb

def smooth_cos_01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)

SMOOTH_LEN_FRONT = 27.0  # inches of smoothing near the front join

def r_front(x: float, p: HullDims) -> float:
    Ls = max(0.0, min(SMOOTH_LEN_FRONT, p.Lf))
    x0 = p.Lf - Ls
    if x <= 0.0:
        return 0.0
    if x >= p.Lf:
        return p.R
    r_par = p.R * math.sqrt(max(0.0, x / p.Lf))
    if x < x0:
        return r_par
    r0 = p.R * math.sqrt(x0 / p.Lf)
    drdx0 = p.R / (2.0 * math.sqrt(max(1e-12, p.Lf * x0)))
    r1 = p.R
    t = (x - x0) / max(1e-12, Ls)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * r0 + h10 * (Ls * drdx0) + h01 * r1 + h11 * (Ls * 0.0)

def r_back(x: float, p: HullDims) -> float:
    xi = x - p.x_cyl_end
    if xi <= 0.0:
        return p.R
    if xi >= p.Lb:
        return p.r_tip
    a = p.Lb / (1.0 - (p.r_tip / p.R) ** 2)
    r_par = p.R * math.sqrt(max(0.0, 1.0 - xi / a))
    s = smooth_cos_01(xi / p.Lb)
    r_val = p.R - s * (p.R - r_par)
    # snap very close to the tip to avoid FP jitter
    if p.Lb - xi < 1e-9:
        r_val = p.r_tip
    return r_val

def r_cyl(x: float, p: HullDims) -> float:
    return p.R

def r_hull(x: float, p: HullDims) -> float:
    if x < 0.0:
        return 0.0
    if x <= p.x_front_end:
        return r_front(x, p)
    if x <= p.x_cyl_end:
        return r_cyl(x, p)
    if x <= p.x_back_end:
        return r_back(x, p)
    return 0.0

# ----------------------------
# PLOTTER
# ----------------------------
def plot_hull_with_cb(inp: Inputs, r_cb: dict, title="TEX Hull Profile"):
    p = HullDims(R=inp.R_out_in, Lf=inp.L_front_in, Lc=inp.L_cyl_in, Lb=inp.L_back_in)
    x = np.linspace(0, p.x_back_end, 1200)
    r = np.array([r_hull(xi, p) for xi in x])

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, r, color="#0052CC", lw=2.2, label="Outer Hull",
             solid_capstyle="round", solid_joinstyle="round")
    plt.plot(x, -r, color="#0052CC", lw=2.2,
             solid_capstyle="round", solid_joinstyle="round")

    # --- Section markers ---
    r_front_join = r_hull(p.x_front_end, p)
    r_back_join = r_hull(p.x_cyl_end, p)
    r_stern = r_hull(p.x_back_end, p)
    plt.vlines(p.x_front_end, -r_front_join, r_front_join, linestyle='--', color="#666666")
    plt.vlines(p.x_cyl_end, -r_back_join, r_back_join, linestyle='--', color="#666666")
    plt.vlines(p.x_back_end, -r_stern, r_stern, lw=2.2, color="#0052CC")

    # --- Waterline height and CB ---
    d_used = r_cb["draft_used_in"]
    y_wl = d_used - p.R               # height of WL above hull centerline
    plt.axhline(y_wl, linestyle='--', color='#888888', lw=1.1, label="Waterline")

    x_cb = r_cb["CBx_in"]
    y_cb = y_wl - r_cb["z_CB_below_WL_in"]
    plt.scatter([x_cb], [y_cb], s=70, color="#d62728", zorder=5, label=f'CB ({d_used:.3g}" draft)')
    plt.annotate(f'CB ({d_used:.3g}")',
                 xy=(x_cb, y_cb),
                 xytext=(x_cb + 10, y_cb - 2),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color="#d62728"),
                 fontsize=10, color="#d62728", ha='left', va='center')

    # --- Section labels ---
    y_label_offset = -1.55 * p.R
    plt.text(p.Lf / 2, y_label_offset, "Front Cone",
             ha='center', va='top', fontsize=10, fontweight='bold', color="#1f77b4",
             bbox=dict(facecolor='white', edgecolor='#1f77b4', boxstyle='round,pad=0.3', lw=1.0))
    plt.text(p.Lf + p.Lc / 2, y_label_offset, "Cylinder",
             ha='center', va='top', fontsize=10, fontweight='bold', color="#2ca02c",
             bbox=dict(facecolor='white', edgecolor='#2ca02c', boxstyle='round,pad=0.3', lw=1.0))
    plt.text(p.Lf + p.Lc + p.Lb / 2, y_label_offset, "Back Cone",
             ha='center', va='top', fontsize=10, fontweight='bold', color="#d62728",
             bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3', lw=1.0))

    # --- Symmetric view with margin (so WL is visibly above hull) ---
    ax = plt.gca()
    margin = 0.25 * p.R
    ax.set_ylim(-(p.R + margin), +(p.R + margin))
    ax.set_aspect('equal', 'box')

    # --- Labels and legend ---
    plt.xlabel('x [in] (nose → stern)', labelpad=15)
    plt.ylabel('r [in]')
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.7, -0.25), ncol=1, frameon=True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# ONE-CALL HELPER
# ----------------------------
def draw_cb(draft_in: float = None, W_total_lbf: float = None, fully_submerged: bool = False):
    inp = Inputs()
    inp.fully_submerged = bool(fully_submerged)
    if inp.fully_submerged:
        inp.solve_equilibrium = False  # ignore solver when submerged
    elif W_total_lbf is not None:
        inp.solve_equilibrium = True
        inp.W_total_lbf = float(W_total_lbf)
    else:
        if draft_in is None:
            raise ValueError("Provide draft_in, W_total_lbf, or set fully_submerged=True.")
        inp.solve_equilibrium = False
        inp.draft_in = float(draft_in)

    result = compute_cb_surface(inp)
    inp.draft_in = result["draft_used_in"]
    title = "TEX Hull Profile — CB (Fully Submerged)" if inp.fully_submerged \
            else ("TEX Hull Profile — CB (Solved Draft)" if inp.solve_equilibrium
                  else "TEX Hull Profile — CB @ Given Draft")
    plot_hull_with_cb(inp, result, title=title)
    print_block(inp, result, title=("CB @ Fully Submerged (Motor Bay Only)" if inp.fully_submerged
                                    else "CB @ Solved Draft (Motor Bay Only)" if inp.solve_equilibrium
                                    else "CB @ Given Draft (Motor Bay Only)"))


# ----------------------------
# RUNS cccc
# ----------------------------
if __name__ == "__main__":
    draw_cb(fully_submerged=True)