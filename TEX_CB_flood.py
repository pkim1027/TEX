# ---------------------------------------------
# TEX_CB_surface_motorbay.py
# Center of Buoyancy (SURFACE MODE) for sealed motor bay only.
# Free-flooding sections are neutral; no foam considered.
#
# NEW:
#  - Explicit "submerged volume" of the sealed motor bay at the specified draft
#  - Optional solver to find equilibrium draft for a given total weight W_total_lbf
# ---------------------------------------------

import math
from dataclasses import dataclass
from typing import Optional

# ----------------------------
# Segment geometry (circular segment)
# ----------------------------
def seg_area_in2(d_in: float, R_in: float) -> float:
    """Submerged circular-segment area (in^2) for radius R_in (in), draft d_in (in) measured up from bottom."""
    d = max(0.0, min(d_in, 2.0 * R_in))
    if d == 0.0:
        return 0.0
    if abs(d - 2.0 * R_in) < 1e-12:
        return math.pi * R_in * R_in
    return (R_in**2) * math.acos((R_in - d) / R_in) - (R_in - d) * math.sqrt(max(0.0, 2 * R_in * d - d * d))

def seg_theta(d_in: float, R_in: float) -> float:
    return 2.0 * math.acos((R_in - d_in) / R_in) if R_in > 0 else 0.0

def seg_centroid_below_WL_in(d_in: float, R_in: float) -> float:
    """Centroid depth (in) measured downward from WL to the segment centroid."""
    if d_in <= 0:
        return 0.0
    if d_in >= 2.0 * R_in:
        return 4.0 * R_in / 3.0
    theta = seg_theta(d_in, R_in)
    denom = (theta - math.sin(theta))
    if abs(denom) < 1e-14:
        return 0.0
    return (4.0 * R_in * (math.sin(theta / 2.0) ** 3)) / (3.0 * denom)

# ----------------------------
# Inputs
# ----------------------------
@dataclass
class Inputs:
    # Hull
    R_out_in: float = 12.0       # outer radius (in) for a 24" OD tube
    t_wall_in: float = 0.25      # wall thickness (info only; not needed for surface-mode buoyancy)
    L_front_in: float = 29.5     # front cone length (in)
    L_cyl_in: float = 182.0      # straight cylinder length (in)
    L_back_in: float = 31.5      # back cone length (in)

    # Hydro
    draft_in: float = 12.0       # outside draft at motor-bay section (in)
    gamma: float = 62.4          # lbf/ft^3 (fresh water)

    # Motor bay placement
    back_offset_from_back_cone_in: float = 61.5  # distance from start of back cone to bay BACK (in)
    motor_bay_length_in: float = 50.0            # sealed length (in)

    # Optional: use this (with solve_equilibrium=True) to compute the no-foam draft for total weight
    W_total_lbf: Optional[float] = None
    solve_equilibrium: bool = False              # if True, solve for draft such that B(d)=W_total_lbf

# ----------------------------
# Core computations
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
    A_in2 = seg_area_in2(draft_in, R_out_in)      # submerged cross-sectional area (in^2)
    V_in3 = A_in2 * max(0.0, L_bay_in)            # in^3
    return V_in3 / 1728.0                         # ft^3

def buoyancy_lbf_from_draft(R_out_in: float, draft_in: float, L_bay_in: float, gamma: float) -> float:
    return gamma * displaced_volume_from_draft_ft3(R_out_in, draft_in, L_bay_in)

def solve_draft_no_foam(W_lbf: float, R_out_in: float, L_bay_in: float, gamma: float,
                        d_lo_in: float = 0.0, d_hi_in: float = None, iters: int = 60) -> float:
    """
    Solve for draft (inches) so that buoyancy equals W_lbf for the sealed bay only.
    Uses bisection on d in [0, 2R].
    """
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
    # Determine bay front/back along the cylinder from the back offset & length
    s_front, s_back = bay_span_from_back_offset(inp.L_cyl_in, inp.back_offset_from_back_cone_in, inp.motor_bay_length_in)
    L_bay_in = max(0.0, s_back - s_front)

    # If requested, solve for equilibrium draft so that B(d) = W_total_lbf
    draft_used_in = inp.draft_in
    if inp.solve_equilibrium:
        if inp.W_total_lbf is None or inp.W_total_lbf <= 0:
            raise ValueError("Set W_total_lbf > 0 when solve_equilibrium=True.")
        draft_used_in = solve_draft_no_foam(inp.W_total_lbf, inp.R_out_in, L_bay_in, inp.gamma)

    # Submerged volume and buoyancy at the (given or solved) draft
    V_ft3 = displaced_volume_from_draft_ft3(inp.R_out_in, draft_used_in, L_bay_in)  # <-- submerged volume
    B_lbf = inp.gamma * V_ft3

    # Longitudinal CB_x from nose: nose→front cone + bay midpoint in cylinder
    x_front_bay_from_nose = inp.L_front_in + s_front
    x_back_bay_from_nose  = inp.L_front_in + s_back
    CBx_in = 0.5 * (x_front_bay_from_nose + x_back_bay_from_nose)

    # Vertical CB: centroid of the segment below WL
    ybar_in = seg_centroid_below_WL_in(draft_used_in, inp.R_out_in)  # depth below WL (in)
    z_CB_from_bottom_in = draft_used_in - ybar_in
    z_CB_below_WL_in = ybar_in

    return {
        "draft_used_in": draft_used_in,
        "L_bay_in": L_bay_in,
        "submerged_area_in2": seg_area_in2(draft_used_in, inp.R_out_in),
        "submerged_volume_ft3": V_ft3,       # <-- explicit
        "buoyant_force_lbf": B_lbf,
        "CBx_in": CBx_in,
        "z_CB_from_bottom_in": z_CB_from_bottom_in,
        "z_CB_below_WL_in": z_CB_below_WL_in,
        "s_front_in": s_front,
        "s_back_in": s_back,
    }

# ----------------------------
# Pretty print
# ----------------------------
def fmt(x, nd=4):
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"

def print_block(inp: Inputs, r: dict, title="CB (Surface Mode) — Motor Bay Only"):
    print(f"=== {title} ===")
    print("[Inputs]")
    if inp.solve_equilibrium:
        print(f"Equilibrium draft solved for W_total (lbf): {fmt(inp.W_total_lbf,3)}")
    print(f"Draft used (in): {fmt(r['draft_used_in'],3)} | gamma (lbf/ft^3): {fmt(inp.gamma,3)}")
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
# Default Run
# ----------------------------
if __name__ == "__main__":
    base = Inputs()
    base.solve_equilibrium = False
    base.draft_in = 12.0
    res = compute_cb_surface(base)
    print_block(base, res, title="CB @ Given Draft (Motor Bay Only)")

    base.solve_equilibrium = False
    base.draft_in = 24.0
    res = compute_cb_surface(base)
    print_block(base, res, title="CB @ Given Draft (Motor Bay Only)")

    # Equilibrium draft for a total weight:
    base.solve_equilibrium = True
    base.W_total_lbf = 1000.0
    res = compute_cb_surface(base)
    print_block(base, res, title="CB @ Solved Draft for W_total (Motor Bay Only)")
