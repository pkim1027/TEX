import math
from dataclasses import dataclass

# ----------------------------
# Inputs
# ----------------------------
@dataclass
class Inputs:
    # Hydro target
    W_total_lbf: float = 1000      # total vessel weight (lbf)
    draft_in: float = 12.0            # target outside draft at motor bay (inches from outside bottom)
    gamma: float = 62.4               # lbf/ft^3 (fresh water)

    # Global geometry
    R_out_in: float = 12.0            # outer radius (in)
    t_wall_in: float = 0.25           # wall thickness (in)

    # Section lengths (inches)
    L_front_in: float = 29.5          # front cone length (in)
    L_cyl_in: float   = 182.0         # straight-cylinder length (in)
    L_back_in: float  = 31.5          # back cone length (in)

    # Back cone cut (outer tip radius; set 0 if true tip)
    r_back_tip_out_in: float = 3.5    # in; inner tip becomes max(0, r_out - t_wall)

    # Sealed motor bay inside the cylinder
    motor_bay_front_in: float = 40.0  # in from cylinder start
    motor_bay_back_in: float  = 90.0  # in from cylinder start

    # Foam balls
    ball_d_in: float = 1.5            # diameter (in)
    ball_weight_g: float = 0.6        # grams
    eta_packing: float = 0.60         # packing efficiency of spheres

    # Integration resolution
    N_slices: int = 20000              # longitudinal slices for cones+cyl (flood region)

    # Overfill tolerance (% of above-WL capacity)
    overfill_tolerance_pct: float = 10.0


# ----------------------------
# Helper geometry
# ----------------------------
def seg_area(d_ft: float, R_ft: float) -> float:
    """
    Submerged circular-segment area (ft^2) for a circle radius R_ft
    with waterline draft d_ft measured from circle bottom (0..2R).
    """
    if R_ft <= 0.0:
        return 0.0
    d_ft = max(0.0, min(d_ft, 2.0*R_ft))
    if d_ft == 0.0:
        return 0.0
    if abs(d_ft - 2.0*R_ft) < 1e-12:
        return math.pi * R_ft**2
    return (R_ft**2)*math.acos((R_ft - d_ft)/R_ft) - (R_ft - d_ft)*math.sqrt(max(0.0, 2*R_ft*d_ft - d_ft**2))

def bay_buoyancy_at_draft(R_out_in: float, draft_in: float, L_bay_in: float, gamma: float) -> float:
    """ Natural buoyancy from sealed motor bay at given draft (no solving). """
    in_to_ft = 1/12.0
    R_ft = R_out_in * in_to_ft
    d_ft = draft_in * in_to_ft
    A = seg_area(d_ft, R_ft)
    L_bay_ft = L_bay_in * in_to_ft
    return gamma * A * L_bay_ft, A

def grams_to_lbf(g: float) -> float:
    return g * 0.00220462

def ball_properties(ball_d_in: float, ball_weight_g: float, gamma: float):
    in_to_ft = 1/12.0
    d_ft = ball_d_in * in_to_ft
    r_ft = 0.5 * d_ft
    V_ball = (4.0/3.0)*math.pi*(r_ft**3)     # ft^3
    W_ball_lbf = grams_to_lbf(ball_weight_g) # lbf
    b_ball = gamma*V_ball - W_ball_lbf       # net buoyancy per ball
    return V_ball, b_ball, W_ball_lbf

# ----------------------------
# Radius profiles (outer -> inner)
# ----------------------------
def R_out_front(x_in: float, L_front_in: float, R_base_out_in: float) -> float:
    """ Linear from tip (0) to base (R_base_out) over 0..L_front """
    if L_front_in <= 0: return 0.0
    return (R_base_out_in/L_front_in) * max(0.0, min(x_in, L_front_in))

def R_out_back(x_in: float, L_back_in: float, R_base_out_in: float, r_tip_out_in: float) -> float:
    """ Linear from base (R_base_out) to tip (r_tip_out) over 0..L_back """
    if L_back_in <= 0: return 0.0
    t = max(0.0, min(x_in, L_back_in)) / L_back_in
    return (1 - t) * R_base_out_in + t * max(0.0, r_tip_out_in)

def R_in_from_R_out(R_out_in: float, t_wall_in: float) -> float:
    return max(0.0, R_out_in - t_wall_in)

# ----------------------------
# Free-flooding volume integration (above/below WL), excluding motor bay
# ----------------------------
def integrate_free_flood_volumes_above_below(inp: Inputs):
    """
    Integrate interior areas above/below WL across:
      front cone [0, Lf], cylinder [0, Lc] EXCLUDING motor bay segment, back cone [0, Lb].
    Uses interior radius at each x and inside draft d_inside_in = max(0, draft_in - t_wall_in).
    """
    in_to_ft = 1/12.0
    d_inside_in = max(0.0, inp.draft_in - inp.t_wall_in)
    d_inside_ft = d_inside_in * in_to_ft

    # FRONT CONE
    V_above_front = 0.0
    V_below_front = 0.0
    if inp.L_front_in > 0:
        dx_in = inp.L_front_in / inp.N_slices
        for i in range(inp.N_slices):
            x = (i + 0.5) * dx_in
            R_out_local_in = R_out_front(x, inp.L_front_in, inp.R_out_in)
            R_in_local_in  = R_in_from_R_out(R_out_local_in, inp.t_wall_in)
            R_in_local_ft  = R_in_local_in * in_to_ft
            if R_in_local_ft <= 0.0:
                continue
            A_sub = seg_area(min(d_inside_ft, 2*R_in_local_ft), R_in_local_ft)
            A_full = math.pi * (R_in_local_ft**2)
            A_abv  = max(0.0, A_full - A_sub)
            V_below_front += A_sub * (dx_in*in_to_ft)
            V_above_front += A_abv * (dx_in*in_to_ft)

    # CYLINDER (excluding motor bay)
    V_above_cyl = 0.0
    V_below_cyl = 0.0
    if inp.L_cyl_in > 0:
        R_in_cyl_ft = R_in_from_R_out(inp.R_out_in, inp.t_wall_in) * in_to_ft
        A_sub_cyl = seg_area(min(d_inside_ft, 2*R_in_cyl_ft), R_in_cyl_ft)
        A_full_cyl = math.pi * (R_in_cyl_ft**2)
        A_abv_cyl  = max(0.0, A_full_cyl - A_sub_cyl)

        # Cylinder segments: [0, L_cyl] minus [motor_front, motor_back]
        L1 = max(0.0, min(inp.motor_bay_front_in, inp.L_cyl_in))                 # before bay
        L2 = max(0.0, min(inp.L_cyl_in, inp.L_cyl_in) - max(inp.motor_bay_back_in, 0.0))  # after bay
        V_below_cyl = (A_sub_cyl * (L1 + L2) * in_to_ft)
        V_above_cyl = (A_abv_cyl * (L1 + L2) * in_to_ft)

    # BACK CONE
    V_above_back = 0.0
    V_below_back = 0.0
    if inp.L_back_in > 0:
        dx_in = inp.L_back_in / inp.N_slices
        for i in range(inp.N_slices):
            x = (i + 0.5) * dx_in
            R_out_local_in = R_out_back(x, inp.L_back_in, inp.R_out_in, inp.r_back_tip_out_in)
            R_in_local_in  = R_in_from_R_out(R_out_local_in, inp.t_wall_in)
            R_in_local_ft  = R_in_local_in * in_to_ft
            if R_in_local_ft <= 0.0:
                continue
            A_sub = seg_area(min(d_inside_ft, 2*R_in_local_ft), R_in_local_ft)
            A_full = math.pi * (R_in_local_ft**2)
            A_abv  = max(0.0, A_full - A_sub)
            V_below_back += A_sub * (dx_in*in_to_ft)
            V_above_back += A_abv * (dx_in*in_to_ft)

    V_above_total = V_above_front + V_above_cyl + V_above_back
    V_below_total = V_below_front + V_below_cyl + V_below_back
    return V_above_total, V_below_total, d_inside_in

# ----------------------------
# Main computation
# ----------------------------
def compute_with_cones(inp: Inputs):
    # Sealed motor bay buoyancy at target draft (for shortfall)
    L_bay_in = max(0.0, inp.motor_bay_back_in - inp.motor_bay_front_in)  # inches
    B_bay, _ = bay_buoyancy_at_draft(inp.R_out_in, inp.draft_in, L_bay_in, inp.gamma)
    deltaB = max(0.0, inp.W_total_lbf - B_bay)

    # Balls needed for buoyancy shortfall
    V_ball, b_ball, W_ball = ball_properties(inp.ball_d_in, inp.ball_weight_g, inp.gamma)
    if b_ball <= 0:
        raise ValueError("Ball net buoyancy <= 0; check ball size/weight/density.")
    n_for_buoy = math.ceil(deltaB / b_ball) if deltaB > 0 else 0

    # Integrate free-flood volumes ABOVE/BELOW WL incl. cones, excluding motor bay
    V_above, V_below, d_inside_in = integrate_free_flood_volumes_above_below(inp)

    # Foam volumes
    V_foam_solid = n_for_buoy * V_ball
    V_foam_packed = V_foam_solid / inp.eta_packing if inp.eta_packing > 0 else float('inf')

    # Jam check vs above-WL capacity
    jam_threshold_ft3 = V_above
    jam_deficit_ft3 = max(0.0, jam_threshold_ft3 - V_foam_packed)
    jam_overfill_ft3 = max(0.0, V_foam_packed - jam_threshold_ft3)

    jam_status = "meets"
    if V_foam_packed + 1e-9 < jam_threshold_ft3:
        jam_status = "insufficient"
    elif V_foam_packed > jam_threshold_ft3 * (1 + inp.overfill_tolerance_pct/100.0):
        jam_status = "overfill_excess"

    # Extra balls to reach jam threshold if insufficient
    n_more_to_jam = 0
    if jam_deficit_ft3 > 0 and V_ball > 0:
        n_more_to_jam = math.ceil((jam_deficit_ft3 * inp.eta_packing) / V_ball)

    # Balls to remove to be within tolerance if overfilled
    n_remove_to_within_tolerance = 0
    if jam_status == "overfill_excess" and V_ball > 0:
        target_max_packed = jam_threshold_ft3 * (1 + inp.overfill_tolerance_pct/100.0)
        excess_packed = max(0.0, V_foam_packed - target_max_packed)
        n_remove_to_within_tolerance = math.ceil((excess_packed * inp.eta_packing) / V_ball)

    return {
        "inputs": inp,
        "B_bay_lbf": B_bay,
        "deltaB_lbf": deltaB,
        "ball_V_ft3": V_ball,
        "ball_net_buoy_lbf": b_ball,
        "ball_weight_lbf": W_ball,
        "n_balls_for_buoy": n_for_buoy,
        "V_above_WL_ft3": V_above,
        "V_below_WL_ft3": V_below,
        "V_foam_solid_ft3": V_foam_solid,
        "V_foam_packed_ft3": V_foam_packed,
        "jam_threshold_ft3": jam_threshold_ft3,
        "jam_status": jam_status,
        "jam_deficit_ft3": jam_deficit_ft3,
        "jam_overfill_ft3": jam_overfill_ft3,
        "n_more_to_jam": n_more_to_jam,
        "n_remove_to_within_tolerance": n_remove_to_within_tolerance,
        "d_inside_in": d_inside_in,
        "L_bay_in": L_bay_in,
    }

# ----------------------------
# Pretty print
# ----------------------------
def fmt(x, nd=4):
    if isinstance(x, int): return str(x)
    return f"{x:.{nd}f}"

def print_report(r: dict):
    inp = r["inputs"]
    print("=== Above-WL Capacity with Cones (Jam Check) ===")
    print(f"Total weight (lbf):                     {fmt(inp.W_total_lbf,3)}")
    print(f"Target outside draft (in):              {fmt(inp.draft_in,3)}   (inside draft: {fmt(r['d_inside_in'],3)} in)")
    print(f"Ball Ø (in), mass (g), η:               {fmt(inp.ball_d_in,3)}, {fmt(inp.ball_weight_g,3)}, {fmt(inp.eta_packing,3)}")
    print("---- Hydro & Need")
    print(f"Motor-bay buoyancy at target (lbf):     {fmt(r['B_bay_lbf'],3)}")
    print(f"Buoyancy shortfall ΔB (lbf):            {fmt(r['deltaB_lbf'],3)}")
    print(f"Balls to meet ΔB (count):               {r['n_balls_for_buoy']}")
    print("---- Volumes (ft^3)")
    print(f"Interior BELOW WL (ft^3):               {fmt(r['V_below_WL_ft3'],3)}")
    print(f"Interior ABOVE WL (ft^3):               {fmt(r['V_above_WL_ft3'],3)}")
    print(f"Foam packed envelope (ft^3):            {fmt(r['V_foam_packed_ft3'],3)}")
    print("---- Jam Evaluation")
    print(f"Jam status:                             {r['jam_status']}")
    print(f"Deficit vs jam (ft^3):                  {fmt(r['jam_deficit_ft3'],3)}")
    print(f"Overfill vs jam (ft^3):                 {fmt(r['jam_overfill_ft3'],3)}")
    print(f"Extra balls needed to jam (count):      {r['n_more_to_jam']}")
    print(f"Balls to remove (within tol) (count):   {r['n_remove_to_within_tolerance']}")

# ----------------------------
# Run with defaults
# ----------------------------
if __name__ == "__main__":
    inp = Inputs()
    res = compute_with_cones(inp)
    print_report(res)
