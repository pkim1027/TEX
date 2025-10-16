# ---------------------------------------------
# TEX_max_balls_freeflood.py
# Max count of 1.5" balls in free-flooding sections (cones + cylinder minus motor bay)
# ---------------------------------------------

import math
from dataclasses import dataclass

@dataclass
class Inputs:
    # Geometry (inches)
    R_out_in: float = 12.0          # outer radius (24" OD -> 12")
    t_wall_in: float = 0.25         # wall thickness
    L_front_in: float = 29.5        # front cone length
    L_cyl_in: float   = 182.0       # straight cylinder length
    L_back_in: float  = 31.5        # back cone length
    r_back_tip_out_in: float = 3.5  # back cone tip *outer* radius (0 = sharp tip)

    # Motor bay (sealed) inside the cylinder (inches from cylinder start)
    motor_bay_front_in: float = 40.0
    motor_bay_back_in: float  = 88.0

    # Ball & packing
    ball_d_in: float = 1.5          # ball diameter (inches)
    eta_packing: float = 0.60       # packing efficiency (0.60--0.64 typical for spheres)

    # Integration resolution for cones
    N_slices: int = 4000

# ---- Radius profiles ----
def R_out_front(x_in: float, L_front_in: float, R_base_out_in: float) -> float:
    """0..Lf, linear from 0 to R_base_out"""
    if L_front_in <= 0: return 0.0
    return (R_base_out_in / L_front_in) * max(0.0, min(x_in, L_front_in))

def R_out_back(x_in: float, L_back_in: float, R_base_out_in: float, r_tip_out_in: float) -> float:
    """0..Lb, linear from R_base_out to r_tip_out"""
    if L_back_in <= 0: return 0.0
    t = max(0.0, min(x_in, L_back_in)) / L_back_in
    return (1 - t) * R_base_out_in + t * max(0.0, r_tip_out_in)

def R_in_from_R_out(R_out_in: float, t_wall_in: float) -> float:
    return max(0.0, R_out_in - t_wall_in)

# ---- Volumes ----
def cone_interior_volume_ft3(L_in: float, R_out_fn, R_out_args: tuple, t_wall_in: float, N: int) -> float:
    """Integrate interior volume of a cone-like section by slicing its varying inner radius."""
    if L_in <= 0: return 0.0
    in_to_ft = 1/12.0
    dx_in = L_in / N
    V_ft3 = 0.0
    for i in range(N):
        x = (i + 0.5) * dx_in
        R_out_local_in = R_out_fn(x, *R_out_args)
        R_in_local_in  = R_in_from_R_out(R_out_local_in, t_wall_in)
        if R_in_local_in <= 0.0:
            continue
        A_in_local_ft2 = math.pi * ( (R_in_local_in*in_to_ft)**2 )
        V_ft3 += A_in_local_ft2 * (dx_in*in_to_ft)
    return V_ft3

def cylinder_interior_volume_ft3(L_in: float, R_out_in: float, t_wall_in: float, include_segments_in: list) -> float:
    """Interior volume of cylinder segments (subtract sealed motor bay). include_segments_in: list of (start_in, end_in) to include."""
    in_to_ft = 1/12.0
    R_in_ft = R_in_from_R_out(R_out_in, t_wall_in) * in_to_ft
    A_in_ft2 = math.pi * (R_in_ft**2)
    total_len_in = sum(max(0.0, end - start) for (start, end) in include_segments_in)
    return A_in_ft2 * (total_len_in * in_to_ft)

# ---- Ball capacity ----
def ball_volume_ft3(ball_d_in: float) -> float:
    in_to_ft = 1/12.0
    r_ft = 0.5 * ball_d_in * in_to_ft
    return (4.0/3.0) * math.pi * (r_ft**3)

def max_balls_capacity(inputs: Inputs):
    # Front cone
    V_front = cone_interior_volume_ft3(
        inputs.L_front_in,
        R_out_front, (inputs.L_front_in, inputs.R_out_in),
        inputs.t_wall_in, inputs.N_slices
    )

    # Cylinder minus motor bay
    mb_f, mb_b = inputs.motor_bay_front_in, inputs.motor_bay_back_in
    mb_f = max(0.0, min(mb_f, inputs.L_cyl_in))
    mb_b = max(0.0, min(mb_b, inputs.L_cyl_in))
    if mb_b < mb_f:
        mb_f, mb_b = mb_b, mb_f  # swap if out of order
    cyl_include = [(0.0, mb_f), (mb_b, inputs.L_cyl_in)]
    V_cyl = cylinder_interior_volume_ft3(inputs.L_cyl_in, inputs.R_out_in, inputs.t_wall_in, cyl_include)

    # Back cone
    V_back = cone_interior_volume_ft3(
        inputs.L_back_in,
        R_out_back, (inputs.L_back_in, inputs.R_out_in, inputs.r_back_tip_out_in),
        inputs.t_wall_in, inputs.N_slices
    )

    V_free_ft3 = V_front + V_cyl + V_back
    V_ball = ball_volume_ft3(inputs.ball_d_in)

    # Packed capacity
    n_max = math.floor( (inputs.eta_packing * V_free_ft3) / V_ball ) if V_ball > 0 else 0

    return {
        "V_front_ft3": V_front,
        "V_cyl_ft3": V_cyl,
        "V_back_ft3": V_back,
        "V_free_total_ft3": V_free_ft3,
        "ball_V_ft3": V_ball,
        "eta_packing": inputs.eta_packing,
        "n_max_balls": n_max
    }

def fmt(x, nd=4):
    if isinstance(x, int): return str(x)
    return f"{x:.{nd}f}"

if __name__ == "__main__":
    inp = Inputs()
    res = max_balls_capacity(inp)
    print("=== Max 1.5\" Balls Capacity in Free-Flooding Sections ===")
    print(f"Ball diameter (in):                 {fmt(inp.ball_d_in,3)}")
    print(f"Packing efficiency η:               {fmt(inp.eta_packing,3)}")
    print("---- Volumes (ft^3)")
    print(f"TOTAL free-flood interior:          {fmt(res['V_free_total_ft3'],3)}")
    print("---- Capacity")
    print(f"Single ball volume (ft^3):          {fmt(res['ball_V_ft3'],6)}")
    print(f"Max balls (packed):                 {res['n_max_balls']}")
