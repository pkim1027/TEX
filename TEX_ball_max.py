# -------------------------------------------------
# TEX_max_balls_freeflood_Hermite.py
# Max count of 1.5" balls in free-flooding sections (Hermite + cosine hull form)
# -------------------------------------------------

import math
from dataclasses import dataclass, field
import numpy as np

# ----------------------------
# Hull geometry (Hermite front, cosine back)
# ----------------------------
@dataclass
class HullDims:
    R: float = 12.0
    Lf: float = 29.5
    Lc: float = 182.0
    Lb: float = 31.5
    r_tip: float = 3.5
    t_wall_in: float = 0.25
    @property
    def x_front_end(self): return self.Lf
    @property
    def x_cyl_end(self):   return self.Lf + self.Lc
    @property
    def x_back_end(self):  return self.Lf + self.Lc + self.Lb

# smooth cosine transition helper
def smooth_cos_01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)

# Hermite-smoothed front radius
SMOOTH_LEN_FRONT = 27.0
def r_front(x: float, p: HullDims) -> float:
    Ls = max(0.0, min(SMOOTH_LEN_FRONT, p.Lf))
    x0 = p.Lf - Ls
    if x <= 0.0: return 0.0
    if x >= p.Lf: return p.R
    r_par = p.R * math.sqrt(max(0.0, x / p.Lf))
    if x < x0: return r_par
    r0 = p.R * math.sqrt(x0 / p.Lf)
    drdx0 = p.R / (2.0 * math.sqrt(max(1e-12, p.Lf * x0)))
    t = (x - x0) / max(1e-12, Ls)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*r0 + h10*(Ls*drdx0) + h01*p.R + h11*(Ls*0.0)

# Cosine-blended back cone
def r_back(x: float, p: HullDims) -> float:
    xi = x - p.x_cyl_end
    if xi <= 0.0: return p.R
    if xi >= p.Lb: return p.r_tip
    a = p.Lb / (1.0 - (p.r_tip / p.R)**2)
    r_par = p.R * math.sqrt(max(0.0, 1.0 - xi / a))
    s = smooth_cos_01(xi / p.Lb)
    r_val = p.R - s * (p.R - r_par)
    if p.Lb - xi < 1e-9: r_val = p.r_tip
    return r_val

def r_cyl(x: float, p: HullDims) -> float:
    return p.R

def r_hull(x: float, p: HullDims) -> float:
    if x < 0.0: return 0.0
    if x <= p.x_front_end: return r_front(x, p)
    if x <= p.x_cyl_end:   return r_cyl(x, p)
    if x <= p.x_back_end:  return r_back(x, p)
    return 0.0

# ----------------------------
# Ball + geometry parameters
# ----------------------------
@dataclass
class Inputs:
    ball_d_in: float = 1.5
    eta_packing: float = 0.60
    N_slices: int = 20000
    motor_bay_front_in: float = 40.0
    motor_bay_back_in: float  = 88.0
    p: HullDims = field(default_factory=HullDims)

# ----------------------------
# Volume integrators
# ----------------------------
def section_volume_interior_ft3(p: HullDims, N: int = 20000):
    """
    Integrates the entire interior free-flooding volume of the hull
    (front cone + cylinder + back cone) using the Hermite geometry.
    """
    in_to_ft = 1 / 12.0
    x = np.linspace(0.0, p.x_back_end, N)
    r_out = np.array([r_hull(xi, p) for xi in x])
    r_in = np.clip(r_out - p.t_wall_in, 0.0, None)
    dx = (p.x_back_end - 0.0) / (N - 1)
    V_in3 = np.trapz(math.pi * r_in**2, dx=dx)
    return V_in3 / 1728.0  # ft^3

def cylinder_interior_volume_excluding_bay(p: HullDims, mb_front: float, mb_back: float):
    """Cylinder-only interior volume excluding motor bay."""
    in_to_ft = 1 / 12.0
    R_in_ft = (p.R - p.t_wall_in) * in_to_ft
    A_in_ft2 = math.pi * (R_in_ft**2)
    segs = []
    mb_front = max(0.0, min(mb_front, p.Lc))
    mb_back = max(0.0, min(mb_back, p.Lc))
    if mb_back < mb_front: mb_front, mb_back = mb_back, mb_front
    if mb_front > 0.0: segs.append(mb_front)
    if mb_back < p.Lc: segs.append(p.Lc - mb_back)
    return A_in_ft2 * sum(segs) * in_to_ft

def free_flood_volume_excluding_bay(inp: Inputs):
    p = inp.p
    in_to_ft = 1 / 12.0

    # full hull interior
    V_total_ft3 = section_volume_interior_ft3(p, N=inp.N_slices)

    # motor bay interior (sealed)
    R_in_ft = (p.R - p.t_wall_in) * in_to_ft
    A_in_ft2 = math.pi * R_in_ft**2
    L_bay_ft = max(0.0, inp.motor_bay_back_in - inp.motor_bay_front_in) * in_to_ft
    V_bay_ft3 = A_in_ft2 * L_bay_ft

    return V_total_ft3 - V_bay_ft3, V_total_ft3, V_bay_ft3

# ----------------------------
# Ball packing capacity
# ----------------------------
def ball_volume_ft3(ball_d_in: float):
    in_to_ft = 1/12.0
    r_ft = 0.5 * ball_d_in * in_to_ft
    return (4/3) * math.pi * (r_ft**3)

def max_balls_capacity(inp: Inputs):
    V_free_ft3, V_total_ft3, V_bay_ft3 = free_flood_volume_excluding_bay(inp)
    V_ball = ball_volume_ft3(inp.ball_d_in)
    n_max = math.floor((inp.eta_packing * V_free_ft3) / V_ball) if V_ball > 0 else 0
    return {
        "V_free_ft3": V_free_ft3,
        "V_total_ft3": V_total_ft3,
        "V_bay_ft3": V_bay_ft3,
        "V_ball_ft3": V_ball,
        "eta_packing": inp.eta_packing,
        "n_max_balls": n_max
    }

# ----------------------------
# Output
# ----------------------------
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
    print(f"Total interior (all):               {fmt(res['V_total_ft3'],3)}")
    print(f"Free-flood available:               {fmt(res['V_free_ft3'],3)}")
    print("---- Capacity")
    print(f"Single ball volume (ft^3):          {fmt(res['V_ball_ft3'],6)}")
    print(f"Max packed balls:                   {res['n_max_balls']}")
