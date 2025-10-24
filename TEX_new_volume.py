# -------------------------------------------------
# TEX Hull — Volume Integration
# -------------------------------------------------
# - Front: parabolic √(x/Lf) with C¹ Hermite smoothing to cylinder
# - Cylinder: constant radius
# - Back: cosine-blended taper to truncated tip
# -------------------------------------------------

import math
import numpy as np
from dataclasses import dataclass

# ----------------------------
# Hull geometry profile
# ----------------------------
@dataclass(frozen=True)
class HullDims:
    R: float = 12.0
    Lf: float = 29.5
    Lc: float = 182.0
    Lb: float = 31.5
    r_tip: float = 3.5

    @property
    def x_front_end(self): return self.Lf
    @property
    def x_cyl_end(self):   return self.Lf + self.Lc
    @property
    def x_back_end(self):  return self.Lf + self.Lc + self.Lb

def smooth_cos_01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)

SMOOTH_LEN_FRONT = 27.0  # [in] Hermite window near the front join

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
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -   t**2
    return h00*r0 + h10*(Ls*drdx0) + h01*p.R + h11*(Ls*0.0)

def r_cyl(x: float, p: HullDims) -> float:
    return p.R

def r_back(x: float, p: HullDims) -> float:
    xi = x - p.x_cyl_end
    if xi <= 0.0: return p.R
    if xi >= p.Lb: return p.r_tip
    a = p.Lb / (1.0 - (p.r_tip / p.R)**2)
    r_par = p.R * math.sqrt(max(0.0, 1.0 - xi / a))
    s = smooth_cos_01(xi / p.Lb)
    r_val = p.R - s * (p.R - r_par)
    if p.Lb - xi < 1e-9:
        r_val = p.r_tip
    return r_val

def r_hull(x: float, p: HullDims) -> float:
    if x < 0.0: return 0.0
    if x <= p.x_front_end: return r_front(x, p)
    if x <= p.x_cyl_end:   return r_cyl(x, p)
    if x <= p.x_back_end:  return r_back(x, p)
    return 0.0

# ----------------------------
# Inputs for volume integration
# ----------------------------
@dataclass
class VolInputs:
    R_out_in: float = 12.0
    t_wall_in: float = 0.25
    L_front_in: float = 29.5
    L_cyl_in: float = 182.0
    L_back_in: float = 31.5
    r_tip_out_in: float = 3.5
    motor_bay_front_in: float = 40.0
    motor_bay_back_in:  float = 90.0
    N_samples: int = 20000

# ----------------------------
# Numerical integration helpers
# ----------------------------
def _trapz_pi_r2(x: np.ndarray, r: np.ndarray) -> float:
    """Compute π ∫ r^2 dx using trapezoidal rule on a uniform grid (returns in³)."""
    r2 = r * r
    dx = (x[-1] - x[0]) / (len(x) - 1)
    return math.pi * (0.5 * dx) * (r2[0] + 2.0 * np.sum(r2[1:-1]) + r2[-1])

def _sample_profile(p: HullDims, t_wall_in: float, x0: float, x1: float, n: int):
    x = np.linspace(x0, x1, n)
    r_out = np.array([r_hull(xi, p) for xi in x])
    r_in  = np.clip(r_out - t_wall_in, 0.0, None)
    return x, r_out, r_in

def _integrate_interval(p: HullDims, t_wall_in: float, a: float, b: float, n: int) -> tuple[float, float]:
    if b <= a:
        return 0.0, 0.0
    n = max(3, n)
    x, r_out, r_in = _sample_profile(p, t_wall_in, a, b, n)
    return _trapz_pi_r2(x, r_out), _trapz_pi_r2(x, r_in)

# ----------------------------
# Main compute
# ----------------------------
def compute_volumes_with_motor(inp: VolInputs):
    p = HullDims(R=inp.R_out_in, Lf=inp.L_front_in, Lc=inp.L_cyl_in, Lb=inp.L_back_in, r_tip=inp.r_tip_out_in)
    Ltot = p.x_back_end

    # Total hull volumes
    V_out_in3, V_in_in3 = _integrate_interval(p, inp.t_wall_in, 0.0, Ltot, inp.N_samples)

    # Motor bay absolute span (clamped within cylinder)
    bay_a = p.Lf + max(0.0, min(inp.motor_bay_front_in, inp.L_cyl_in))
    bay_b = p.Lf + max(0.0, min(inp.motor_bay_back_in,  inp.L_cyl_in))
    if bay_b < bay_a:
        bay_a, bay_b = bay_b, bay_a
    n_bay = max(5, int(inp.N_samples * (bay_b - bay_a) / max(1e-9, Ltot)))
    V_out_bay_in3, V_in_bay_in3 = _integrate_interval(p, inp.t_wall_in, bay_a, bay_b, n_bay)

    # Section-by-section integration
    n_sec = inp.N_samples // 10
    Vf_out, Vf_in = _integrate_interval(p, inp.t_wall_in, 0.0, p.x_front_end, n_sec)
    Vc_out, Vc_in = _integrate_interval(p, inp.t_wall_in, p.x_front_end, p.x_cyl_end, n_sec)
    Vb_out, Vb_in = _integrate_interval(p, inp.t_wall_in, p.x_cyl_end, p.x_back_end, n_sec)

    in3_to_ft3 = 1.0 / 1728.0

    return {
        "inputs": inp,

        # Totals
        "V_out_in3": V_out_in3,
        "V_in_in3":  V_in_in3,
        "V_wall_in3": max(0.0, V_out_in3 - V_in_in3),
        "V_out_ft3": V_out_in3 * in3_to_ft3,
        "V_in_ft3":  V_in_in3  * in3_to_ft3,
        "V_wall_ft3": (V_out_in3 - V_in_in3) * in3_to_ft3,

        # Motor bay only
        "V_out_bay_in3": V_out_bay_in3,
        "V_in_bay_in3":  V_in_bay_in3,
        "V_wall_bay_in3": max(0.0, V_out_bay_in3 - V_in_bay_in3),
        "V_out_bay_ft3": V_out_bay_in3 * in3_to_ft3,
        "V_in_bay_ft3":  V_in_bay_in3  * in3_to_ft3,
        "V_wall_bay_ft3": (V_out_bay_in3 - V_in_bay_in3) * in3_to_ft3,

        # Section volumes
        "Vf_out_ft3": Vf_out * in3_to_ft3, "Vf_in_ft3": Vf_in * in3_to_ft3,
        "Vc_out_ft3": Vc_out * in3_to_ft3, "Vc_in_ft3": Vc_in * in3_to_ft3,
        "Vb_out_ft3": Vb_out * in3_to_ft3, "Vb_in_ft3": Vb_in * in3_to_ft3,

        # Bay placement (absolute)
        "x_bay_front_in": bay_a,
        "x_bay_back_in":  bay_b,
        "L_bay_in":       max(0.0, bay_b - bay_a),
    }

# ----------------------------
# Pretty print
# ----------------------------
def _fmt(v, nd=6): return f"{v:.{nd}f}"

def print_volumes_with_motor(r: dict):
    inp = r["inputs"]
    print("=== TEX Hull Volumes ===")
    print("\n-- TOTAL HULL --")
    print(f"Outer  V_out : {_fmt(r['V_out_in3'])} in³ = {_fmt(r['V_out_ft3'],4)} ft³")
    print(f"Inner  V_in  : {_fmt(r['V_in_in3'])} in³ = {_fmt(r['V_in_ft3'],4)} ft³")
    print(f"Material V_w : {_fmt(r['V_wall_in3'])} in³ = {_fmt(r['V_wall_ft3'],4)} ft³")

    print("\n-- MOTOR SECTION --")
    print(f"Outer  V_out : {_fmt(r['V_out_bay_in3'])} in³ = {_fmt(r['V_out_bay_ft3'],4)} ft³")
    print(f"Inner  V_in  : {_fmt(r['V_in_bay_in3'])} in³ = {_fmt(r['V_in_bay_ft3'],4)} ft³")
    print(f"Material V_w : {_fmt(r['V_wall_bay_in3'])} in³ = {_fmt(r['V_wall_bay_ft3'],4)} ft³")

    print("\n-- BY SECTION (ft³) --")
    print(f"Front cone : outer {_fmt(r['Vf_out_ft3'],4)} / inner {_fmt(r['Vf_in_ft3'],4)}")
    print(f"Cylinder   : outer {_fmt(r['Vc_out_ft3'],4)} / inner {_fmt(r['Vc_in_ft3'],4)}")
    print(f"Back cone  : outer {_fmt(r['Vb_out_ft3'],4)} / inner {_fmt(r['Vb_in_ft3'],4)}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    vin = VolInputs(
        R_out_in=12.0,
        t_wall_in=0.25,
        L_front_in=29.5,
        L_cyl_in=182.0,
        L_back_in=31.5,
        r_tip_out_in=3.5,
        motor_bay_front_in=40.0,
        motor_bay_back_in=90.0,  # 50" bay
        N_samples=20000
    )
    res = compute_volumes_with_motor(vin)
    print_volumes_with_motor(res)
