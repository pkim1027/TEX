# -------------------------------------------------
# TEX Hull — Profile-Only Plotter
# -------------------------------------------------
# Draws the hull shape (front cone, cylinder, back cone)
# -------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------------------
# HULL GEOMETRY DEFINITIONS
# ----------------------------
@dataclass(frozen=True)
class HullDims:
    R: float = 12.0       # [in] outer radius
    Lf: float = 29.5      # [in] front cone length
    Lc: float = 182.0     # [in] straight cylinder length
    Lb: float = 31.5      # [in] back cone length
    r_tip: float = 3.5    # [in] back tip radius

    @property
    def x_front_end(self): return self.Lf
    @property
    def x_cyl_end(self):   return self.Lf + self.Lc
    @property
    def x_back_end(self):  return self.Lf + self.Lc + self.Lb

# Smooth transition function
def smooth_cos_01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)

SMOOTH_LEN_FRONT = 27.0  # [in] of Hermite smoothing near front join

def r_front(x: float, p: HullDims) -> float:
    """Front cone radius profile (smoothed parabola)."""
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

def r_back(x: float, p: HullDims) -> float:
    """Back cone radius profile."""
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

def r_cyl(x: float, p: HullDims) -> float:
    """Straight cylinder."""
    return p.R

def r_hull(x: float, p: HullDims) -> float:
    """Full hull shape function."""
    if x < 0.0: return 0.0
    if x <= p.x_front_end: return r_front(x, p)
    if x <= p.x_cyl_end:   return r_cyl(x, p)
    if x <= p.x_back_end:  return r_back(x, p)
    return 0.0

# ----------------------------
# PLOTTER
# ----------------------------
def plot_hull_only():
    p = HullDims()
    x = np.linspace(0, p.x_back_end, 1200)
    r = np.array([r_hull(xi, p) for xi in x])

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, r,  color="#0052CC", lw=2.2, label="Outer Hull", solid_capstyle="round")
    plt.plot(x, -r, color="#0052CC", lw=2.2, solid_capstyle="round")

    # Section lines
    plt.vlines(p.x_front_end, -p.R, p.R, linestyle='--', color="#888888")
    plt.vlines(p.x_cyl_end,   -p.R, p.R, linestyle='--', color="#888888")
    plt.vlines(p.x_back_end,  -p.r_tip, p.r_tip, color="#0052CC", lw=2.2)

    # Section labels
    plt.text(p.Lf / 2, -1.6*p.R, "Front Cone", ha='center', va='top', color="#1f77b4", fontsize=10)
    plt.text(p.Lf + p.Lc / 2, -1.6*p.R, "Cylinder", ha='center', va='top', color="#2ca02c", fontsize=10)
    plt.text(p.Lf + p.Lc + p.Lb / 2, -1.6*p.R, "Back Cone", ha='center', va='top', color="#d62728", fontsize=10)

    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x [in] (nose → stern)", labelpad=20)
    ax.set_ylabel("r [in]")
    plt.title("TEX Hull Shape")
    plt.tight_layout()
    plt.show()

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    plot_hull_only()
