import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# =====================================================================
#  HULL GEOMETRY
# =====================================================================

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

def smooth_cos_01(t):
    t = min(1, max(0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)

SMOOTH_LEN_FRONT = 27.0

def r_front(x, p: HullDims):
    Ls = min(SMOOTH_LEN_FRONT, p.Lf)
    x0 = p.Lf - Ls
    if x <= 0: return 0
    if x >= p.Lf: return p.R

    r_par = p.R * math.sqrt(x / p.Lf)
    if x < x0: return r_par

    r0 = p.R * math.sqrt(x0 / p.Lf)
    drdx0 = p.R / (2 * math.sqrt(max(1e-12, p.Lf * x0)))
    t = (x - x0) / Ls

    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2

    return h00*r0 + h10*(Ls*drdx0) + h01*p.R

def r_back(x, p: HullDims):
    xi = x - p.x_cyl_end
    if xi <= 0: return p.R
    if xi >= p.Lb: return p.r_tip

    a = p.Lb / (1 - (p.r_tip/p.R)**2)
    r_par = p.R * math.sqrt(max(0, 1 - xi/a))
    s = smooth_cos_01(xi / p.Lb)
    return p.R - s*(p.R - r_par)

def r_cyl(x, p): return p.R

def r_hull(x, p):
    if x <= p.x_front_end: return r_front(x, p)
    if x <= p.x_cyl_end:   return r_cyl(x, p)
    if x <= p.x_back_end:  return r_back(x, p)
    return 0

# =====================================================================
#  CENTER OF BUOYANCY (CB)
# =====================================================================

def seg_area_in2(d, R):
    d = max(0, min(d, 2*R))
    if d <= 0: return 0
    if d >= 2*R: return math.pi*R*R
    u = (R - d)/R
    return R*R*math.acos(u) - (R-d)*math.sqrt(max(0, 2*R*d - d*d))

def seg_theta(d, R):
    return 2*math.acos((R-d)/R)

def seg_centroid_below_WL_in(d, R):
    if d <= 0: return 0
    if d >= 2*R: return R
    t = seg_theta(d, R)
    denom = t - math.sin(t)
    return (4*R*(math.sin(t/2)**3)) / (3*denom)

@dataclass
class Inputs:
    R_out_in: float = 12.0
    L_front_in: float = 29.5
    L_cyl_in: float = 182.0
    L_back_in: float = 31.5
    draft_in: float = 12.0
    gamma: float = 62.4

def compute_cb_surface(inp: Inputs):
    p = HullDims()
    ybar = seg_centroid_below_WL_in(inp.draft_in, inp.R_out_in)
    CBx = p.Lf + p.Lc / 2
    z_CB = inp.draft_in - ybar
    return {
        "draft_used_in": inp.draft_in,
        "CBx_in": CBx,
        "z_CB_from_bottom_in": z_CB
    }

# =====================================================================
#  CENTER OF GRAVITY (CG)
# =====================================================================

in_to_ft = 1/12

@dataclass
class CGSystem:
    p: HullDims
    t_wall_in: float = 0.25
    points: list = field(default_factory=list)
    W_sum: float = 0
    Mx_sum: float = 0
    Mz_sum: float = 0

    def add_point(self, name, W, x, z):
        self.points.append((name, W, x, z))
        self.W_sum += W
        self.Mx_sum += W*x
        self.Mz_sum += W*z

    def add_hull_shell(self, density):
        x = np.linspace(0, self.p.x_back_end, 4000)
        r = np.array([r_hull(xi, self.p) for xi in x])

        z = r  # hull surface measured from keel

        dV = 2*np.pi*(r*in_to_ft)*(self.t_wall_in*in_to_ft)*(x[1]-x[0])*in_to_ft
        dW = dV * density

        xcg = np.average(x, weights=dW)
        zcg = np.average(z - 0.5*self.t_wall_in, weights=dW)

        self.add_point("HullShell", np.sum(dW), xcg, zcg)

    def cg(self):
        return {
            "x_in": self.Mx_sum / self.W_sum,
            "z_in": self.Mz_sum / self.W_sum
        }

# =====================================================================
#  ONE PLOT — CB + CG + CG'
# =====================================================================

def plot_hull_with_cb_and_cg(inp: Inputs, r_cb, cg_sys: CGSystem):

    p = cg_sys.p

    x = np.linspace(0, p.x_back_end, 1500)
    r = np.array([r_hull(xi, p) for xi in x])

    x = np.append(x, p.x_back_end)
    r = np.append(r, r_hull(p.x_back_end, p))

    r_top = r + p.R
    r_bot = -r + p.R

    plt.figure(figsize=(14, 6))

    plt.plot(x, r_top, color="#0052CC", lw=2.2)
    plt.plot(x, r_bot, color="#0052CC", lw=2.2)
    plt.vlines(p.x_back_end, r_bot[-1], r_top[-1], color="#0052CC", lw=2.2)

    y_wl = inp.draft_in
    plt.axhline(y_wl, color="#888", ls="--", lw=1.2)

    # Existing CB + CG
    x_cb = r_cb["CBx_in"]
    z_cb = r_cb["z_CB_from_bottom_in"]

    cg = cg_sys.cg()
    x_cg = cg["x_in"]
    z_cg = cg["z_in"]

    plt.scatter([x_cb], [z_cb], s=60, color="#E74C3C")
    plt.scatter([x_cg], [z_cg], s=60, color="#2ECC71")

    plt.text(x_cb - 2.5, z_cb, "CB", ha='right', va='center', fontsize=11)
    plt.text(x_cg + 2.5, z_cg, "CG", ha='left', va='center', fontsize=11)

    # ==========================================
    # CG' EXACTLY inline with CB at z = 3.5"
    # ==========================================
    x_newcg = x_cb
    z_newcg = 3.5

    plt.scatter([x_newcg], [z_newcg], s=60, color="#8E44AD")
    plt.text(x_newcg + 2.5, z_newcg, "CG'", ha='left', va='center', fontsize=11)

    # Legend
    cb_label = f'CB: ({x_cb:.1f}", {z_cb:.1f}")'
    cg_label = f'CG: ({x_cg:.1f}", {z_cg:.1f}")'
    cg_new_label = f'CG\': ({x_newcg:.1f}", {z_newcg:.1f}")'

    plt.legend(
        [
            plt.Line2D([], [], marker="o", color="#E74C3C", linestyle=""),
            plt.Line2D([], [], marker="o", color="#2ECC71", linestyle=""),
            plt.Line2D([], [], marker="o", color="#8E44AD", linestyle=""),
            plt.Line2D([], [], color="#888", linestyle="--")
        ],
        [
            cb_label,
            cg_label,
            cg_new_label,
            "Waterline"
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.32),
        ncol=1
    )

    plt.xlim(-5, p.x_back_end + 10)
    plt.ylim(-5, 2*p.R + 10)

    plt.xlabel("x [inch] (nose tostern)")
    plt.ylabel("z [inch] (from keel)")
    plt.title("TEX — CB, CG, and Target CG'")
    plt.grid(False)
    plt.gca().set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    p = HullDims()

    cg = CGSystem(p=p, t_wall_in=0.25)
    cg.add_hull_shell(170.0)
    cg.add_point("Motor", 145.0, p.Lf + 103.5, 24.0)
    cg.add_point("Conning Tower", 40.0, p.Lf + 103.5, 42.0)
    cg.add_point("Electronics", 20.0, p.Lf + 103.5, 85.5)

    inp = Inputs(draft_in=12)
    r_cb = compute_cb_surface(inp)

    plot_hull_with_cb_and_cg(inp, r_cb, cg)
