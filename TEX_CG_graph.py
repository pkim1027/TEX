import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# ----------------------------
# Geometry
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

SMOOTH_LEN_FRONT = 27.0  # [in] smoothing near front join

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
    return h00*r0 + h10*(Ls*drdx0) + h01*p.R

def r_back(x: float, p: HullDims) -> float:
    xi = x - p.x_cyl_end
    if xi <= 0.0: return p.R
    if xi >= p.Lb: return p.r_tip
    a = p.Lb / (1.0 - (p.r_tip / p.R)**2)
    r_par = p.R * math.sqrt(max(0.0, 1.0 - xi / a))
    s = smooth_cos_01(xi / p.Lb)
    return p.R - s * (p.R - r_par)

def r_cyl(x: float, p: HullDims) -> float:
    return p.R

def r_hull(x: float, p: HullDims) -> float:
    if x <= p.x_front_end: return r_front(x, p)
    if x <= p.x_cyl_end:   return r_cyl(x, p)
    if x <= p.x_back_end:  return r_back(x, p)
    return 0.0

# ----------------------------
# Center of Gravity
# ----------------------------
in_to_ft = 1/12

@dataclass
class CGSystem:
    p: HullDims
    t_wall_in: float = 0.25
    points: list = field(default_factory=list)
    W_sum: float = 0.0
    Mx_sum: float = 0.0
    Mz_sum: float = 0.0

    def add_point(self, name: str, W_lbf: float, x_in: float, z_in: float):
        self.points.append((name, W_lbf, x_in, z_in))
        self.W_sum += W_lbf
        self.Mx_sum += W_lbf * x_in
        self.Mz_sum += W_lbf * z_in

    def add_hull_shell(self, wdens_lbf_ft3: float, N: int = 10000, name="Hull Shell"):
        x = np.linspace(0, self.p.x_back_end, N)
        r_out = np.array([r_hull(xi, self.p) for xi in x])
        dV_ft3 = (2 * math.pi * (r_out*in_to_ft) * (self.t_wall_in*in_to_ft) * (x[1]-x[0])*in_to_ft)
        dW = wdens_lbf_ft3 * dV_ft3
        W_shell = np.sum(dW)
        x_cg = np.average(x, weights=dW)
        z_cg = np.average(r_out - 0.5*self.t_wall_in, weights=dW)
        self.add_point(name, W_shell, x_cg, z_cg)

    def cg(self):
        if self.W_sum == 0: return {"W_lbf": 0, "x_in": 0, "z_in": 0}
        return {"W_lbf": self.W_sum, "x_in": self.Mx_sum/self.W_sum, "z_in": self.Mz_sum/self.W_sum}

# ----------------------------
# Plotter
# ----------------------------
def plot_hull_with_cg(cg_sys: CGSystem):
    p = cg_sys.p
    x = np.linspace(0, p.x_back_end, 1000)
    r = np.array([r_hull(xi, p) for xi in x])

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, r, color="#0052CC", lw=2.2)
    plt.plot(x, -r, color="#0052CC", lw=2.2)

    # === Vertical stern line ===
    x_stern = p.x_back_end
    r_stern = r_hull(x_stern, p)
    plt.vlines(x_stern, -r_stern, r_stern, color="#0052CC", lw=2.2)

    # === Centerline ===
    plt.axhline(y=0, color='gray', linestyle='--', lw=1.2, alpha=0.6)

    # CG point
    cg = cg_sys.cg()
    x_cg, z_cg = cg["x_in"], cg["z_in"]
    plt.scatter([x_cg], [z_cg - p.R], s=80, color="#d62728", zorder=5, label="Center of Gravity (CG)")

    plt.text(x_cg, (z_cg - p.R) - 3,
             f'CG ({x_cg:.1f}", {z_cg:.1f}")',
             ha='center', va='top', fontsize=9, color='#d62728',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3))

    plt.title("TEX — Center of Gravity", pad=10)
    plt.xlabel("x [in] (nose → stern)")
    plt.ylabel("z [in] (above keel)")
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True, alpha=0.3)

    plt.legend(loc='upper center', bbox_to_anchor=(0.75, -0.2),
               ncol=1, frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Output
# ----------------------------
if __name__ == "__main__":
    p = HullDims()
    cg = CGSystem(p=p, t_wall_in=0.25)
    cg.add_hull_shell(170.0)
    cg.add_point("Motor", 145.0, x_in=p.Lf + 103.5, z_in=24.0)
    cg.add_point("Conning Tower", 40.0, x_in=p.Lf + 103.5, z_in=42.0)
    cg.add_point("Electronics", 20.0, x_in=p.Lf + 103.5, z_in=85.5)
    plot_hull_with_cg(cg)
