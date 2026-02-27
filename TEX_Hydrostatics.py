import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# ============================================================
# 1) WEIGHT TABLE -> CG CALCULATIONS
# ============================================================

@dataclass
class Item:
    name: str
    w: float                 # weight [lb]
    kg: float                # vertical arm (ft above BL)
    lcg: float               # longitudinal arm (ft, +FWD / -AFT from midship)
    tcg: float = 0.0         # transverse arm (ft, +STBD / -PORT), optional
    note: str = ""           # label only (CTR/PORT/STBD)

def compute_cg(items: List[Item]) -> Dict[str, float]:
    W = sum(it.w for it in items)
    if W <= 0.0:
        raise ValueError("Total weight must be > 0.")

    Mz = sum(it.w * it.kg  for it in items)
    Mx = sum(it.w * it.lcg for it in items)
    My = sum(it.w * it.tcg for it in items)

    return {
        "W_total": W,
        "Mz_total": Mz,
        "Mx_total": Mx,
        "My_total": My,
        "KG":  Mz / W,
        "LCG": Mx / W,
        "TCG": My / W,
    }

def print_cg_report(items: List[Item], title: str = "CG Report") -> Dict[str, float]:
    r = compute_cg(items)
    print(f"=== {title} ===")
    print(f"{'Item':<30} {'W':>10} {'KG':>10} {'VM':>12} {'LCG':>10} {'LM':>12} {'TCG':>10} {'TM':>12}")
    print("-"*120)
    for it in items:
        print(f"{it.name:<30} "
              f"{it.w:>10.2f} {it.kg:>10.3f} {it.w*it.kg:>12.2f} "
              f"{it.lcg:>10.3f} {it.w*it.lcg:>12.2f} "
              f"{it.tcg:>10.3f} {it.w*it.tcg:>12.2f}")
    print("-"*120)
    print(f"{'TOTAL':<30} {r['W_total']:>10.2f} {'':>10} {r['Mz_total']:>12.2f} "
          f"{'':>10} {r['Mx_total']:>12.2f} {'':>10} {r['My_total']:>12.2f}")
    print()
    print(f"KG  = {r['KG']:.4f} ft (above BL)")
    print(f"LCG = {r['LCG']:.4f} ft (+FWD / -AFT midship)")
    print(f"TCG = {r['TCG']:.4f} ft (+STBD / -PORT)")
    return r

# ============================================================
# 2) HULL GEOMETRY (outer radius profile), inches
# ============================================================

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

SMOOTH_LEN_FRONT = 27.0  # [in]

def r_front(x: float, p: HullDims) -> float:
    Ls = max(0.0, min(SMOOTH_LEN_FRONT, p.Lf))
    x0 = p.Lf - Ls
    if x <= 0.0:  return 0.0
    if x >= p.Lf: return p.R

    r_par = p.R * math.sqrt(max(0.0, x / p.Lf))
    if x < x0:
        return r_par

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
    if xi <= 0.0:   return p.R
    if xi >= p.Lb:  return p.r_tip

    a = p.Lb / (1.0 - (p.r_tip / p.R)**2)
    r_par = p.R * math.sqrt(max(0.0, 1.0 - xi / a))
    s = smooth_cos_01(xi / p.Lb)
    r_val = p.R - s * (p.R - r_par)

    if p.Lb - xi < 1e-9:
        r_val = p.r_tip
    return r_val

def r_hull_out(x: float, p: HullDims) -> float:
    if x < 0.0:              return 0.0
    if x <= p.x_front_end:   return r_front(x, p)
    if x <= p.x_cyl_end:     return r_cyl(x, p)
    if x <= p.x_back_end:    return r_back(x, p)
    return 0.0

# ============================================================
# 3) HYDROSTATICS (draft varies), inches input, ft output
# ============================================================

@dataclass
class HydroModel:
    R_out_in: float = 12.0
    L_front_in: float = 29.5
    L_cyl_in: float = 182.0
    L_back_in: float = 31.5
    r_tip_out_in: float = 3.5

    gamma_lb_ft3: float = 62.4
    N_samples: int = 20000

def seg_area_in2(d_in: float, R_in: float) -> float:
    if R_in <= 0.0:
        return 0.0
    d = max(0.0, min(d_in, 2.0 * R_in))
    if d <= 0.0:
        return 0.0
    if d >= 2.0 * R_in:
        return math.pi * R_in * R_in

    u = (R_in - d) / max(1e-12, R_in)
    u = max(-1.0, min(1.0, u))

    A1 = (R_in**2) * math.acos(u)
    rad = max(0.0, 2.0 * R_in * d - d * d)
    A2 = (R_in - d) * math.sqrt(rad)
    return A1 - A2

def waterline_chord_in(d_in: float, R_in: float) -> float:
    if R_in <= 0.0:
        return 0.0
    d = max(0.0, min(d_in, 2.0 * R_in))
    if d <= 0.0 or d >= 2.0 * R_in:
        return 0.0
    return 2.0 * math.sqrt(max(0.0, 2.0 * R_in * d - d * d))

def trapz_uniform(x: np.ndarray, y: np.ndarray) -> float:
    dx = (x[-1] - x[0]) / (len(x) - 1)
    return 0.5 * dx * (y[0] + 2.0*np.sum(y[1:-1]) + y[-1])

def seg_centroid_from_bottom_in(d_in: float, R_in: float) -> float:
    """Vertical centroid of submerged circular segment from BOTTOM of circle [in]."""
    if R_in <= 0.0:
        return 0.0

    d = max(0.0, min(d_in, 2.0 * R_in))
    if d <= 0.0:
        return 0.0
    if d >= 2.0 * R_in:
        return R_in

    theta = 2.0 * math.acos((R_in - d) / R_in)
    denom = (theta - math.sin(theta))
    if denom <= 1e-12:
        return 0.0

    y_bar_from_center = (4.0 * R_in * (math.sin(theta / 2.0) ** 3)) / (3.0 * denom)
    return R_in - y_bar_from_center

def build_hydro_solver(model: HydroModel):
    p = HullDims(
        R=model.R_out_in, Lf=model.L_front_in, Lc=model.L_cyl_in,
        Lb=model.L_back_in, r_tip=model.r_tip_out_in
    )

    Ltot = p.x_back_end
    x_midship_bow_in = 0.5 * Ltot

    n = max(3, int(model.N_samples))
    x_bow = np.linspace(0.0, Ltot, n)                # [in]
    r = np.array([r_hull_out(xi, p) for xi in x_bow]) # [in]
    x_mid = x_bow - x_midship_bow_in                  # [in] +FWD / -AFT

    in2_to_ft2 = 1.0 / 144.0
    in3_to_ft3 = 1.0 / 1728.0
    in4_to_ft4 = 1.0 / (12.0**4)

    def hydro(T_in: float) -> dict:
        c = np.array([waterline_chord_in(T_in, ri) for ri in r])         # [in]
        Asec = np.array([seg_area_in2(T_in, ri) for ri in r])            # [in^2]
        zc = np.array([seg_centroid_from_bottom_in(T_in, ri) for ri in r])  # [in]

        Aw_in2 = trapz_uniform(x_bow, c)
        Vsub_in3 = trapz_uniform(x_bow, Asec)

        Aw_ft2 = Aw_in2 * in2_to_ft2
        Vsub_ft3 = Vsub_in3 * in3_to_ft3
        W_disp_lb = model.gamma_lb_ft3 * Vsub_ft3

        # LCF + Ixx/Iyy
        if Aw_in2 <= 1e-12:
            xLCF_bow_in = 0.0
            xLCF_mid_in = 0.0
            Ixx_in4 = 0.0
            Iyy_in4 = 0.0
        else:
            xLCF_bow_in = trapz_uniform(x_bow, x_bow * c) / Aw_in2
            xLCF_mid_in = xLCF_bow_in - x_midship_bow_in
            Ixx_in4 = trapz_uniform(x_bow, (c**3) / 12.0)
            Iyy_in4 = trapz_uniform(x_bow, ((x_mid - xLCF_mid_in) ** 2) * c)

        # KB (KBL)
        if Vsub_in3 <= 1e-12:
            KB_in = 0.0
        else:
            Mz_in4 = trapz_uniform(x_bow, zc * Asec)
            KB_in = Mz_in4 / Vsub_in3

        KB_ft = KB_in / 12.0
        Ixx_ft4 = Ixx_in4 * in4_to_ft4
        Iyy_ft4 = Iyy_in4 * in4_to_ft4

        # BM and KM
        if Vsub_ft3 <= 1e-12:
            BMt_ft = 0.0
            BMl_ft = 0.0
        else:
            BMt_ft = Ixx_ft4 / Vsub_ft3
            BMl_ft = Iyy_ft4 / Vsub_ft3

        KMt_ft = KB_ft + BMt_ft
        KMl_ft = KB_ft + BMl_ft

        return {
            "T_in": float(T_in),

            "Aw_ft2": float(Aw_ft2),
            "Vsub_ft3": float(Vsub_ft3),
            "W_disp_lb": float(W_disp_lb),

            "xLCF_mid_ft": float(xLCF_mid_in / 12.0),

            "KB_ft": float(KB_ft),
            "KBL_ft": float(KB_ft),

            "Ixx_ft4": float(Ixx_ft4),
            "Iyy_ft4": float(Iyy_ft4),

            "BMt_ft": float(BMt_ft),
            "BMl_ft": float(BMl_ft),

            "KMt_ft": float(KMt_ft),
            "KMl_ft": float(KMl_ft),

            "Ltot_ft": float(Ltot / 12.0),
        }

    hydro.x_bow = x_bow
    hydro.x_mid = x_mid
    hydro.r = r
    hydro.Ltot_in = Ltot
    return hydro

def fmt(v, nd=6) -> str:
    return f"{v:.{nd}f}"

# ============================================================
# 4) RUN BOTH + COMPUTE GM USING KG
# ============================================================

if __name__ == "__main__":
    # ---- Weight table ----
    items = [
        Item("LIGHTSHIP", 425.00,  1.000,  0.000, 0.0, "CTR"),
        Item("WEIGHTED KEEL (FWD)", 250.00, -2.000,  7.000, 0.0, "CTR"),
        Item("WEIGHTED KEEL (AFT)", 250.00, -2.000, -7.000, 0.0, "CTR"),
        Item("MOTOR", 145.00,  1.500, -1.080, 0.0, "CTR"),
        Item("CONNING TOWER", 40.00,  3.500, -1.080, 0.0, "CTR"),
        Item("ELECTRONICS", 10.00,  7.125, -1.080, 0.0, "PORT"),
        Item("BATTERY", 20.85,  0.583,  3.083, 0.0, "STBD"),
        Item("GAS TANK", 33.92,  0.583,  3.833, 0.0, "STBD"),
        Item("DIVING PLANE FRONT", 3.10,  1.000,  7.500, 0.0, "STBD"),
        Item("DIVING PLANE BACK", 3.10,  1.000, -7.500, 0.0, "STBD"),
        Item("RUDDER BACK", 3.10,  1.000, -8.542, 0.0, "STBD"),
        Item("RUDDER FRONT", 3.10,  1.000,  8.583, 0.0, "STBD"),
        Item("SERVO+BOX+OIL FRONT", 20.10,  1.000,  7.500, 0.0, "STBD"),
        Item("SERVO+BOX+OIL BACK", 20.10,  1.000, -7.500, 0.0, "STBD"),
        Item("SERVO+BOX+OIL RUDDER FRONT", 20.10,  1.000,  8.583, 0.0, "STBD"),
        Item("SERVO+BOX+OIL RUDDER BACK", 20.10,  1.000, -8.542, 0.0, "STBD"),
    ]

    cg = print_cg_report(items, title="TEX Weight & CG Calculation")
    KG = cg["KG"]
    W_table = cg["W_total"]

    # ---- Hydro model ----
    model = HydroModel(
        R_out_in=12.0,
        L_front_in=29.5,
        L_cyl_in=182.0,
        L_back_in=31.5,
        r_tip_out_in=3.5,
        gamma_lb_ft3=62.4,
        N_samples=20000,
    )
    hydro = build_hydro_solver(model)

    # Choose draft
    T_in = 12.0
    out = hydro(T_in)

    # ---- GM ----
    GMt_ft = out["KMt_ft"] - KG
    GMl_ft = out["KMl_ft"] - KG

    print("\n=== Hydro + Stability ===")
    print(f"Draft T        = {fmt(T_in,3)} in")
    print(f"Aw             = {fmt(out['Aw_ft2'],4)} ft^2")
    print(f"Vsub           = {fmt(out['Vsub_ft3'],4)} ft^3")
    print(f"Wdisp(hydro)   = {fmt(out['W_disp_lb'],2)} lb")
    print(f"xLCF(mid)      = {fmt(out['xLCF_mid_ft'],4)} ft (+FWD / -AFT midship)")

    print(f"\nKB (=KBL)       = {fmt(out['KB_ft'],6)} ft")

    print(f"\nIxx            = {fmt(out['Ixx_ft4'],6)} ft^4")
    print(f"Iyy            = {fmt(out['Iyy_ft4'],6)} ft^4")
    print(f"BMt            = {fmt(out['BMt_ft'],6)} ft")
    print(f"BMl            = {fmt(out['BMl_ft'],6)} ft")
    print(f"KMt            = {fmt(out['KMt_ft'],6)} ft")
    print(f"KMl            = {fmt(out['KMl_ft'],6)} ft")

    print(f"\nKG (from table) = {fmt(KG,6)} ft")
    print(f"GMt            = {fmt(GMt_ft,6)} ft")
    print(f"GMl            = {fmt(GMl_ft,6)} ft")

    print("\n=== Displacement check ===")
    print(f"W_total(table)  = {fmt(W_table,2)} lb")
    print(f"Wdisp(hydro)    = {fmt(out['W_disp_lb'],2)} lb")
    print(f"Delta           = {fmt(out['W_disp_lb'] - W_table,2)} lb (hydro - table)")