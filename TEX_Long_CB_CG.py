import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Geometry (inches)
# ----------------------------
R_out = sp.Rational(12, 1)
t_wall = sp.Rational(1, 4)
R_in  = R_out - t_wall

lf = sp.Rational(59,2)              # 29.5
lb = sp.Rational(63,2)              # 31.5
L  = sp.Rational(243,1) - (lf + lb) # 182
r_tip_out = sp.Rational(7,2)        # 3.5
r_tip_in  = r_tip_out - t_wall      # 3.25

# ----------------------------
# Symbols
# ----------------------------
x = sp.symbols("x", nonnegative=True)
h, lb_sym, R_sym, r_sym = sp.symbols("h lb R r", positive=True)

# ----------------------------
# Front cone (shells): y = - (lf/R^2) x^2 + lf
# ----------------------------
def front_cone_V_M(R_val, lf_val):
    fx = (-(lf_val/(R_val**2)))*(x**2) + lf_val
    V = (2*sp.pi) * sp.integrate(fx*x, (x, 0, R_val))
    M = (sp.pi)   * sp.integrate((fx**2)*x, (x, 0, R_val))
    return sp.simplify(V), sp.simplify(M)

# ----------------------------
# Tube: V = π R^2 L ; M_abs = V * (lf + L/2)
# ----------------------------
def tube_V_M(R_val, L_val, lf_val):
    V = sp.pi * (R_val**2) * L_val
    sbar = lf_val + L_val/2
    M = V * sbar
    return sp.simplify(V), sp.simplify(M)

# ----------------------------
# Back cone with tip cut at radius r
# ----------------------------
def back_cone_V_M(R_val, r_val, lb_val, lf_val, L_val):
    fx_uncut = (-(lb_sym + h)/(R_sym**2))*(x**2) + (lb_sym + h)
    h_sol = sp.solve(sp.Eq(fx_uncut.subs({x: r_sym, lb_sym: lb_sym, R_sym: R_sym}), lb_sym), h)[0]
    fx = sp.simplify(fx_uncut.subs({h: h_sol, lb_sym: lb_val, R_sym: R_val, r_sym: r_val}))

    V_shells = (2*sp.pi) * sp.integrate(fx*x, (x, r_val, R_val))
    V_flat   = (2*sp.pi) * sp.integrate(lb_val*x, (x, 0, r_val))
    Vb_local = sp.simplify(V_shells + V_flat)

    M_shells_local = (sp.pi) * sp.integrate((fx**2)*x, (x, r_val, R_val))
    M_flat_local   = (2*sp.pi) * sp.integrate((lb_val**2)/2 * x, (x, 0, r_val))
    Mb_local = sp.simplify(M_shells_local + M_flat_local)

    offset = lf_val + L_val
    Vb_abs = Vb_local
    Mb_abs = Mb_local + Vb_local*offset
    return sp.simplify(Vb_abs), sp.simplify(Mb_abs)

# ----------------------------
# Outer totals
# ----------------------------
Vf_out, Mf_out = front_cone_V_M(R_out, lf)
Vm_out, Mm_out = tube_V_M(R_out, L, lf)
Vb_out, Mb_out = back_cone_V_M(R_out, r_tip_out, lb, lf, L)

V_out = sp.simplify(Vf_out + Vm_out + Vb_out)
M_out = sp.simplify(Mf_out + Mm_out + Mb_out)
sbar_out = sp.simplify(M_out / V_out)  # CB_long approx

# ----------------------------
# Inner totals
# ----------------------------
Vf_in, Mf_in = front_cone_V_M(R_in, lf)
Vm_in, Mm_in = tube_V_M(R_in, L, lf)
Vb_in, Mb_in = back_cone_V_M(R_in, r_tip_in, lb, lf, L)

V_in = sp.simplify(Vf_in + Vm_in + Vb_in)
M_in = sp.simplify(Mf_in + Mm_in + Mb_in)
sbar_in = sp.simplify(M_in / V_in)

# ----------------------------
# Shell centroid
# ----------------------------
V_shell = sp.simplify(V_out - V_in)
M_shell = sp.simplify(M_out - M_in)
sbar_shell = sp.simplify(M_shell / V_shell)

# ----------------------------
# Motor CG from edges:
# front edge from nose = 29.5 + 91.5 = 121.0
# back edge from nose  = 29.5 + L - 66  (since "prop end of the tube to the back edge" = 66)
# center = (front_edge + back_edge)/2
# ----------------------------
front_edge = lf + sp.Rational(183,2)           # 29.5 + 91.5 = 121.0
back_edge  = lf + L - sp.Rational(66,1)        # 29.5 + 182 - 66 = 145.5
x_motor    = (front_edge + back_edge)/2        # 133.25

# ----------------------------
# Combine CGs by weight
# ----------------------------
W_total = sp.Rational(570,1)
W_shell = sp.Rational(425,1)
W_motor = sp.Rational(145,1)

# Evaluate numerically
R = sp.symbols('R')
def eval_at_R(expr, Rval=12.0):
    f = sp.lambdify(R, expr, 'numpy')
    return float(f(float(Rval)))

x_shell = sbar_shell
x_CB    = sbar_out
x_CG    = (W_shell*x_shell + W_motor*x_motor) / W_total

x_shell_num = eval_at_R(x_shell, 12.0)
x_CB_num    = eval_at_R(x_CB, 12.0)
x_CG_num    = eval_at_R(x_CG, 12.0)

print("=== Updated with total length ≈ 243 in (L=182) ===")
print("Motor edges (in): front =", float(front_edge), ", back =", float(back_edge))
print("Motor center (in):", float(x_motor))
print("Shell centroid (in):", x_shell_num)
print("Combined CG (in):   ", x_CG_num)
print("CB approx (in):     ", x_CB_num)

# ----------------------------
# Plot
# ----------------------------
stations = [0.0, float(lf), float(lf+L), float(lf+L+lb)]
zeros = [0.0]*len(stations)
labels = ["Front Cone", "Tube Front", "Tube End", "Back Cone"]

plt.figure(figsize=(9, 3))
plt.plot(stations, zeros, linewidth=2)

# Plot CG and CB markers
plt.scatter([x_CG_num], [0], s=70, marker="x", label=f"CG = {x_CG_num:.2f} in")
plt.scatter([x_CB_num], [0], s=70, marker="x", label=f"CB ≈ {x_CB_num:.2f} in")

# Vertical dashed lines with labels under the x-axis
for xi, label in zip(stations, labels):
    plt.axvline(x=xi, ymin=0.2, ymax=0.8, linestyle="--", linewidth=1)
    plt.text(xi, -0.25, label, rotation=90, ha="center", va="top",
             fontsize=8, transform=plt.gca().get_xaxis_transform())

plt.title("Longitudinal CG and CB (Tip of Front Cone at 0 in)")
plt.xlabel("s [in]")
plt.yticks([])
plt.xlim(-5, float(lf+L+lb)+5)
plt.legend(loc="upper center", ncol=2, frameon=False)
plt.tight_layout()
plt.show()