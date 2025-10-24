import sympy as sp

# ----------------------------
# Geometry
# ----------------------------
R_out = 12                       # outer radius [in]
t_wall = sp.Rational(1, 4)       # wall thickness [in]
R_in = R_out - t_wall            # inner radius [in]

lf = 29.5                        # front cone length [in]
lb = 31.5                        # back cone nominal length [in]
lb_to_cylin = 7
lf_to_cylin = 5
# Use total ≈ 243 in -> straight tube length:
L_total = 243
L = sp.Rational(L_total, 1) - (sp.Rational(lf,1) + sp.Rational(lb,1))  # => 182 in

r_tip_out = 3.5                  # back cone cut radius (outer) [in]
r_tip_in  = r_tip_out - t_wall   # inner cut radius (no Max; = 3.25 in)

# ----------------------------
# Symbols
# ----------------------------
x, h, lb_sym, R_sym, r_sym = sp.symbols("x h lb R r")

# ----------------------------
# Helper: front cone profile (parabola), integrated with shells
# y = f(x) = - (lf/R^2) x^2 + lf
# ----------------------------
def front_cone_volume(R_val, lf_val):
    fx = (-(lf_val/(R_val**2)))*(x**2) + lf_val
    V = (2*sp.pi) * sp.integrate(fx*x, (x, 0, R_val))
    return sp.simplify(V)

# ----------------------------
# Middle tube volumes (cylinders)
# ----------------------------
def tube_volume(R_val, L_val):
    return sp.pi * (R_val**2) * L_val

# ----------------------------
# Back cone with cut tip at radius r (paraboloid extension)
# Start: f(x) = -((lb+h)/R^2) x^2 + (lb+h); choose h so f(r) = lb
# Volume = 2π∫_{r..R} x f(x) dx + 2π∫_{0..r} x*lb dx
# ----------------------------
def back_cone_volume_with_cut(R_val, r_val, lb_val):
    fx_uncut = (-(lb_sym + h)/(R_sym**2))*(x**2) + (lb_sym + h)
    h_sol = sp.solve(sp.Eq(fx_uncut.subs({x: r_sym, lb_sym: lb_sym, R_sym: R_sym}), lb_sym), h)[0]
    fx = sp.simplify(fx_uncut.subs(h, h_sol))

    V1 = (2*sp.pi) * sp.integrate(fx*x, (x, r_sym, R_sym))   # shells
    V2 = (2*sp.pi) * sp.integrate(lb_sym*x, (x, 0, r_sym))   # flat core
    Vb = sp.simplify(V1 + V2)

    return sp.simplify(Vb.subs({R_sym: R_val, r_sym: r_val, lb_sym: lb_val}))

# ----------------------------
# OUTER volumes
# ----------------------------
Vf_out = front_cone_volume(R_out, lf)                # in^3
Vm_out = tube_volume(R_out, L)                       # in^3
Vb_out = back_cone_volume_with_cut(R_out, r_tip_out, lb)  # in^3

V_total_out_in3 = sp.simplify(Vf_out + Vm_out + Vb_out)   # in^3
V_total_out_ft3 = V_total_out_in3 / (12**3)               # ft^3

# ----------------------------
# INNER volumes
# ----------------------------
Vf_in = front_cone_volume(R_in, lf)                  # in^3
Vm_in = tube_volume(R_in, L)                         # in^3
Vb_in = back_cone_volume_with_cut(R_in, r_tip_in, lb)     # in^3

V_total_in_in3 = sp.simplify(Vf_in + Vm_in + Vb_in)       # in^3
V_total_in_ft3 = V_total_in_in3 / (12**3)                 # ft^3


def roundnum(V_expr):
    coeff_pi = sp.simplify(V_expr / sp.pi)
    if coeff_pi.has(sp.pi):
        return sp.nsimplify(sp.N(V_expr, 6))
    return sp.Rational(round(sp.N(coeff_pi, 10), 3)).limit_denominator() * sp.pi

V_out_ft3_rounded = roundnum(V_total_out_ft3)
V_in_ft3_rounded  = roundnum(V_total_in_ft3)

# ----------------------------
# Print results
# ----------------------------
print("=== Using total length ≈ 243 in ⇒ L = 182 in ===")
print("=== OUTER (displacement) geometry ===")
print("V_front_out  (ft^3):", sp.N(Vf_out/(12**3), 6))
print("V_tube_out   (ft^3):", sp.N(Vm_out/(12**3), 6))
print("V_back_out   (ft^3):", sp.N(Vb_out/(12**3), 6))
print("V_total_out  (ft^3):", sp.N(V_total_out_ft3, 6))

print("\n=== INNER (floodable/fillable) geometry (t = 0.25 in) ===")
print("V_front_in   (ft^3):", sp.N(Vf_in/(12**3), 6))
print("V_tube_in    (ft^3):", sp.N(Vm_in/(12**3), 6))
print("V_back_in    (ft^3):", sp.N(Vb_in/(12**3), 6))
print("V_total_in   (ft^3):", sp.N(V_total_in_ft3, 6))

# Material volume (solid laminate/metal), for sanity check
V_wall_in3 = sp.simplify(V_total_out_in3 - V_total_in_in3)
V_wall_ft3 = V_wall_in3 / (12**3)
print("\n=== Wall / material volume (outer - inner) ===")
print("V_wall (ft^3):", sp.N(V_wall_ft3, 6))
