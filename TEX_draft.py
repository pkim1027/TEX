import sympy as sp

# ---------------------------------
# Geometry (inches) — your numbers
# ---------------------------------
R_out = sp.Rational(12, 1)
t_wall = sp.Rational(1, 4)
R_in  = R_out - t_wall

lf = sp.Rational(59,2)              # 29.5
lb = sp.Rational(63,2)              # 31.5
L  = sp.Rational(243,1) - (lf + lb) # 182
r_tip_out = sp.Rational(7,2)        # 3.5
r_tip_in  = r_tip_out - t_wall      # 3.25

x = sp.symbols("x", nonnegative=True)
h, lb_sym, R_sym, r_sym = sp.symbols("h lb R r", positive=True)

# ----------------------------
# Volume helpers (your style)
# ----------------------------
def front_cone_volume(R_val, lf_val):
    fx = (-(lf_val/(R_val**2)))*(x**2) + lf_val
    return (2*sp.pi) * sp.integrate(fx*x, (x, 0, R_val))

def tube_volume(R_val, L_val):
    return sp.pi * (R_val**2) * L_val

def back_cone_volume_with_cut(R_val, r_val, lb_val):
    fx_uncut = (-(lb_sym + h)/(R_sym**2))*(x**2) + (lb_sym + h)
    h_sol = sp.solve(sp.Eq(fx_uncut.subs({x: r_sym, lb_sym: lb_sym, R_sym: R_sym}), lb_sym), h)[0]
    fx = sp.simplify(fx_uncut.subs(h, h_sol))
    V1 = (2*sp.pi) * sp.integrate(fx*x, (x, r_sym, R_sym))   # shells
    V2 = (2*sp.pi) * sp.integrate(lb_sym*x, (x, 0, r_sym))   # flat core
    return sp.simplify((V1+V2).subs({R_sym:R_val, r_sym:r_val, lb_sym:lb_val}))

# Exact outer volume (in^3)
Vf_out = front_cone_volume(R_out, lf)
Vm_out = tube_volume(R_out, L)
Vb_out = back_cone_volume_with_cut(R_out, r_tip_out, lb)
V_out_in3 = sp.simplify(Vf_out + Vm_out + Vb_out)

# ---------------------------------
# Equivalent length (matches volume)
# A full-radius cylinder of length L_eq has the same volume as V_out.
# ---------------------------------
A_full = sp.pi * R_out**2             # in^2 (circle area)
L_eq   = sp.simplify(V_out_in3 / A_full)  # in

# ---------------------------------
# Circular-segment relations (tube-dominant hydrostatics)
# T in [0, 2R], alpha = arccos(1 - T/R)
# Segment area: A_seg(T) = R^2 * (alpha - 0.5*sin(2*alpha))
# ---------------------------------
T = sp.symbols("T", real=True)

def A_segment(T_expr, R_val):
    alpha = sp.acos(1 - T_expr/R_val)             # radians
    return (R_val**2) * (alpha - sp.Rational(1,2)*sp.sin(2*alpha))

# Submerged volume as function of T (in^3)
V_sub_in3 = sp.simplify(A_segment(T, R_out) * L_eq)

# ---------------------------------
# Draft solver for given weight W (lbf), freshwater 62.4 lbf/ft^3
# ---------------------------------
gamma = sp.Rational(624,10)  # 62.4
def draft_from_weight(W_lbf, T_guess_in=sp.Rational(12,1)):
    V_req_ft3  = sp.Rational(W_lbf,1) / gamma
    V_req_in3  = V_req_ft3 * (12**3)
    f = sp.lambdify(T, V_sub_in3 - V_req_in3, 'numpy')
    # numeric solve (nsolve with sympy to keep your style)
    T_sol = sp.nsolve(V_sub_in3 - V_req_in3, T, float(T_guess_in))
    return sp.N(T_sol, 6)

# ---------------------------------
# Example: your case W = 570 lbf
# ---------------------------------
W = sp.Rational(570,1)
T_570_in = draft_from_weight(W, T_guess_in=sp.Rational(6,1))

print("Equivalent length L_eq (in):", sp.N(L_eq, 6))
print("Total outer volume V_out (ft^3):", sp.N(V_out_in3/(12**3), 6))
print("Draft for W=570 lbf (in):", T_570_in)

# ---------------------------------
# OPTIONAL: CB vertical below WL for the circular segment at this draft
# y_bar (distance of segment centroid below waterline):
#   with half-angle alpha = arccos(1 - T/R),
#   y_bar = (4*R*sin(alpha)**3) / (3*(2*alpha - sin(2*alpha)))
# ---------------------------------
def cb_vertical_below_wl(T_val_in):
    alpha = sp.acos(1 - T_val_in/R_out)
    ybar  = (4*R_out*sp.sin(alpha)**3) / (3*(2*alpha - sp.sin(2*alpha)))
    return sp.N(ybar, 6)

print("CB vertical below WL at 570 lbf (in):", cb_vertical_below_wl(T_570_in))
