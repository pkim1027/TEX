import sympy as sp

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

# ----------------------------
# Symbols
# ----------------------------
x, h, lb_sym, R_sym, r_sym = sp.symbols("x h lb R r", positive=True)

# ----------------------------
# Volumes (outer only, what water "sees")
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

Vf_out = front_cone_volume(R_out, lf)
Vm_out = tube_volume(R_out, L)
Vb_out = back_cone_volume_with_cut(R_out, r_tip_out, lb)
V_out_in3 = sp.simplify(Vf_out + Vm_out + Vb_out)
V_out_ft3 = sp.simplify(V_out_in3 / (12**3))

# ----------------------------
# Weight required for T = 12 in (half-volume displaced)
# ----------------------------
gamma = sp.Rational(624,10)  # 62.4 lbf/ft^3
W_T12 = sp.simplify(gamma * (sp.Rational(1,2) * V_out_ft3))

print("Required weight for draft T = 12 in (lbf):", sp.N(W_T12, 3))

# (Optional) CB vertical below WL at T=12 in for a circle of radius 12 in:
# y_bar = 4R/(3π)
y_bar = sp.simplify(4*R_out/(3*sp.pi))
print("CB vertical below WL at T=12 in (in):", sp.N(y_bar, 6))
