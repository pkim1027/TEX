import sympy as sp

# ----------------------------
# Symbols & Constants
# ----------------------------
R_ft, L_bay_ft, d_ft = sp.symbols("R_ft L_bay_ft d_ft", positive=True)
gamma = sp.nsimplify(62.4)  # lbf/ft^3 (fresh water)

# ----------------------------
# Units
# ----------------------------
def inch_to_ft(x_in):
    return sp.nsimplify(x_in) / 12

def ft_to_in(x_ft):
    return sp.N(12 * x_ft)

# ----------------------------
# Circular segment (submerged) geometry
# ----------------------------
def seg_area(d, R):
    """
    Submerged circular-segment area for a circle of radius R with waterline draft d
    measured from the *bottom* of the circle. Valid for 0 <= d <= 2R.
    A(h) = R^2 * acos((R - h)/R) - (R - h) * sqrt(2*R*h - h^2)
    """
    h = d
    return R**2 * sp.acos((R - h)/R) - (R - h) * sp.sqrt(2*R*h - h**2)

def seg_theta(d, R):
    return 2 * sp.acos((R - d)/R)

def seg_centroid_below_WL(d, R):
    """
    Depth of segment centroid below the waterline (positive downward), useful for hydrostatic moments.
    ybar = (4 R sin^3(θ/2)) / (3 (θ - sin θ)), θ = 2*acos((R-h)/R)
    """
    theta = seg_theta(d, R)
    return (4*R * sp.sin(theta/2)**3) / (3*(theta - sp.sin(theta)))

# ----------------------------
# Buoyant force
# ----------------------------
def buoyant_force_lbf(d_ft_val, R_ft_val, L_bay_ft_val):
    # Guard domain 0..2R
    if not (0 <= float(d_ft_val) <= 2*float(R_ft_val) + 1e-12):
        raise ValueError(f"draft d={d_ft_val} ft must be in [0, 2R]={2*float(R_ft_val)} ft")
    A = seg_area(d_ft_val, R_ft_val)
    V = A * L_bay_ft_val
    return sp.N(gamma * V)

def cb_depth_below_wl_ft(d_ft_val, R_ft_val):
    return sp.N(seg_centroid_below_WL(d_ft_val, R_ft_val))

# ----------------------------
# Pretty printer for a single case
# ----------------------------
def print_natural(R_in=12, L_bay_ft_num=4.0, d_in=12.0):
    R_ft_num = float(inch_to_ft(R_in))
    d_ft_num = float(inch_to_ft(d_in))
    B_lbf = float(buoyant_force_lbf(d_ft_num, R_ft_num, L_bay_ft_num))
    ycb_ft = float(cb_depth_below_wl_ft(d_ft_num, R_ft_num))
    A_ft2 = float(seg_area(d_ft_num, R_ft_num))
    V_ft3 = A_ft2 * L_bay_ft_num
    print("=== Buoyancy: Sealed Motor Bay Only ===")
    print(f"R (in): {R_in:.3f}, L_bay (ft): {L_bay_ft_num:.3f}")
    print(f"Draft at bay, d (in): {d_in:.4f}")
    print(f"Buoyant area A (ft^2): {A_ft2:.6f}")
    print(f"Volume V (ft^3):       {V_ft3:.6f}")
    print(f"Buoyancy B (lbf):      {B_lbf:.6f}")
    print(f"CB below WL (in):      {ft_to_in(ycb_ft):.4f}")
    print("")

# ----------------------------
# Example usage (adjust as needed)
# ----------------------------
if __name__ == "__main__":
    # Example geometry: 24 in OD (R=12 in), 4 ft sealed motor bay
    R_in_default = 12
    L_bay_ft_default = 4.0

    # Example drafts at the motor bay (inches from bottom)
    drafts_in = [12, 24]  # note: 24 in == 2R, full submergence

    for d_in in drafts_in:
        print_natural(R_in=R_in_default, L_bay_ft_num=L_bay_ft_default, d_in=d_in)
