import sympy as sp

# ----------------------------
# Symbols & Constants
# ----------------------------
# Geometry
R_ft, L_bay_ft = sp.symbols("R_ft L_bay_ft", positive=True)  # Radius of hull (ft), length of sealed bay (ft)
d_ft = sp.symbols("d_ft", nonnegative=True)  # Draft at the sealed bay cross-section (ft), 0..2R

# Loads
W_lbf = sp.symbols("W_lbf", nonnegative=True)  # Weight to support (lbf)

# Water specific weight (fresh)
gamma = sp.nsimplify(62.4)  # lbf/ft^3

# ----------------------------
# Units
# ----------------------------
def inch_to_ft(x_in):
    return sp.nsimplify(x_in) / 12

def ft_to_in(x_ft):
    return sp.N(12 * x_ft)

# ----------------------------
# Circular segment formulas (submerged segment for a circle of radius R_ft with draft d_ft from the bottom)
# ----------------------------
def seg_area(d_ft, R_ft):
    """
    Area of circular segment (ft^2) for a waterline at draft d_ft from the bottom of a circle radius R_ft.
    Valid for 0 <= d <= 2R.
    A(h) = R^2 * acos((R - h)/R) - (R - h) * sqrt(2*R*h - h^2)
    """
    h = d_ft
    R = R_ft
    return R**2 * sp.acos((R - h)/R) - (R - h) * sp.sqrt(2*R*h - h**2)

def seg_theta(d_ft, R_ft):
    """
    Central angle (rad) subtended by the segment (used for centroid).
    θ = 2 * acos((R - h)/R)
    """
    h = d_ft
    R = R_ft
    return 2 * sp.acos((R - h)/R)

def seg_centroid_below_WL(d_ft, R_ft):
    """
    Depth of the *segment centroid below the waterline* (ft). Positive downward from WL.
    ybar = (4 R sin^3(θ/2)) / (3 (θ - sin θ)), where θ = 2*acos((R-h)/R)
    """
    R = R_ft
    theta = seg_theta(d_ft, R_ft)
    return (4*R * sp.sin(theta/2)**3) / (3*(theta - sp.sin(theta)))

# ----------------------------
# Buoyancy & Draft Solvers
# ----------------------------
def buoyant_force_lbf(d_ft_val, R_ft_val, L_bay_ft_val):
    """
    Buoyant force produced by the sealed bay at draft d_ft (lbf).
    B = gamma * A(d) * L
    """
    A = seg_area(d_ft_val, R_ft_val)
    V = A * L_bay_ft_val
    return sp.N(gamma * V)

def solve_draft_for_weight(W_lbf_val, R_ft_val, L_bay_ft_val, guess_in=12):
    # Quick feasibility check: max buoyancy occurs at d = 2R
    B_max = float(62.4 * (sp.pi * R_ft_val**2) * L_bay_ft_val)
    if float(W_lbf_val) > B_max + 1e-9:
        raise ValueError(
            f"Infeasible: W={W_lbf_val} lbf exceeds B_max≈{B_max:.2f} lbf for R={R_ft_val} ft, L={L_bay_ft_val} ft"
        )

    """
    Solve for draft d (ft) such that gamma*A(d)*L = W.
    Uses nsolve with a heuristic initial guess (inches).
    """
    # Clamp guess between a small positive and 2R - small margin
    twoR = 2*R_ft_val
    d_guess_ft = float(max(0.1/12.0, min(twoR - 0.1/12.0, guess_in/12.0)))
    d = sp.symbols("d", nonnegative=True)
    eq = sp.Eq(gamma * seg_area(d, R_ft_val) * L_bay_ft_val, W_lbf_val)
    try:
        sol = sp.nsolve(eq, d, d_guess_ft, tol=1e-14, maxsteps=200)
        # Bound to [0, 2R] just in case of minor numerical drift
        sol = max(0, min(float(sol), float(2*R_ft_val)))
        return sol
    except Exception as e:
        # Try a secondary guess if the first fails (e.g., shallow or deep drafts)
        alt_guesses_in = [2, 6, 10, 16, 20, 22, 24]
        for g in alt_guesses_in:
            try:
                d_guess_ft = float(max(0.1/12.0, min(twoR - 0.1/12.0, g/12.0)))
                sol = sp.nsolve(eq, d, d_guess_ft, tol=1e-14, maxsteps=200)
                sol = max(0, min(float(sol), float(2*R_ft_val)))
                return sol
            except Exception:
                continue
        raise RuntimeError(f"nsolve failed for W={W_lbf_val} lbf with R={R_ft_val} ft, L={L_bay_ft_val} ft. Last error: {e}")

# ----------------------------
# Pretty printer
# ----------------------------
def print_case(W_lbf_num, R_in=12, L_bay_ft_num=4.0, guess_in=12):
    """
    Convenience: R_in in inches (outer radius), L_bay_ft in feet.
    Prints draft at sealed bay, CB depth below WL (local), and buoyant force check.
    """
    R_ft_num = float(inch_to_ft(R_in))
    d_ft_sol = solve_draft_for_weight(W_lbf_num, R_ft_num, L_bay_ft_num, guess_in=guess_in)
    A_ft2 = float(seg_area(d_ft_sol, R_ft_num))
    V_ft3 = A_ft2 * L_bay_ft_num
    B_lbf   = float(gamma * V_ft3)
    ycb_ft  = float(seg_centroid_below_WL(d_ft_sol, R_ft_num))  # below WL (downwards +)
    # Report
    print("=== Sealed Motor Bay Buoyancy ===")
    print(f"R (in): {R_in:.3f}, L_bay (ft): {L_bay_ft_num:.3f}")
    print(f"W (lbf): {W_lbf_num:.3f}")
    print(f"Draft at bay, d (in): {ft_to_in(d_ft_sol):.4f}")
    print(f"CB below WL (in):     {ft_to_in(ycb_ft):.4f}")
    print(f"Buoyant area A (ft^2): {A_ft2:.6f}")
    print(f"Volume V (ft^3):       {V_ft3:.6f}")
    print(f"Buoyancy B (lbf):      {B_lbf:.6f}")
    print("")

# ----------------------------
# Example usage (adjust as needed)
# ----------------------------
if __name__ == "__main__":
    # Geometry: 24 in diameter -> R = 12 in, Sealed motor bay length ~ 4 ft
    R_in_default = 12  # inches
    L_bay_ft_default = 4.0  # feet

    # Example loads to check
    W_list = [570, 700, 760]

    for W in W_list:
        print_case(W, R_in=R_in_default, L_bay_ft_num=L_bay_ft_default, guess_in=16)
