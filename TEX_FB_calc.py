# TEX RSV buoyancy vs foam fraction with aligned axes and equilibrium waterlines
# Self‑contained script. Produces ONE large graph with:
#  - Buoyancy envelope (min/max F over waterline h ∈ [−R, +R])
#  - Horizontal dashed weight lines for 570–3000 lbf
#  - Equilibrium waterline curves h(B) for each weight (free‑flooding ball bays; sealed other sections)
#  - Aligned right axis: −10 in ↔ 0 lbf and +10 in ↔ 3500 lbf; right‑axis ticks mapped from left axis
#
# Notes:
#  * Uses matplotlib only (no seaborn, default colors).
#  * Change WEIGHTS list or alignment map as needed.
# Fully Submerged RSV: Buoyant Force vs Foam Fraction
# Assumptions:
# - Entire RSV ~6 ft underwater
# - Only motor compartment is sealed (volume Vtm)
# - All other sections are free-flooding and filled with balls (packing density pd)
# Model: F(B) = pw * ( Vtm + pd * B * V_ff_total )

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Geometry & constants
# ----------------------------
R_in   = 12.0         # tube radius [in]
L_in   = 240.0        # full tube length [in] (used only to get V_tb with motor volume removed)
lf_in  = 29.5         # front cone length [in]
lb_in  = 31.5         # back cone nominal length [in]
r_tip_in = 3.5        # back cone cut radius at tip [in]
in_to_ft = 1.0/12.0

R   = R_in*in_to_ft
lf  = lf_in*in_to_ft
lb  = lb_in*in_to_ft
r_tip = r_tip_in*in_to_ft

pw  = 62.4           # lbf/ft^3
p_d = 0.74           # packing density of balls

# Volumes
Vm_total = np.pi * R**2 * (L_in*in_to_ft)  # whole tube volume [ft^3]
Vtm      = 4*np.pi                          # sealed motor compartment volume [ft^3] (given)
Vtb      = Vm_total - Vtm                   # ball compartments geometric volume [ft^3]

# Motor sealed tube length (so that its volume is Vtm)
Lm = Vtm / (np.pi * R**2)

# Back‑cone paraboloid extension (so that r(lb)=r_tip)
h_extra = ( (r_tip**2) / (R**2 - r_tip**2) ) * lb
lb_tot  = lb + h_extra

# Radius profiles
def r_front(y):         # y ∈ [0, lf]
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lf))

def r_back(y):          # y ∈ [0, lb]
    return R * np.sqrt(np.maximum(0.0, 1.0 - y/lb_tot))

def r_tube_motor(y):    # y ∈ [0, Lm]
    return R * np.ones_like(y)

# Circular‑segment submerged area for circle radius a at waterline offset h_w (ft)
def A_submerged(a, h_w):
    a = np.asarray(a, dtype=float)
    h = np.broadcast_to(np.asarray(h_w, dtype=float), a.shape)
    A = np.zeros_like(a, dtype=float)
    dry = h <= -a
    wet = h >=  a
    mid = (~dry) & (~wet)
    # normalized offset
    z = np.empty_like(a, dtype=float)
    z[mid] = np.clip(h[mid]/a[mid], -1.0, 1.0)
    # segment area
    A[mid] = (a[mid]**2) * (np.pi - np.arccos(z[mid]) + z[mid]*np.sqrt(1 - z[mid]**2))
    A[wet] = np.pi * a[wet]**2
    A[dry] = 0.0
    return A

# Displaced volume for sealed sections (front cone + motor tube + back cone) at waterline h_ft
def Vdisp_other(h_ft):
    Ny = 6001
    y_f = np.linspace(0,  lf, Ny)
    y_m = np.linspace(0,  Lm, Ny)
    y_b = np.linspace(0,  lb, Ny)
    Af = A_submerged(r_front(y_f), h_ft)
    Am = A_submerged(r_tube_motor(y_m), h_ft)
    Ab = A_submerged(r_back(y_b),  h_ft)
    return np.trapz(Af, y_f) + np.trapz(Am, y_m) + np.trapz(Ab, y_b)

# Total buoyant force with free‑flooding ball bays (W=1): F(B,h) = pw * [Vdisp_other(h) + Vtb * p_d * B]
def F_total(B, h_in):
    return pw * ( Vdisp_other(h_in*in_to_ft) + Vtb * p_d * B )

# ----------------------------
# Build envelope vs B
# ----------------------------
B_grid = np.linspace(0.0, 1.0, 61)
h_grid_in = np.linspace(-R_in, R_in, 241)
Vother_vec = np.array([Vdisp_other(h*in_to_ft) for h in h_grid_in])

Fmin_vs_B = []
Fmax_vs_B = []
for B in B_grid:
    F_vals = pw*(Vother_vec + Vtb * p_d * B)
    Fmin_vs_B.append(F_vals.min())
    Fmax_vs_B.append(F_vals.max())

Fmin_vs_B = np.array(Fmin_vs_B)
Fmax_vs_B = np.array(Fmax_vs_B)

# ----------------------------
# Equilibrium waterline curves h(B) for selected weights
# ----------------------------
WEIGHTS = [570, 800, 1000, 1200, 1500, 2000, 2500, 3000]  # lbf

def solve_h_for_equilibrium(B, W, h_min_in=-R_in, h_max_in=R_in, tol=1e-3, maxit=60):
    a, b = h_min_in, h_max_in
    Fa = F_total(B, a) - W
    Fb = F_total(B, b) - W
    if Fa>0 and Fb>0:    # too buoyant for bracket
        return np.nan
    if Fa<0 and Fb<0:    # not enough buoyancy in bracket
        return np.nan
    for _ in range(maxit):
        c  = 0.5*(a+b)
        Fc = F_total(B, c) - W
        if abs(Fc) < 1e-2 or (b-a)/2 < tol:
            return c
        if np.sign(Fc) == np.sign(Fa):
            a, Fa = c, Fc
        else:
            b, Fb = c, Fc
    return c

h_curves = [np.array([solve_h_for_equilibrium(B, W) for B in B_grid]) for W in WEIGHTS]

# ----------------------------
# Aligned axes mapping and plotting
# ----------------------------
# Alignment: map 0 lbf ↔ −10 in and 3500 lbf ↔ +10 in
F0, F1 = 0.0, 3500.0
h0, h1 = -10.0, 10.0

def F_to_h(F):
    return h0 + (F - F0) * (h1 - h0) / (F1 - F0)

fig, ax1 = plt.subplots(figsize=(20, 8))  # large canvas


# Buoyancy envelope (min/max over waterline)
ax1.plot(B_grid, Fmin_vs_B, label="Min F over h∈[−R,+R]")
ax1.plot(B_grid, Fmax_vs_B, label="Max F over h∈[−R,+R]")

# Horizontal dashed weight lines
for W in WEIGHTS:
    ax1.hlines(W, 0, 1, linestyles="dashed", linewidth=1, label=f"W={W} lbf")

ax1.set_xlabel("Balls fraction B (0–1)", fontsize=12)
ax1.set_ylabel("Buoyant force F (lbf)", fontsize=12)
ax1.set_ylim(F0, F1)
ax1.set_yticks(np.arange(0, 3501, 500))
ax1.set_xlim(0, 1)                       # still 0 → 1
ax1.set_xticks(np.linspace(0, 1, 11))     # e.g. 5 ticks at 0, 0.25, 0.5, 0.75, 1
ax1.grid(True)

# Right axis: equilibrium waterline curves (default color cycle)
ax2 = ax1.twinx()
for hs, W in zip(h_curves, WEIGHTS):
    ax2.plot(B_grid, hs, linewidth=2, alpha=0.9, label=f"h(B), W={W} lbf")

ax2.set_ylabel("Equilibrium waterline h (inches)", fontsize=12)
ax2.set_ylim(h0, h1)
# Map right‑axis ticks from left‑axis force ticks
h_ticks = F_to_h(ax1.get_yticks())
ax2.set_yticks(h_ticks)
ax2.set_yticklabels([f"{htick:.1f}" for htick in h_ticks])

# Legend outside, with extra right margin
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2,
           loc="center left", bbox_to_anchor=(1.1, 0.5),
           frameon=True, fontsize=10)

fig.suptitle("TEX RSV: Buoyancy Envelope & Equilibrium Waterline vs Foam Fraction\n",
             fontsize=14, fontweight="bold")

fig.tight_layout(rect=[0,0,0.73,1])
plt.subplots_adjust(left=0.12, right=0.7) 
plt.show()
# Geometry / constants (ft^3)
V_total   = 57.815
Vtm       = 4*np.pi                # sealed motor volume
pd        = 0.74                   # packing density
pw        = 62.4                   # lbf/ft^3

V_ff_total = V_total - Vtm         # free-flooding total volume

# Line form: F(B) = intercept + slope * B
intercept = pw * Vtm
slope     = pw * pd * V_ff_total

# Candidate weights
weights = [570, 800, 1000, 1200, 1500, 2000, 2500, 3000]

# Minimum foam fraction for each W (clip to [0,1] for plotting)
def B_min_for_W(W):
    return (W - intercept) / slope

Bmins = [max(0.0, min(1.0, B_min_for_W(W))) for W in weights]

# ----- Print table -----
header = f"{'Weight (lbf)':>12} | {'B_min':>6} | {'Foam vol (ft^3)':>15} | {'Check F(B_min) (lbf)':>20}"
print(header)
print("-"*len(header))
for W, Bm in zip(weights, Bmins):
    foam_vol = pd * V_ff_total * Bm
    F_check  = intercept + slope * Bm
    print(f"{W:12.0f} | {Bm:6.3f} | {foam_vol:15.2f} | {F_check:20.1f}")

# ----- Graph -----
B = np.linspace(0, 1, 400)
F_line = intercept + slope * B

plt.figure(figsize=(9,5))
plt.plot(B, F_line, linewidth=2, label="F(B) = 62.4 [ Vtm + pd * B * V_ff ]")

# Weight lines + intersection markers
for W, Bm in zip(weights, Bmins):
    plt.hlines(W, 0, 1, linestyles="dashed", alpha=0.6)
    if 0.0 <= Bm <= 1.0:
        plt.plot(Bm, W, marker='o')
        plt.text(Bm, W, f"  B={Bm:.2f}", va='bottom', ha='left', fontsize=9)

plt.xlabel("Balls fraction B (0–1)")
plt.ylabel("Buoyant force F (lbf)")
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.title("Fully Submerged RSV: Minimum Foam Fraction vs Weight\n(sealed motor only; others free-flooding)")
plt.grid(True, axis="y", linestyle=":", linewidth=0.8)
plt.tight_layout()
plt.show()

print("\nLine parameters:")
print(f"  Intercept F(B=0) = {intercept:.1f} lbf (from Vtm)")
print(f"  Slope dF/dB      = {slope:.1f} lbf per unit B")
print(f"  Max F at B=1     = {intercept + slope:.1f} lbf")
