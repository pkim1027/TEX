# Number of 1.5-inch balls needed to achieve B_min for each weight
# Uses same assumptions/constants as the F(B) script.

import numpy as np
import matplotlib.pyplot as plt

# Constants (must match the previous script)
V_total   = 57.815
Vtm       = 4*np.pi
pd        = 0.74
pw        = 62.4

V_ff_total = V_total - Vtm
intercept = pw * Vtm
slope     = pw * pd * V_ff_total

weights = [570, 800, 1000, 1200, 1500, 2000, 2500, 3000]

def B_min_for_W(W):
    return (W - intercept) / slope

Bmins = [max(0.0, min(1.0, B_min_for_W(W))) for W in weights]

# Ball geometry: diameter = 1.5 in ⇒ radius = 0.75 in = 0.0625 ft
r_ft = 0.75 / 12.0
V_ball = (4.0/3.0) * np.pi * r_ft**3   # ≈ 0.001021 ft^3

# Solid foam volume required and number of balls
foam_volumes = [pd * V_ff_total * Bm for Bm in Bmins]       # ft^3 of solid foam
ball_counts  = [int(round(Vf / V_ball)) for Vf in foam_volumes]

# ----- Print table -----
header = f"{'Weight (lbf)':>12} | {'B_min':>6} | {'Foam vol (ft^3)':>15} | {'Ball volume (ft^3)':>17} | {'# Balls':>8}"
print(header)
print("-"*len(header))
for W, Bm, Vf, N in zip(weights, Bmins, foam_volumes, ball_counts):
    print(f"{W:12.0f} | {Bm:6.3f} | {Vf:15.2f} | {V_ball:17.6f} | {N:8d}")

# ----- Graph: number of balls vs weight -----
plt.figure(figsize=(9,5))
plt.plot(weights, ball_counts, marker="o", linewidth=2)
plt.xlabel("RSV weight (lbf)")
plt.ylabel("# of 1.5-inch balls")
plt.title("Balls Required for Neutral Buoyancy at B_min(W)\n(fully submerged; sealed motor only)")
plt.grid(True, linestyle=":", linewidth=0.8)
plt.tight_layout()
plt.show()
