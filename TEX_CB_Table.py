import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# ============================================================
# CG calculator (V and L). Optional transverse too.
# Units:
#   weight: lbm (or lbf consistently)
#   arms:   ft
# Outputs:
#   KG (ft above BL), LCG (ft, +FWD / -AFT relative to midship datum you used)
# ============================================================

@dataclass
class Item:
    name: str
    w: float                 # weight [lbm]
    kg: float                # vertical arm (VCG) [ft above BL]
    lcg: float               # longitudinal arm [ft] (+FWD, -AFT)
    tcg: float = 0.0         # transverse arm [ft] (+STBD, -PORT) optional
    note: str = ""           # optional tag (e.g., "STBD", "PORT", "CTR")

def compute_cg(items: List[Item]) -> Dict[str, float]:
    W = sum(it.w for it in items)
    if W <= 0.0:
        raise ValueError("Total weight must be > 0.")

    Mz = sum(it.w * it.kg  for it in items)   # vertical moment
    Mx = sum(it.w * it.lcg for it in items)   # longitudinal moment
    My = sum(it.w * it.tcg for it in items)   # transverse moment

    KG  = Mz / W
    LCG = Mx / W
    TCG = My / W

    return {
        "W_total": W,
        "Mz_total": Mz,
        "Mx_total": Mx,
        "My_total": My,
        "KG": KG,
        "LCG": LCG,
        "TCG": TCG,
    }

def print_cg_report(items: List[Item], title: str = "CG Report") -> None:
    r = compute_cg(items)
    print(f"=== {title} ===")
    print(f"{'Item':<28} {'W':>10} {'KG':>10} {'VM':>12} {'LCG':>10} {'LM':>12} {'TCG':>10} {'TM':>12}")
    print("-"*116)

    for it in items:
        print(f"{it.name:<28} "
              f"{it.w:>10.2f} {it.kg:>10.3f} {it.w*it.kg:>12.2f} "
              f"{it.lcg:>10.3f} {it.w*it.lcg:>12.2f} "
              f"{it.tcg:>10.3f} {it.w*it.tcg:>12.2f}")

    print("-"*116)
    print(f"{'TOTAL':<28} "
          f"{r['W_total']:>10.2f} {'':>10} {r['Mz_total']:>12.2f} "
          f"{'':>10} {r['Mx_total']:>12.2f} "
          f"{'':>10} {r['My_total']:>12.2f}")
    print()
    print(f"KG  = {r['KG']:.4f} ft (above BL)")
    print(f"LCG = {r['LCG']:.4f} ft (+FWD / -AFT)")
    print(f"TCG = {r['TCG']:.4f} ft (+STBD / -PORT)")

if __name__ == "__main__":
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

    print_cg_report(items, title="TEX Weight & CG Calculation")