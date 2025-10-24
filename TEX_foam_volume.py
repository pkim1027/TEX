from dataclasses import dataclass

@dataclass
class Foam:
    name: str
    density_lbft3: float  # Foam density (lb/ft³)

# Water density (freshwater)
WATER_DENSITY_LBFT3 = 62.4

def foam_buoyancy(foam: Foam, target_lift_lbf: float):
    """
    Compute the required foam volume, self-weight, and displaced water
    to achieve a given net lift (factoring in foam's own weight).
    """
    rho_f = foam.density_lbft3
    rho_w = WATER_DENSITY_LBFT3
    net_lift_per_ft3 = rho_w - rho_f  # net lift per ft³ (after foam weight)

    # Required foam volume to achieve target lift
    V_ft3 = target_lift_lbf / net_lift_per_ft3

    # Foam self-weight and displaced water
    foam_weight = V_ft3 * rho_f
    water_displaced = V_ft3 * rho_w
    net_lift = water_displaced - foam_weight  # should match target_lift_lbf

    return {
        "foam": foam.name,
        "density_lbft3": rho_f,
        "volume_ft3": V_ft3,
        "foam_weight_lbf": foam_weight,
        "water_displaced_lbf": water_displaced,
        "net_lift_lbf": net_lift
    }

def print_report(result: dict, target_lbf: float):
    print(f"=== {result['foam']} Foam Buoyancy Report ===")
    print(f"Target net lift:            {target_lbf:.1f} lbf")
    print(f"Foam density:               {result['density_lbft3']:.1f} lb/ft³")
    print(f"Required foam volume:       {result['volume_ft3']:.2f} ft³")
    print(f"Foam self-weight:           {result['foam_weight_lbf']:.1f} lbf")
    print(f"Water displaced:            {result['water_displaced_lbf']:.1f} lbf")
    print(f"Net buoyant lift (check):   {result['net_lift_lbf']:.1f} lbf\n")

# -------------------------------------------
# Output
# -------------------------------------------
if __name__ == "__main__":
    target_lift = 600  # desired net buoyant lift (lbf)

    foams = [
        Foam("R3312", 12),
        Foam("XPS Insulation (FOAMULAR F-250)", 2),
        Foam("Volara", 2)
    ]

    for f in foams:
        result = foam_buoyancy(f, target_lift)
        print_report(result, target_lift)
