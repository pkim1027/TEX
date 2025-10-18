import matplotlib as py
import sympy as sp

# ----------------------------
# Constants
# ----------------------------

# === Center of Buoyancy ===
CBx: float = 
CBy: float = 0
CBz: float = 

# === Center of Gravity ===
CGx: float = 
CGy: float = 0
CGz: float = 

# === Outer Hull Dimensions ===
R_out_in: float = 12.0           # [in] outer radius (24" OD)
t_wall_in: float = 0.25          # [in] wall thickness

# === Section Lengths ===
L_front_in: float = 29.5         # [in] front cone length
L_cyl_in: float   = 182.0        # [in] middle cylindrical length (total tube length)
L_back_in: float  = 31.5         # [in] back cone nominal length
r_tip_out_in: float = 3.5        # [in] outer cut radius at back cone tip
r_tip_in_in: float  = max(0.0, r_tip_out_in - t_wall_in)  # [in] inner cut radius

# === Motor Bay Placement ===
MB_offset_from_back_in: float = 61.5  # [in] distance from start of back cone to bay BACK
MB_length_in: float = 50.0            # [in] sealed motor bay length

# === Hydrostatics ===
gamma_water: float = 62.4        # [lbf/ft^3] freshwater
rho_water: float = gamma_water / 32.174  # [slug/ft^3], for dynamic calcs if needed

# === Reference / Derived Dimensions ===
Total_length_in: float = L_front_in + L_cyl_in + L_back_in  # [in]
Total_length_ft: float = Total_length_in / 12.0             # [ft]
R_out_ft: float = R_out_in / 12.0                           # [ft]
D_out_ft: float = 2 * R_out_ft                              # [ft] outer diameter in ft

# === Common Drafts / Weights for Surface Tests ===
draft_surface_in: float = 12.0      # [in] design surface draft
draft_submerged_in: float = 24.0    # [in] full submergence
Weight_surface_lbf: float = 610.0   # [lbf] total estimated surface weight

# ----------------------------
# Hull Shape
# ----------------------------

# === Front Cone ===





# === Cylinder ===

# === Back Cone ===