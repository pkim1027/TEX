import sympy as sp

# This will look at the buoyant force of the styrofoam balls that will be inside TEX. 
pd = 0.74 # Packing density of 1.5 inch diameter styrofoam balls
p = 1.5 # lbs/ft^3, density of the styrofoam balls
pw = 62.4 # lbs/ft^3, density of water in the scope of project
V = 57.815 # ft^3, approximate volume of TEX
Vf = 3.8615 # ft^3, approximate volume of front cone compartment
Vb = 4.4741 # ft^3, approximate volume of back cone compartment
Vc = Vf + Vb # Volume of the cone compartments

# We are thinking of compartmentalizing TEX into compartments.
# Two cone compartment, two ball compartments, one motor compartment.
# The two ball compartments will be on either ends of the middle tube. 

Vm = 63*sp.pi/4 # Volume of the tube
Vtm = 4*sp.pi # Approximate volume of the motor compartment
Vtb = Vm - Vtm

# If we filled both compartments completely full of styrofoam balls, 
# 74% of it would be filled volume-wise.

real_Vtb = Vtb*pd # Actual volume of the balls in the compartment
Fbb = real_Vtb*pw # Buoyant force of the ball compartment if filled all the way with balls and water.

Fbc = pw*Vtm # Displaced volume of cone compartment if fully submerged
Fbm = pw*Vc # Displaced volume of motor compartment if fully submerged

# Total buoyant force of TEX if fully submerged
TFb = Fbb + Fbc + Fbm
print(sp.N(TFb, 7), "lbf")

