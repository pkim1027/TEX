import sympy as sp

R = 12.0         # tube outer radius [in]
L = 240.0 - 51.0 # tube length shortened by 51 in = 189 in
lf = 29.5        # front cone length [in]
lb = 31.5        # back cone nominal length [in]
r = 3.5      # back cone cut radius [in]

pw = 62.4           # lbf/ft^3 (water weight density)
pd = 0.74           # packing density of balls (solid fraction)
pb = 1.5            # lbf/ft^3 (foam weight density)

x = sp.symbols("x")

# Front Cone
# The front cone's length is 29.5 inches

fx = ((-lf/(R**2))*(x**2)) + lf # Approximation of the front cone's arc

Vf = ((2*sp.pi)*(sp.integrate((fx*x), (x, 0, R))))/(12**3) # Volume of front cone in feet cubed

# Middle Tube

Vm = ((sp.pi)*(R**2)*L)/(12**3) # Volume of middle tube in feet cubed

# Back Cone
# The back cone has a length of 31.5 inches but there is a 7 inch diameter circle at the end.
# We know that this approximate curve has (3.5, 31.5) and (12, 0) as solutions
# h is the extra length that the curve would extend if it was not cut

x, h, lb, R, r = sp.symbols("x, h, lb, R, r") # Set as symbols

fx = (-(lb+h)/(R**2))*(x**2)+(lb+h) # Approximate curve equation without the cut
h_sol = sp.solve(sp.Eq(fx.subs(x, r), lb), h)[0] # Solve for h in terms of lb, R, and r

# Plug in h into the approximate equation
fx = sp.simplify(fx.subs(h, h_sol))

Vb1 = ((2*sp.pi)*(sp.integrate((fx*x), (x, r, R))))/(12**3) # Volume of uncut back cone in feet cubed
Vb2 = ((2*sp.pi)*(sp.integrate((lb*x), (x, 0, r))))/(12**3) # Volume of cut portion of back cone in feet cubed
Vb = sp.simplify(Vb1 + Vb2) # Total volume of cut back cone

vals = {R: 12, r: 3.5, lb: 31.5} # Restate values
Vb = Vb.subs(vals) # Substitute the values into the equation

# Total Volume

V = Vf + Vm + Vb # Add all three sections

V = round(V/sp.pi, 3)*sp.pi # Round to the neearest thousandth
V_N = sp.N(V,5)

# Volume of the motor compartment
R = 1 # foot
Lmc = 4 # feet
Vmc = sp.pi * R**2 * Lmc #feet^3

Vball = sp.N(V_N  - Vmc, 5)
Vball = Vball * pd

fx = 570


print(Vball)