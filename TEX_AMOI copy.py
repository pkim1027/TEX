import numpy as np

# Area Moment of Inertia Calculations
# Rough calculation of moment of inertia using cylinders

total_vol = 57.7507 #ft^3
motor_vol = 13.09   #ft^3
front_vol = (total_vol-motor_vol)/2
back_vol = front_vol

d = 2   #ft
r = 1   #ft
# cylinder = pi*r^2 * h

#Roll
def rmoi(M,R):
    Ixx = (M*(R**2))/2
    return

#Pitch
def pmoi(M,R,L):
    Iyy = (M*(R**2))/4 + (M*(L**2))/12
    return 

#Lengths of Sections
mot_L = motor_vol/(np.pi * r**2)
front_L = front_vol/(np.pi * r**2)
back_L = back_vol/(np.pi * r**2)

tot_L = mot_L+front_L+back_L

if tot_L != 20:
    cylin_L = 20 - mot_L
    front_L = cylin_L / 2
    back_L = cylin_L / 2

tot_L = mot_L+front_L+back_L

front_lb = 21.25*front_L      #lbf
back_lb = 21.25*back_L      #lbf
motor_lb = 21.25*mot_L + 200   #lbf
g = 32.2            #ft/s^2

frm = front_lb/g    #lbm
bkm = back_lb/g     #lbm
mtm = motor_lb/g    #lbm

def front(M, L):
    fIxx = rmoi(M, L)
    fIyy = pmoi(M, 1, L)
    fl = L/2 + mot_L/2
    return fIxx, fIyy, fl

def back(M, L):
    bIxx = rmoi(M, L)
    bIyy = pmoi(M, 1, L)
    bl = L/2 + mot_L/2
    return bIxx, bIyy, bl

def mot(M, L):
    mIxx = rmoi(M, L)
    mIyy = pmoi(M, 1, L)
    return mIxx, mIyy

def Ixx(fI, bI, mI):
    tot_Ixx = fI + bI + mI
    return tot_Ixx

def Iyy(fI, fl, bI, bl, mI):
    fIyy = (fI + m*(fl**2)) + (bI+m*(bl**2)) + mI
