
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp 
import FVMtools as fvm 
import time 


H0 = [1.2e-5, 3.6e-6, 3.0e-4]
H0A = H0[0]
H0BC = H0[1]
H0D = H0[2]

beta = [8.41, 8.41, 4.57]
betaA = beta[0]
betaBC = beta[1]
betaD = beta[2]


qmax = 200 # for A, B and C

F = 5 #CV/h

kkin = np.array([15, 18, 43.4])/F
kkinA = kkin[0]
kkinBC = kkin[1]
kkinD = kkin[2]

M = 58.44
TotPep = 5/1
cAin = 0.169*TotPep
cBCin = 0.662*TotPep
cDin = 0.169*TotPep

SF = 0.10
SA = 0.05
SB = 0.35

ec = 0.42 
ep = 0.62 
L = 10e-2 #m

v = L #m/CV

Pe = 0.5 
Dp = 90e-6

# relations 
ee = ec + (1-ec)*ep 

Dax = (v*Dp)/Pe

N = 20

h = L/N

Ss = []
ts = []


def adsmodel(t, cq): 
    N  =  int(len(cq)/6) 
    cA = cq[:N]
    cBC = cq[N:2*N]
    cD = cq[2*N:3*N]
    qA = cq[3*N:4*N]
    qBC = cq[4*N:5*N]
    qD = cq[5*N:6*N]
    

 
    if t<0:
        cinA = cAin
        cinBC = cBCin
        cinD = cDin
    else:
        cinA = 0
        cinBC = 0
        cinD = 0
    
    if t<0 : #Load
        S = SF 
    elif t<1: #Wash
        S = SA
    elif t<51 : #Elution
        Bstart = 0.1
        Bx = 0.01*(t-1) + Bstart
        S = Bx*SB+(1-Bx)*SA
    else: #Regeneration
        S = SB
        
    Ss.append(S)
    ts.append(t)
    HA = H0A*S**-betaA
    HBC = H0BC*S**-betaBC
    HD = H0D*S**-betaD
    
    qtot = qA+qBC+qD
    
    # adsorption 
    radsA = kkinA*(HA*cA*(1-(1/qmax)*qtot))-qA
    radsBC = kkinBC*(HBC*cBC*(1-(1/qmax)*qtot))-qBC
    radsD = kkinD*(HD*cD*(1-(1/qmax)*qtot))-qD
 
    # discr 
    [A2,A2f] = fvm.FVMdisc2nd(N, h, '3pc') 
    [A1,A1f] = fvm.FVMdisc1st(N, h, '2pb') 
    [B1,B0] = fvm.FVMdiscBV(N, h, [0, 1], [[1, -1],[0, 0]]) 
 
    # assembly 
    dcAdt = Dax*(A2@cA + A2f@(B1@cA + B0*cinA)) - v/ee*(A1@cA + A1f@(B1@cA + B0*cinA)) - (1-ec)/ec*radsA
    dcBCdt = Dax*(A2@cBC + A2f@(B1@cBC + B0*cinBC)) - v/ee*(A1@cBC + A1f@(B1@cBC + B0*cinBC)) - (1-ec)/ec*radsBC
    dcDdt = Dax*(A2@cD + A2f@(B1@cD + B0*cinD)) - v/ee*(A1@cD + A1f@(B1@cD + B0*cinD)) - (1-ec)/ec*radsD
   
    dcdt = np.hstack((dcAdt, dcBCdt, dcDdt))
    
    dqAdt = radsA
    dqBCdt = radsBC
    dqDdt = radsD
    
    dqdt = np.hstack((dqAdt, dqBCdt, dqDdt))
    
    dcqdt = np.hstack((dcdt, dqdt)) 
    return dcqdt 

def adssim():
    tic = time.time()
    
    tspan = [-0.5,56]
    cqinit = np.hstack((0*np.ones(N),0*np.ones(N),0*np.ones(N), 0*np.ones(N), 0*np.ones(N), 0*np.ones(N) ))
    sol = solve_ivp(lambda t, cq: adsmodel(t, cq), tspan, cqinit, method='BDF')
    t = sol.t
    cA = sol.y[N-1,:] 
    cBC = sol.y[2*N-1,:]
    cD = sol.y[3*N-1,:]
    
    c = np.vstack((cA,cBC,cD)).T
    

  
    
    
   
    tac = time.time()
    
    plt.figure()
    plt.plot(t,cA)
    plt.plot(t,cBC)
    plt.plot(t,cD)
    
adssim()
plt.plot(ts,Ss,'.')