# -*- coding: utf-8 -*-
"""
CALIBRATION AND OPTIMIZATION OF REACTOR 
OBJECTIVE Novo Nordisk has hired ProCon to model and optimize one of their PEGylation reactors. They 
want us to  
 Develop a model and calibrate it to experimental data, see appendix 
 Optimize the reactor 
CONSTRAINTS 
 One batch has to be finished within 20 hours. 
 To stabilize the protein, the minimum amount of Ca is 10 CCa/Cprotein. 

BACKGROUND 
Production of pharmaceutical drugs is nowadays either performed by organic synthesis or 
fermentation with genetically modified yeast or bacteria. The drug must then be purified to 
reach the high safety and efficacy requirements on pharmaceutical drugs. The fermentation or 
synthesis is called upstream processing and the purification is called downstream processing. 
During the downstream processing, the drug is also modified to enhance for example the drugs 
activity, solubility, and half-life in the body. Up to 80 % of the total production cost can stem 
from the downstream processing, making it a good target for advanced process modeling and 
optimization.  

DELIVERABLES Memo that contains the following: 
 presentation of your results of the modelling and calibration, and 
 presentation of your results of the optimization. 


REACTION KINETICS 
The PEGylation of the protein is catalyzed by an enzyme, E. The PEGylation can happen on two 
sites where both sites give an active product. The peptide can also be di-PEGylated which makes 
the protein unwanted. The reaction speed is not site-specific, but the di-PEGylation is slower 
than the first PEGylation, giving k1 = k2 and k3 = k4.  

A is the starting protein, B and C are the sought PEGylated product, D is di-PEGylated peptide and S 
is the reagent. The purity analysis does not distinguish between B and C, and the enzyme is not 
detectable in the purity analysis. 

The Calcium also forms a complex with the enzyme making it inactive according to the following 
equilibrium:  
 
    Enzyme + x * Ca <-> Enzyme^0
 
ECONOMIC DATA 
Price for protein and PEG is the same in $/g, Price for enzyme is 50 times higher.  
EXPERIMENTAL DATA 
Experimental data can be found in the appendix: Stationary experiments with four different molar 
ratios (substrate/reactant), 3 time series for 3 different starting concentrations, and reaction data 
with varying Calcium concentration.  
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
import pandas as pd

df = pd.read_excel("Project3_PEGylation_appendix.xlsx", header=None)


MW = df.iloc[4:7, 1]  # g/mol

# c_Ca0 = df.iloc[3,6] #M
x_E0 = df.iloc[6, 6]  # mole Enzyme/mole protein

c01 = df.iloc[10, 1] / MW[4]  # g/l
tbatch1 = df.iloc[11, 1]  # h
data1 = df.iloc[14:18, 0:4].to_numpy()

c02 = df.iloc[21, 1] / MW[4]  # g/l
x02 = df.iloc[22, 1]  # mole substrate/reactant
data2 = df.iloc[24:30, 0:4].to_numpy()

c03 = df.iloc[35, 1] / MW[4]  # g/l
x03 = df.iloc[36, 1]  # mole substrate/reactant
data3 = df.iloc[39:45, 0:4].to_numpy()

c04 = df.iloc[50, 1] / MW[4]  # g/l
x04 = df.iloc[51, 1]  # mole substrate/reactant
data4 = df.iloc[54:60, 0:4].to_numpy()

c05 = df.iloc[66, 1] / MW[4]  # g/l
x05 = df.iloc[67, 1]  # mole substrate/reactant
x_Ca05 = df.iloc[68, 1] / MW[5]  # mole Ca/mole protein
data5 = df.iloc[71:77, 0:4].to_numpy()

c06 = df.iloc[81, 1] / MW[4]  # g/l
x06 = df.iloc[82, 1]  # mole substrate/reactant
x_Ca06 = df.iloc[83, 1] / MW[5]  # mole Ca/mole protein
data6 = df.iloc[86:92, 0:4].to_numpy()

cA0 = MW[4]
PEG = MW[5]
cE0 = x_E0 / MW[6]
downtime = 2

N = 6
bounds = (0, np.inf)
kguess = np.array([1e6, 1e5, 1e2, 10, 2])  # kBC kD kE Keq x


def objfun(dv, plotFlag=False):
    cA = dv[0]
    tbatch = dv[1]
    cCa = dv[2]

    cout = dynexp(cA, tbatch, cCa, plotFlag)
    yield1 = cout[1]
    prod = yield1 / (tbatch + downtime)
    return -prod


# def objfun2(dv, cA, plotFlag=False):
#     #cA=dv[0]
#     tbatch=dv[0]
#     cCa=dv[1]


#     cout = dynexp(cA,tbatch,cCa, plotFlag)
#     yield1 = 1 - cout[1]/(10/cA0)
#     return yield1


def dynexp(cA, tbatch, cCa, plotFlag=False):
    cBC0 = 0
    cD0 = 0
    cS0 = 1.5 * (10 / cA0)
    cinit = np.array([10 / cA0, cBC0, cD0, cS0])
    tspan = [0, tbatch]

    sol = solve_ivp(lambda t, y: dynmodel(t, y, cA0, cCa), tspan, cinit, method="BDF")

    c = sol.y
    cA = c[0, -1]
    cBC = c[1, -1]
    cD = c[2, -1]

    if plotFlag:
        plt.figure(10)
        plt.plot(sol.t, sol.y.T)

    cout = np.hstack((cA, cBC, cD))
    return cout


def dynmodel(t, y, cA0, cCa):
    # model file
    cA = y[0]
    cBC = y[1]
    cD = y[2]
    cS = y[3]

    Parameters = [
        3.12473462e09,
        1.34799887e09,
        7.59239649e09,
        5.53629568e03,
        2.00000000e00,
    ]

    kBC = Parameters[0]
    kD = Parameters[1]
    kE = Parameters[2]
    Keq = Parameters[3]
    c_Ca0 = 0.0033
    # cE=1
    x = 2
    cE = cE0 / (1 + Keq * c_Ca0**x)
    r_BC = 2 * kBC * cA * cS * cE
    r_D = kD * cS * cBC * cE
    r_E = kE * cE

    dcAdt = -r_BC
    dcBCdt = r_BC - r_D
    dcDdt = r_D
    dcSdt = -r_BC - r_D

    dcdt = np.array([dcAdt, dcBCdt, dcDdt, dcSdt])  # A BC D S Ca

    return dcdt


dvguess = [4e-9, 1, 1]
bounds = ([1e-5, 1e-2], [1, 20], [10, 30])
plotbounds = (1e-5, 1e-2, 1, 20, 10, 30)

print("==== SLSQP ====")
optim1 = {"disp": True, "ftol": 1e-2}
res = minimize(
    objfun,
    dvguess,
    bounds=bounds,
    method="SLSQP",
    jac="2-point",
    options={"finite_diff_rel_step": 1e-4},
)


objfun(res.x, plotFlag=True)

print(f"Optimal [cA0]:{res.x[0]}")
print(f"Optimal [tbatch]:{res.x[1]}")
print(f"Optimal [cCa]/[cA0]:{res.x[2]}")


if True:
    nx, ny = 10, 11  # not equal
    xmin, xmax = (plotbounds[0], plotbounds[1])
    ymin, ymax = (plotbounds[2], plotbounds[3])
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xv, yv)
    z = np.zeros((nx, ny))
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            dvv = np.array([x, y, 10])
            z[i, j] = objfun(dvv)
    X, Y = np.meshgrid(xv, yv)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, -z.T, cmap="jet")
    ax.set_xlabel("cA")
    ax.set_ylabel("Tbatch")
    ax.zlabel("cCa")
