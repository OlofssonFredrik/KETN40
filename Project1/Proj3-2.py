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
import FVMtools as fvm
df = pd.read_excel('Project3_PEGylation_appendix.xlsx', header=None)



MW = df.iloc[4:7, 1] #g/mol

c_Ca0 = df.iloc[3,6] #M
x_E0 = df.iloc[6,6] #mole Enzyme/mole protein

c01 = df.iloc[10,1]/MW[4] #g/l
tbatch1 = df.iloc[11,1] #h
data1 = df.iloc[14:18, 0:4].to_numpy()

c02 = df.iloc[21,1]/MW[4] #g/l
x02 = df.iloc[22,1] #mole substrate/reactant
data2 = df.iloc[24:30, 0:4].to_numpy()

c03 = df.iloc[35,1]/MW[4] #g/l
x03 = df.iloc[36,1] #mole substrate/reactant
data3 = df.iloc[39:45, 0:4].to_numpy()

c04 = df.iloc[50,1]/MW[4] #g/l
x04 = df.iloc[51,1] #mole substrate/reactant
data4 = df.iloc[54:60, 0:4].to_numpy()

c05 = df.iloc[66,1]/MW[4] #g/l
x05 = df.iloc[67,1] #mole substrate/reactant
x_Ca05 = df.iloc[68,1]/MW[5] #mole Ca/mole protein
data5 = df.iloc[71:77, 0:4].to_numpy()

c06 = df.iloc[81,1]/MW[4] #g/l
x06 = df.iloc[82,1] #mole substrate/reactant
x_Ca06 = df.iloc[83,1]/MW[5] #mole Ca/mole protein
data6 = df.iloc[86:92, 0:4].to_numpy()

cA0 = MW[4]
PEG = MW[5]
cE0 = x_E0/MW[6]


N=6
bounds = (0,np.inf)
kguess=np.array([1e7,1e6,1e6,10,10])

#okända kBC kD cBC cD 

def dyn2(tdyn,cexp):
     # kBC kD Keq x
    kest, pcov = curve_fit(lambda tdyn, kBC, kD, kE, Keq,x:\
                           dynexp(tdyn, kBC, kD, kE, Keq,x,),\
                               tdyn, cexp, kguess,\
                                   bounds=bounds)
    
    
    cmod = dynexp(tdyn, *kest)
    res = cexp-cmod 
    Q = np.sum(res**2) 
    std = np.sqrt(Q/(len(tdyn)-1)) 
     
    plt.plot(tdyn,cexp[:N],'x', markersize=3) 
    plt.plot(tdyn,cexp[N:2*N],'x', markersize=3)
    plt.plot(tdyn,cexp[2*N:3*N],'x', markersize=3)
    plt.plot(tdyn,cexp[3*N:4*N],'.', markersize=3)
    plt.plot(tdyn,cexp[4*N:5*N],'.', markersize=3)
    plt.plot(tdyn,cexp[5*N:6*N],'.', markersize=3)
    plt.plot(tdyn,cexp[6*N:7*N],'.', markersize=3)
    plt.plot(tdyn,cexp[7*N:8*N],'.', markersize=3)
    plt.plot(tdyn,cexp[8*N:9*N],'.', markersize=3)
    plt.plot(tdyn,cexp[9*N:10*N],'.', markersize=3)
    plt.plot(tdyn,cexp[10*N:11*N],'.', markersize=3)
    plt.plot(tdyn,cexp[11*N:12*N],'.', markersize=3)
    plt.plot(tdyn,cexp[12*N:13*N],'.', markersize=3)
    plt.plot(tdyn,cexp[13*N:14*N],'.', markersize=3)
    plt.plot(tdyn,cexp[14*N:15*N],'.', markersize=3)
    plt.plot(tdyn,cmod[:N])
    plt.plot(tdyn,cmod[N:2*N])
    plt.plot(tdyn,cmod[2*N:3*N]) 
    plt.plot(tdyn,cmod[3*N:4*N]) 
    plt.plot(tdyn,cmod[4*N:5*N])
    plt.plot(tdyn,cmod[5*N:6*N])
    plt.plot(tdyn,cmod[6*N:7*N])
    plt.plot(tdyn,cmod[7*N:8*N])
    plt.plot(tdyn,cmod[8*N:9*N])
    plt.plot(tdyn,cmod[9*N:10*N])
    plt.plot(tdyn,cmod[10*N:11*N])
    plt.plot(tdyn,cmod[11*N:12*N])
    plt.plot(tdyn,cmod[12*N:13*N])
    plt.plot(tdyn,cmod[13*N:14*N])
    plt.plot(tdyn,cmod[14*N:15*N])
    
    
 
    print('===Parameters===\n'f'kBC = {kest[0]} \nkD = {kest[1]} \nkE = {kest[2]} \nKeq = {kest[3]} \nx = {kest[4]} \nstd: {std}') 
  
    return kest,std 

def dynexp(tdyn,kBC, kD, kE, Keq, x):
    cinit2 = np.array([c02,0,0,c02*x02, c_Ca0]) # A BC D S Ca
    cinit3 = np.array([c03,0,0,c03*x02, c_Ca0]) # A BC D S Ca
    cinit4 = np.array([c04,0,0,c04*x02, c_Ca0]) # A BC D S Ca
    cinit5 = np.array([c05,0,0,c05*x02, x_Ca05*c_Ca0]) # A BC D S Ca
    cinit6 = np.array([c06,0,0,c06*x02, x_Ca06*c_Ca0]) # A BC D S Ca
    tspan = (tdyn2[0], tdyn2[-1])
    
    sol = solve_ivp(lambda t, y: dynmodel(t,y,kBC, kD,kE,Keq,x),\
                    tspan, cinit2, method = 'BDF', t_eval=tdyn) 
        
    cmod2 = np.hstack(([sol.y[0,:], sol.y[1,:],sol.y[2,:]]))/c02 # make it a 1d array 
    
    sol = solve_ivp(lambda t, y: dynmodel(t,y,kBC, kD, kE,Keq,x),\
                    tspan, cinit3, method = 'BDF', t_eval=tdyn) 
    cmod3 = np.hstack(([sol.y[0,:], sol.y[1,:],sol.y[2,:]]))/c03   
    
    sol = solve_ivp(lambda t, y: dynmodel(t,y,kBC, kD, kE,Keq,x),\
                    tspan, cinit4, method = 'BDF', t_eval=tdyn) 
    cmod4 = np.hstack(([sol.y[0,:], sol.y[1,:],sol.y[2,:]]))/c04
        
    sol = solve_ivp(lambda t, y: dynmodel(t,y,kBC, kD, kE,Keq,x),\
                    tspan, cinit5, method = 'BDF', t_eval=tdyn) 
    
    cmod5 = np.hstack(([sol.y[0,:], sol.y[1,:],sol.y[2,:]]))/c05
    
    sol = solve_ivp(lambda t, y: dynmodel(t,y,kBC, kD,kE,Keq,x),\
                    tspan, cinit6, method = 'BDF', t_eval=tdyn) 
    
    cmod6 = np.hstack(([sol.y[0,:], sol.y[1,:],sol.y[2,:]]))/c06
    
    
    cmod = np.hstack((cmod2,cmod3, cmod4,cmod5,cmod6))
    
    return cmod 
     
def dynmodel(t,y, kBC,kD, kE, Keq, x): 
    # model file
    cA = y[0]
    cBC = y[1]
    cD = y[2]
    cS = y[3]
    

    #cE=1
    
    cE = cE0/(1+Keq*c_Ca0**x)
    r_BC = 2*kBC*cA*cS*cE
    r_D = kD*cS*cBC*cE
    r_E = kE*cE 
    
    dcAdt = -r_BC
    dcBCdt = r_BC - r_D
    dcDdt = r_D
    dcSdt = -r_BC - r_D
    dcCadt = - r_E
    
    
    dcdt = np.array([dcAdt, dcBCdt, dcDdt, dcSdt, dcCadt]) #A BC D S 
    
    return dcdt 

tdyn2 = np.hstack((data2[:,0])) #, data2[:,0], data2[:,0], data2[:,0], data2[:,0], data2[:,0]))
data2T = np.hstack((data2[:,1:].T,\
                    data3[:,1:].T,\
                        data4[:,1:].T,\
                            data5[:,1:].T,\
                                data6[:,1:].T))
cexp2 = np.hstack((data2T))
kest = dyn2(tdyn2,cexp2)

#plt.ylim(0,0.9)
#plt.title('data6')
ax = plt.subplot(111)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(['A2', 'BC2', 'D2' , 'A3', 'BC3', 'D3',  'ModA2', 'ModBC2', 'ModD2',  'ModA3', 'ModBC3', 'ModD3'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time(h)')
plt.ylabel('Molar fraction (%)')
plt.title('Wrong')
plt.show()
