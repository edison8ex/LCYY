import math
import numpy as np
from scipy.stats import norm
delta = 0.25
gamma = 0.2
trials = 10000
option_m = (5) * 4
bond_m = option_m + (10) * 4
MCS = []
for trial in range(0,trials):
    f_j = []
    for j in range(0, bond_m+1):
        f_j.append([0.02 + 0.00025 * j])
    for t in range(1,bond_m+1):
        for j in range(t, bond_m+1):
            sigma = 0
            rand = np.random.normal(0, 1, 1)[0]
            for i in range(t, j):
                sigma += (f_j[i][t-1]*delta)/(1+f_j[i][t-1]*delta)*(gamma)
            sigma *= -1
            expterm = math.exp(-(sigma*gamma+(gamma**2)/2)*delta+gamma*math.sqrt(delta)*rand)
            f_j[j].append(f_j[j][t-1] * expterm)
    MCS.append(f_j)
value = []
for trial in range(0,trials):
    tmpdfac = 1
    disFac = []
    Anu = 0
    for tarddate in range(option_m, bond_m-1, 1):
        tmpdfac *= 1/(1+MCS[trial][tarddate][tarddate]*delta)
        disFac.append(tmpdfac)
    for df in range(0, bond_m-option_m-2, 2):
        Anu += disFac[df] * delta*2
    dfac = 1
    for ddate in range(0, option_m-1, 1):
        dfac *= 1/(1+MCS[trial][ddate][ddate]*delta)
    swapRate = (1 - tmpdfac)/(Anu)
    K = swapRate
    Anu = (Anu)*dfac
    d1 = (math.log(swapRate/K) + 0.5*(gamma**2)*option_m*delta)/(gamma*math.sqrt(option_m*delta))
    d2 = d1-(gamma*math.sqrt(option_m*delta))
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    V = (swapRate*Nd1-K*Nd2) * Anu
    value.append(V)
Swaption = sum(value)/len(value)
Swaption