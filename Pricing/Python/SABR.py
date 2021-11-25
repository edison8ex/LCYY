import numpy as np
import pandas as pd
import scipy.stats
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import mean_squared_error

def Blacks_E_Call(F, K, T, vol):
    d1 = (np.log(F/K) + 0.5*(vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - (vol*np.sqrt(T))
    Nd1 = scipy.stats.norm.cdf(d1)
    Nd2 = scipy.stats.norm.cdf(d2)
    V = F*Nd1-K*Nd2
    return V

# Q1a
# Maturity of swaption
T = 5
# Tenor of swap
tau = 10
# Forward Rates
term = 120
f = [0.02 + i * 0.00025 for i in range(0, term)]
# Volatility
gamma = 0.2
# fixed leg dt
fixed_dt = 0.5
# float leg dt
float_dt = 0.25
# Discount Factor
P = [1/(1+float_dt*f[0])]
for i in range(1, term):
    P.append(P[i-1]/(1+float_dt*f[i]))
# Swap Rate
A = 0
for i in list(range(int(T/float_dt), int((T+tau)/float_dt)+1, int(1/fixed_dt))):
    A += P[i]
A *= fixed_dt
R = (P[int(T/float_dt)] - P[int((T+tau)/float_dt)])/A
# Swaption
V0 = A * Blacks_E_Call(R, R, T, gamma)
print("Price for a in-5-to-10 ATM swaption:", V0)

# Q1b
delta = 0.25
gamma = 0.2
MCS = []
for trial in range(0,50):
    f_j = []
    for j in range(0, 120):
        f_j.append([0.02 + 0.00025 * j])
    for t in range(1,120):
        for j in range(t, 120):
            sigma = 0
            rand = np.random.normal(0, 1, 1)[0]
            for i in range(t, j):
                sigma += (f_j[i][t-1]*delta)/(1+f_j[i][t-1]*delta)*(gamma)
            expterm = np.exp((sigma*gamma-(gamma**2)/2)*delta+sigma*np.sqrt(delta)*rand)
            f_j[j].append(f_j[j][t-1] * expterm)
    MCS.append(f_j)
#print(MCS)
K = 0.034124775
value = []
for trial in range(0,50):
    payoff = []
    for swdate in range(20 + 2, 100 + 2, 2):
        tmpdfac = 1
        for tarddate in range(20, swdate, 2):
            tmpdfac *= 1/(1+MCS[trial][tarddate][tarddate]*delta*2)
        payoff.append(max(MCS[trial][swdate][swdate] - K, 0) * delta*2 * tmpdfac)
    b4dis = sum(payoff)
    dfac = 1
    for ddate in range(0, 20, 1):
        dfac *= 1/(1+MCS[trial][ddate][ddate]*delta)
    value.append(b4dis * dfac)
Swaption = sum(value)/len(value)
print("Price for a in-5-to-10 ATM swaption:", Swaption)

def SABR_BlackImpliedVol(F0, K, Tenor, alpha, beta, rho, nu):
    Fm = (F0+K)/2
    beta_s = 1-beta
    epilson = Tenor*(nu**2)
    gamma1 = beta/Fm
    gamma2 = (-1*beta*beta_s)/(Fm**2)
    fin1 = (2*gamma2-(gamma1**2)+1/(Fm**2))/24*((alpha*(Fm**beta)/nu)**2)
    fin2 = (rho*gamma1)/4*(alpha*(Fm**beta)/nu)
    fin3 = (2-3*(rho**2))/24
    hn = fin1+fin2+fin3
    if F0 != K:
        theta = nu/(alpha*beta_s)*(F0**beta_s-K**beta_s)
        D = np.log((np.sqrt(1-2*rho*theta+theta**2)+theta-rho)/(1-rho))
        sigma = nu*np.log(F0/K)*D*(1+hn*epilson)
    else:
        sigma = alpha*(K**beta_s)*(1+hn*epilson)
    return sigma

# Q2
# Params
T = list(range(1, 10+1, 1))
K = [k/100 for k in list(range(1, 8+1, 1))]
term = 120
f = [0.04 for i in range(0, term, 1)]
# discount curve
dt = 0.25
P = [1/(1+dt*f[0])]
for i in range(1, term):
    P.append(P[i-1]/(1+dt*f[i]))
# Loop Params
LenT = len(T)
LenK = len(K)
sigma_B = np.zeros((int(1/dt*LenT), LenK))
cap = np.zeros((LenT, LenK))
# SABR
alpha = 0.03
beta = 0.5
rho = -0.5
nu = 0.25
TenorLoop = 0
for i in range(len(T)):
    #print('i', T[i])
    for j in range(len(K)):
        #print('j', K[j])
        for k in range(int(1/dt*TenorLoop), int(1/dt*T[i])):
            epsilon = ((k+1)*dt)
            sigma_B[k][j] =  SABR_BlackImpliedVol(f[k], K[j], epsilon, alpha, beta, rho, nu)
            cap[i][j] += P[k]*dt*Blacks_E_Call(f[k], K[j], (k+1)*dt, sigma_B[k][j])
            #print('k',k+1,(k+1)*dt,sigma_B[k][j])
        if i < LenT-1:
            cap[i+1][j] = cap[i][j]
        #print(cap[i][j])
    TenorLoop += 1

# Implied Black's Vol
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
# Plot the surface.
X1, Y1 = np.meshgrid(K, list(range(1, int(len(T)*1/dt)+1, 1)))
Z1 = sigma_B
surf = ax.plot_surface(Y1, X1, Z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.invert_xaxis()
ax.view_init(30, -35)
#plt.get_cmap("cividis")

# Cap
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
# Plot the surface.
X1, Y1 = np.meshgrid(K, T)
Z1 = cap
surf = ax.plot_surface(Y1, X1, Z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.invert_xaxis()
ax.view_init(30, -35)

# Q3
SnP = pd.read_csv('UnderlyingOptionsEODCalcs_2020-07-31.csv')
yieldcurve = pd.read_excel('yieldcurve_20200731.xlsx')

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * scipy.stats.norm.cdf(d1) - np.exp(-r * T) * K * scipy.stats.norm.cdf(d2)

def bs_put(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-r * T) * K * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)

def implied_vol_optimizer(sigma, target_value, S, K, T, r, option_type, depth):
    threshold = 1e-3
    precision = 1000
    implied_vol = sigma
    step = 1/(10**(depth))
    for sig in np.arange(sigma-5*step+step, sigma+5*step+step, step):
        price = bs_call(S, K, T, r, sig) if option_type == 'C' else bs_put(S, K, T, r, sig)
        if abs(price - target_value) < precision:
            precision = abs(price - target_value)
            implied_vol = sig
    if (precision <= threshold) or depth > 10:
        #return (implied_vol, precision)
        return implied_vol
    else:
        #return (implied_vol, precision)
        depth += 1
        return implied_vol_optimizer(implied_vol, target_value, S, K, T, r, option_type, depth)
def rate_curve(tenordf, yieldcurve):
    rate = []
    for tn in tenordf:
        if tn <= 1/365:
            rate.append(yieldcurve['1 date'].values[0])
        elif tn <= 30/365:
            rate.append(yieldcurve['1 mo'].values[0])
        elif tn <= 60/365:
            rate.append(yieldcurve['2 mo'].values[0])
        elif tn <= 90/365:
            rate.append(yieldcurve['3 mo'].values[0])
        elif tn <= 180/365:
            rate.append(yieldcurve['6 mo'].values[0])
        elif tn <= 1:
            rate.append(yieldcurve['1 yr'].values[0])
        elif tn <= 2:
            rate.append(yieldcurve['2 yr'].values[0])
        elif tn <= 3:
            rate.append(yieldcurve['3 yr'].values[0])
        elif tn <= 5:
            rate.append(yieldcurve['5 yr'].values[0])
        elif tn <= 7:
            rate.append(yieldcurve['7 yr'].values[0])
        elif tn <= 10:
            rate.append(yieldcurve['10 yr'].values[0])
        elif tn <= 20:
            rate.append(yieldcurve['20 yr'].values[0])
        elif tn <= 30:
            rate.append(yieldcurve['30 yr'].values[0])
    return rate

# date format
SnP['quote_date'] = pd.to_datetime(SnP['quote_date'])
SnP['expiration'] = pd.to_datetime(SnP['expiration'])
# Tenor
SnP['Tenor'] = (SnP['expiration']-SnP['quote_date']+datetime.timedelta(days=1)).dt.days/365
# Rate
SnP['Rate'] = rate_curve(SnP['Tenor'], yieldcurve)
# Mid
SnP['mid_1545'] = (SnP['bid_1545']+SnP['ask_1545'])/2
SnP['underlying_mid_1545'] = (SnP['underlying_bid_1545']+SnP['underlying_ask_1545'])/2
# Moneyness
SnP['Moneyness'] = (SnP['strike']*np.exp(-1*SnP['Rate']*SnP['Tenor']))/SnP['underlying_mid_1545']
# Forward
SnP['underlying_forward_1545'] = SnP['underlying_mid_1545']*np.exp(SnP['Rate']*SnP['Tenor'])
# Implied Vol
SnP['ImpliedVol'] = SnP[['mid_1545', 'underlying_mid_1545', 'strike', 'Tenor', 'Rate', 'option_type']].apply(lambda x: 
                        implied_vol_optimizer(0.5, x[0], x[1], x[2], x[3], x[4], x[5], 1), axis=1)
# Output cache
SnP.to_csv('FormattedUnderlyingOptionsEODCalcs_2020-07-31.csv', index=False)

call = SnP[SnP.option_type == 'C'].reset_index(drop=True)
put = SnP[SnP.option_type == 'P'].reset_index(drop=True)

_outdf = pd.DataFrame(columns=['Date', 'alpha', 'rho', 'nu'])
for curdate in call.expiration.unique():
    # Filtering
    Kfiltered = call[(call.strike <= 4300) & (call.strike >= 3000) 
                     & (call.expiration == pd.Timestamp(curdate))].reset_index(drop=True)
    tar = Kfiltered['ImpliedVol']
    # Optimizer
    looprng = np.arange(0.1, 1, 0.1)
    minMSE = 1e10
    optiParam = (0, 0, 0)
    for a in np.arange(0.01, 0.1, 0.01):
        for r in looprng:
            for v in looprng:
                vol = Kfiltered[['underlying_forward_1545', 'strike', 'Tenor']].apply(lambda x: 
                               SABR_BlackImpliedVol(x[0], x[1], x[2], alpha=a, beta=0.5, rho=r, nu=v), axis=1)
                mse = np.sqrt(mean_squared_error(vol, tar))
                if mse < minMSE:
                    minMSE = mse
                    optiParam = (a, r, v)
    # Plot
    Kfiltered['SABR Implied Vol'] = Kfiltered[['underlying_forward_1545', 'strike', 'Tenor']].apply(lambda x: 
                                       SABR_BlackImpliedVol(x[0], x[1], x[2], alpha=optiParam[0], beta=0.5, 
                                                            rho=optiParam[1], nu=optiParam[2]), axis=1)
    plotdf = Kfiltered[['strike', 'ImpliedVol', 'SABR Implied Vol']]
    plt.plot(plotdf.strike, list(plotdf['ImpliedVol']))
    plt.plot(plotdf.strike, list(plotdf['SABR Implied Vol']))
    plt.show()
    # Append out
    tmpout = pd.DataFrame([pd.to_datetime(curdate).date()] + list(optiParam)).T
    tmpout.columns = ['Date', 'alpha', 'rho', 'nu']
    _outdf = _outdf.append(tmpout)
_outdf = _outdf.reset_index(drop=True)
display(_outdf)
