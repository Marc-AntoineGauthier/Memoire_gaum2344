# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import optimize
from scipy import interpolate

def integrand(x, gap, T, kb=8.617333262145e-2):
    pre = 1/(np.sqrt(x*x + gap*gap))
    arg = (1/(2*kb*T))*np.sqrt(x*x + gap*gap)
    return pre*np.tanh(arg)
# %%
# here we learnt that the value of NV is actually important to the result
def integral(gap, hw, T):
    return integrate.quad(integrand, 0, hw, args=(gap, T))[0]-(1/2)

hw = (8.617333262145e-2*1.5)/(1.14*np.exp(-(1/2)))
x = np.linspace(0, 2, 200)
y = [integral(i, hw, 1.) for i in x]

plt.plot(x,y)

sol = optimize.brentq(integral, 0., 5, args=(hw, 1.))
print(sol)
# %%
# We want to have the value of NV so that the Tc in hw is the same as the one
# found with the plot. When brentq cannot converge, it means that all of the 
# curve is negative, so there is no more solution. This happens when above Tc.
# We want to have a result at Tc that is the smallest possible so the curve
# is as much as possible like it should be.
#nv_l = np.linspace(0.65, 0.5, 100)  # 0.6303
nv = 0.63152995  # pretty much the limit
print(nv)
def integral(gap, hw, T):
    return integrate.quad(integrand, 0, hw, args=(gap, T))[0]-(1/nv)
hw = (8.617333262145e-2*1.5)/(1.14*np.exp(-1/nv))
T_l = np.linspace(0, 1.5, 1000)[1:]  # 1.594
y = np.zeros(len(T_l))
for i, T in enumerate(T_l):
    y[i] = optimize.brentq(integral, 0., 0.5, args=(hw, T))

inter_func = interpolate.interp1d(T_l, y, fill_value='extrapolate')

plt.plot(T_l, y)
print("BCS : ", 1.76*8.617333262145e-2*1.5)
print("What I found : ", y[0])
print(y[-1])
print(inter_func(1.5))
# %%
# We normalize the temperature and the gap norm
T_norm_l = [t/T_l[-1] for t in T_l]
gap_norm_l = [gap/y[0] for gap in y]

plt.scatter(T_norm_l, gap_norm_l, s=1)
# %%
# saving the data to be used in other programs
np.savetxt(
    "BCS_gap_norm_1000.dat",
    [T_norm_l, gap_norm_l]
)
# %%
# verifying that the save is well done.
fig, ax = plt.subplots(figsize=(10, 8))

data = np.loadtxt("BCS_gap_norm_1000.dat")
x = data[0]
y = data[1]
#point = int(1.48/1.5*1000)
#print(x[point])
#ax.scatter(x[point:],y[point:])
#print(y[-1])
BCS_func = interpolate.interp1d(x, y, fill_value='extrapolate')
x = np.linspace(0, 1.5, 200)
#x = np.linspace(x[point], 1, 200)
y = BCS_func(x/1.5)
#print(y[-1])
ax.scatter(x,y)

def bgap_t(T, Tc=1.5):
    if T>Tc:
        return 0
    else:
        return np.sqrt(1 - T/Tc)

def Fgap_t(T, Tc=1.5, b=3):
    if T>Tc:
        return 0
    else:
        arg = b**(Tc) - b**(T) +1
        logb = np.log(arg)/np.log(b)
    return (1/Tc)*logb
x = np.linspace(0, 1.5, 200)
y = [bgap_t(x1) for x1 in x]
ax.plot(x,y)
y = [Fgap_t(x1, 1.5, 10) for x1 in x]
ax.plot(x,y)
# %%
a = np.array([complex(i,j) for i in range(2) for j in range(2)])
a = a.reshape((2,2))
print(a)
b = np.array([complex(i,j) for i in range(2) for j in range(2)])
b = b.reshape((2,2))
print(np.block([[a, np.zeros((2,2))], [np.zeros((2,2)), b]]))
# %%
