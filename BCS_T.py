import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

def integrand(u, d, t):
    pre = 1/np.sqrt(u*u + 1)
    arg = (0.882*d*np.sqrt(u*u + 1))/(t)
    return pre*np.tanh(arg)

def integral(t, d):
    # smiley should be passed as arg
    res = (
        sp.integrate.quad(integrand, 0, np.sinh(smiley)/d, args=(d, t))[0]
        -smiley
    )
    return res

smiley = 1/0.63152995

fig, ax = plt.subplots(figsize=(10, 6))

# here we fix d and plot according to t
d_list = np.linspace(0, 1, 1000)[1:-1]
save_t = np.zeros((len(d_list)))
for i, d in enumerate(d_list):
    save_t[i] = sp.optimize.brentq(integral, 0, 1, args=(d))

#ax.scatter(save_t, d_list, label=f"{smiley:.4f}", color="blue", s=2)

# different smiley
smileys = [4.78]#np.linspace(1, 4.78, 10) # 4.78
colors = ["red", "magenta"]
d_list = np.linspace(0., 1, 1000)[1:-1]
# bad coding practice
for c, smiley in enumerate(smileys):
    save_t = np.zeros((len(d_list)))
    for i, d in enumerate(d_list):
        save_t[i] = sp.optimize.brentq(integral, 0, 1, args=(d))

    ax.scatter(save_t, d_list, label=f"{smiley}", color=colors[c], s=2)

data_BCS = np.loadtxt("BCS_gap_norm_1000.dat")

ax.scatter(data_BCS[0], data_BCS[1], color="hotpink", label="utilisé", s=2)

ax.tick_params(axis="both", labelsize=24)
ax.set_xlabel(r"$\tau$", fontsize=32)
ax.set_ylabel(r"$\delta$", fontsize=32)
ax.legend(markerscale=5, fontsize=20)
plt.show()
