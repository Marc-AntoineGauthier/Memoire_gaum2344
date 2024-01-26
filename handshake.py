import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folderC = r"D:/Quest_for_SRO_gap/C/"
folderDRHA = r"archive/RHA/"
folderDRHAnS = r"archive/RHAnS/"

choices = ["B1g", "B1g_intra"]
number = "100"
m = 10000

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["green", "red"]
labels = ["$B_{1g}$ avec SOC", "$B_{1g}$ intra-orbitale avec SOC"]
for i, choice in enumerate(choices):
    data_C = np.loadtxt(folderC + f"RHA_{choice}_BCS_106_10000")
    #data_D100 = np.loadtxt(folderD + f"{choice}/C2_a{number}_5000_0.dat")
    data_D81 = np.loadtxt(folderDRHA + f"{choice}/C_a{number}_5000_0.dat")

    plt.scatter(
        data_C[0][:5], m*data_C[2][:5]/16, color=colors[i], s=100, marker="s",
        label=labels[i]
        )

    plt.scatter(
        data_D81[0], m*data_D81[1]/8, color=colors[i], s=20,
        label=labels[i] + " (DOS)"
        )

colors = ["blue", "orange"]
labels = ["$B_{1g}$ sans SOC", "$B_{1g}$ intra-orbitale sans SOC"]
for i, choice in enumerate(choices):
    data_C = np.loadtxt(folderC + f"RHAnS_{choice}_BCS_106_10000")
    #data_D100 = np.loadtxt(folderD + f"{choice}/C2_a{number}_5000_0.dat")
    data_D81 = np.loadtxt(folderDRHAnS + f"{choice}/C_a{number}_5000_0.dat")

    plt.scatter(
        data_C[0][:5], m*data_C[2][:5]/16, color=colors[i], s=100, marker="s",
        label=labels[i]
        )

    plt.scatter(
        data_D81[0], m*data_D81[1]/8, color=colors[i], s=20,
        label=labels[i] + " (DOS)"
        )

ax.set_xlabel("T [K]", fontsize=24)
ax.set_ylabel("C/T [arbs]", fontsize=24)

ax.tick_params(axis="both", labelsize=20)

ax.axvline(x=0, color="black")
ax.axhline(y=0, color="black")
ax.set_ylim(-m*0.00001, m*0.00025)

plt.legend(fontsize=20, ncol=2)

plt.show()

def line(x,a,b):
    return a*x + b

xliss = np.linspace(-2.35, -0.8, 100)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["green", "red"]
labels = ["$B_{1g}$ avec SOC", "$B_{1g}$ intra-orbitale avec SOC"]
fit_RHA = [(6, 11), (2, 8)]
for i, choice in enumerate(choices):
    data_D81 = np.loadtxt(folderDRHA + f"{choice}/C_a{number}_5000_0.dat")
    x_log, y_log = np.log10(data_D81[0]), np.log10(data_D81[1]/8*data_D81[0])
    plt.scatter(
        x_log, y_log,
        color=colors[i], s=20,
        #label=labels[i] + " with DOS"
        )
    #curvefit
    a, b = fit_RHA[i]
    popt, pcov = curve_fit(line, x_log[a:b], y_log[a:b], p0=[1., -20])
    plt.plot(
        xliss, line(xliss, *popt),
        label=f"{popt[0]:.4f}" + r" $\pm$ " + f"{pcov[0][0]:.4f}",
        color=colors[i]
    )
    

colors = ["blue", "orange"]
labels = ["$B_{1g}$ sans SOC", "$B_{1g}$ intra-orbitale sans SOC"]
fit_RHAnS = [[1, 10], [2, 7]]
for i, choice in enumerate(choices):
    data_D81 = np.loadtxt(folderDRHAnS + f"{choice}/C_a{number}_5000_0.dat")
    x_log, y_log = np.log10(data_D81[0]), np.log10(data_D81[1]/8*data_D81[0])
    plt.scatter(
        x_log, y_log, color=colors[i], s=20,
        #label=labels[i] + " with DOS"
        )
    
    a, b = fit_RHAnS[i]
    popt, pcov = curve_fit(line, x_log[a:b], y_log[a:b], p0=[1., -20])
    plt.plot(
        xliss, line(xliss, *popt),
        label=f"{popt[0]:.4f}" + r" $\pm$ " + f"{pcov[0][0]:.4f}",
        color=colors[i]
    )

ax.set_xlabel("log(T) [K]", fontsize=24)
ax.set_ylabel("log(C) [arbs]", fontsize=24)

ax.tick_params(axis="both", labelsize=20)

plt.legend(fontsize=20, ncol=2)

plt.show()