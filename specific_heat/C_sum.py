import sys
import numpy as np

from param import params

taille = params["taille"]
n_group = params["n_group"]
model = params["model"]
n_gap = params["n_gap"]
gap1 = params["gap1"]
gap_n1 = params["gap_n1"]
Tc1 = params["Tc1"]
gap2 = params["gap2"]
gap_n2 = params["gap_n2"]
Tc2 = params["Tc2"]
phase = params["phase"]
approx = params["approx"]


C_tot = np.zeros((10))
C_l = []
if n_gap == 1:
    for i in range(n_group):
        data = np.loadtxt(
            f"C/{model}_{gap1}_{approx}_{int(len(C_tot))}_{i}"
        )
        T = data[0]
        C = data[1]
        C_tot += C
elif n_gap == 2:
    for i in range(n_group):
        data = np.loadtxt(
            "C/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                model, approx, gap1, gap_n1, Tc1, gap2, gap_n2, Tc2,
                phase, taille, i
            )
        )
        T = data[0]
        C = data[1]
        C_tot += C

C_tot *= 4/(taille*taille)
C_T = [C_tot[i]/T[i] for i in range(len(T))]

print("shape of T : ", np.shape(T))
print("shape of C : ", np.shape(C_tot))
print("shape of C_T : ", np.shape(C_T))
if n_gap == 1:
    np.savetxt(
        f"{model}_{gap1}_{approx}_{int(len(C_tot))}_{taille}",
        [T, C_tot, C_T]
    )
elif n_gap == 2:
    np.savetxt(
        "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            model, approx, gap1, gap_n1, Tc1, gap2, gap_n2, Tc2,
            phase, taille
        ),
        [T, C_tot, C_T]
    )