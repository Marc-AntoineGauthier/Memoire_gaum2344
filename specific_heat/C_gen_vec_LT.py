import sys
import timeit as tt
from itertools import product as itp

import numpy as np
from scipy import interpolate

from k_gen import kw_gen_squ
from ham_lib import ham_funcs, ham_index
from gap_lib import gap_funcs, gap_index
from param import params


kb = 8.617333262145e-2  # value in meV
def dfdt(E, T, dEdT):
    arg = E/(2*kb*T)
    pre = 1/(kb*T)*1/(np.exp(arg) + np.exp(-arg))**2
    deu = (E/T - dEdT)
    return pre*deu
time_start = tt.default_timer()
# making the k grid generator from the passed args
taille = params["taille"]
n_group = params["n_group"]
k_file = f"sep_for_{taille}_{n_group}_BZ"

inter = np.loadtxt(k_file)[int(sys.argv[1])]
print(inter)

k_grid = kw_gen_squ(
    inter[0], inter[1],
    inter[3], inter[4],
    int(inter[2]), int(inter[5]))

# initializing the Ham
dim = 6
model = params["model"]
funcs_ham = ham_funcs[model]
index_ham = ham_index[model]
# Initializing the Gap
gap1 = params["gap1"]
funcs_gap1 = gap_funcs[gap1]
index_gap1 = gap_index[gap1]
Tc1 = params["Tc1"]

if params["n_gap"] == 1:
    gap2 = "nogap"
    funcs_gap2 = gap_funcs[gap2]
    index_gap2 = gap_index[gap2]
elif params["n_gap"] == 2:
    gap2 = params["gap2"]
    funcs_gap2 = gap_funcs[gap2]
    index_gap2 = gap_index[gap2]
    Tc2 = params["Tc2"]

# making the temperature and C list
T_l = np.linspace(0, 0.2, 31)[1:]
n_C = len(T_l)
C_l = np.zeros((n_C, 2*dim))

# Choosing the approximation
approx = params["approx"]
if approx == "BCS":
    data_BCS = np.loadtxt("BCS_gap_norm_1000.dat")
    T_BCS = data_BCS[0]
    gap_n_BCS = data_BCS[1]
    BCS_func = interpolate.interp1d(T_BCS, gap_n_BCS, fill_value="extrapolate")

H = np.zeros((dim, dim), dtype="complex")
# Hmk = np.zeros((dim, dim), dtype="complex")
D1 = np.zeros((dim, dim), dtype="complex")
D2 = np.zeros((dim, dim), dtype="complex")

phase = np.exp(1j*params["phase"])

# creating the temperature dependance
H_T_dep = np.ones((n_C, 1, 1))
gap_n1 = np.zeros((n_C, 1, 1))
gap_n2 = np.zeros((n_C, 1, 1))
if approx=="BCS":
    for i, T in enumerate(T_l):
        if T<Tc1:
            gap_n1[i] = params["gap_n1"]*BCS_func(T/1.5)
    if params["n_gap"] == 2:
        for i, T in enumerate(T_l):
            if T<Tc2:
                gap_n2[i] = params["gap_n2"]*BCS_func(T/1.5)

evals_T = np.zeros((n_C, 2*dim))
dEdT = np.zeros((n_C, 2*dim))
for k in k_grid:
    liste_val_ham = [f(k) for f in funcs_ham]
    liste_val_gap1 = [f(k) for f in funcs_gap1]
    liste_val_gap2 = [f(k) for f in funcs_gap2]

    H = np.array([liste_val_ham[i] for i in index_ham]).reshape((dim, dim))
    D1 = np.array(
        [info[1]*liste_val_gap1[info[0]] for info in index_gap1]
    ).reshape((dim, dim))
    D2 = np.array(
        [info[1]*liste_val_gap2[info[0]] for info in index_gap2]
    ).reshape((dim, dim))
    Delta_T = gap_n1*D1[None, :, :] + phase*gap_n2*D2[None, :, :]
    H_T = H_T_dep*H
    B = np.block(
        [
            [0.5*H_T, Delta_T],
            [
                np.conjugate(np.transpose(Delta_T, axes=(0, 2, 1))),
                -0.5*np.transpose(H_T, axes=(0, 2, 1))
            ]
        ]
    )
    evals_T, _ = np.linalg.eigh(B)
    # Calculating the derivatives and saving them
    dEdT = np.gradient(evals_T, T_l, edge_order=2, axis=0)
    C_l += evals_T*dfdt(evals_T, T_l[:, None], dEdT)
C_out = np.sum(C_l, axis=1)
# saving the contribution of the k points
if params["n_gap"] == 1:
    np.savetxt(
        "C/{}_{}_{}_{}_{}".format(
            model, gap1, approx, n_C, sys.argv[1]
        ), [T_l, C_out]
    )
elif params["n_gap"] == 2:
    np.savetxt(
        "C/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            model, approx, gap1, params["gap_n1"], Tc1, gap2,
            params["gap_n2"], Tc2, params["phase"],
            taille, sys.argv[1]
        ), [T_l, C_out]
    )
time_end = tt.default_timer()
print("time : ", time_end-time_start)
