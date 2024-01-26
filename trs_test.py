import numpy as np

from k_gen import k_gen
import gap_con as con

dim = 6
ddim = int(dim/2)  # for demi-dim, which is half of the dim of the matrix
taille = 100
dpi = 2*np.pi
def B1g_again(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    #zx = np.sin(kz)*np.sin(kx)
    #zy = np.sin(kz)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*np.kron(con.s2, con.Az)*f1 
        + 1j*np.kron(con.s2, con.Ax)*f1
        + 1j*np.kron(con.s2, con.Ay)*f1
        # Inter-orbital terms
        + 1j*np.kron(con.s1, con.Cz)*f1
        # Problem terms
        + (1j*np.kron(con.s3, con.Cx) - np.kron(con.s0, con.Cy))  # there is a *1j from Oumar
        # 0 at kz=0
        #+ (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

gap = con.B1g

gap1 = np.zeros((taille, taille, dim, dim), dtype=complex)
gap2 = np.zeros((taille, taille, dim, dim), dtype=complex)
# create the gap on the k-grid
for i, ky in enumerate(k_gen(-0.5, 0.5, taille)):
    for j, kx in enumerate(k_gen(-0.5, 0.5, taille)):
        k = [kx, ky, 0.]
        gap1[j, i] = gap(k)/(np.sqrt(2))

# create the gap from the other gap using the transformation that is
# like the Time-Reversal transformation
# switch index, then (+3, +3)%dim and sign(line - ddim)sign(col - ddim)
for i in range(dim):
    for j in range(dim):
        gap2[:,:,i,j] = (  # the +0.1 is so 3-3 gives + and not 0
            np.sign(i-ddim+0.1)*np.sign(j-ddim+0.1)*
            gap1[:,:,(i+ddim)%dim,(j+ddim)%dim]
        )
        #print(i, j, np.sign(i-ddim+0.1)*np.sign(j-ddim+0.1))
# comment if testing hamiltonians
#gap2 = np.conjugate(np.transpose(gap2, axes=(0, 1, 3, 2)))
gap2 = np.conjugate(gap2)
#gap2  = np.transpose(gap2, axes=(0,1,3,2))
# compare the two gaps with np.isclose
respect = np.all(np.isclose(gap1, gap2))
print(respect)

import matplotlib.pyplot as plt
from itertools import product as itp
if not(respect):
    gap_plot = gap1 - gap2
    fig, ax = plt.subplots(dim, dim, figsize=(16, 8))
    low_r, high_r = np.min(np.real(gap_plot)), np.max(np.real(gap_plot))
    for d1, d2 in itp(range(dim), repeat=2):
        data_r = np.real(gap_plot[:,:,d1,d2])
        im = ax[d1][d2].imshow(data_r, origin="lower", cmap="bwr")
        #ax[d1][d2].set_axis_off()
        im.set_clim(low_r, high_r)
        #fig.colorbar(im, ax=ax[d1][d2])
        ax[d1][d2].set_xticks([])
        ax[d1][d2].set_xticklabels([])
        ax[d1][d2].set_yticks([])
        ax[d1][d2].set_yticklabels([])
    fig, ax = plt.subplots(dim, dim, figsize=(16, 8))
    for d1, d2 in itp(range(dim), repeat=2):
        data_r = np.imag(gap_plot[:,:,d1,d2])
        im = ax[d1][d2].imshow(data_r, origin="lower", cmap="bwr")
        #ax[d1][d2].set_axis_off()
        im.set_clim(low_r, high_r)
        #fig.colorbar(im, ax=ax[d1][d2])
        ax[d1][d2].set_xticks([])
        ax[d1][d2].set_xticklabels([])
        ax[d1][d2].set_yticks([])
        ax[d1][d2].set_yticklabels([])
    plt.show()
