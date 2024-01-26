# %%
import numpy as np

import group_trans as gt
import gap_con as con
dpi = 2*np.pi
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=2)
# %%
# Example of how to print a character table for predefine functions
gt.print_header()
gt.print_table(con.A1g, name="A1g")
gt.print_table(con.A2g, name="A2g")
gt.print_table(con.A2g, name="A2g2")
gt.print_table(con.B1g, name="B1g")
gt.print_table(con.B1gi, name="B1gi")
gt.print_table(con.B1gOU, name="B1gOU")
gt.print_table(con.B1giOU, name="B1giOU")

def B1gA2g(k):
    return con.B1g(k) + 1j*con.A2g(k)

gt.print_table(B1gA2g, name="B1g + A2g")
# %%
gt.print_header()

def allo1(k):
    return 1j*np.kron(con.s2, con.Ax) + 1j*np.kron(con.s2, con.Ay)

def allo2(k):
    return -1j*np.kron(con.s2, con.Ax) + 1j*np.kron(con.s2, con.Ay)

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


gt.print_table(allo1, name="plus")
gt.print_table(allo2, name="minus")
gt.print_table(con.B1gF, name="another one")
gt.print_table(con.B1g, name="B1g")
# %%
# Example of how to print the character table for a user define function
# Here the function takes matrices from the "gap_con.py" script.
# The function must output a 6x6 matrix and take as input [kx, ky, kz].
# The k point taken have 0.5 as pi/a.
def test(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    x = np.sin(kx)
    y = np.sin(ky)
    z = np.sin(kz)
    #x2 = 1 - np.cos(kx)
    #y2 = 1 - np.cos(ky)
    pre = -1j*np.kron(con.s0, con.Bx)
    deu = np.kron(con.s3, con.By)
    return 1j*(pre + deu)

gt.print_header()
gt.print_table(test, name="test")
# %%
from k_gen import k_gen
from scipy.integrate import quad
def func1(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    return np.cos(kx) + np.cos(ky)
def func2(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    return np.cos(2*kx) + np.cos(2*ky)
def funint1(x, y):
    return (np.cos(x) - np.cos(y))*(np.cos(x) - np.cos(y))
def funint2 (y):
    return quad(funint1, -np.pi, np.pi, args=(y))[0]

# Verifying the symmetries
gt.print_header()
gt.print_tablek(func1, name="1")
gt.print_tablek(func2, name="2")
# Verifying the orthogonality
print(quad(funint2, -np.pi, np.pi))

# %%
# trying to see the irreps of the Pauli matrices 
def sig(k):
    return np.kron(con.s1, con.Az)
gt.print_header()
gt.print_table(sig, name="s0")
# %%
# redoing the nontrivial Eg cross Eg tables
gt.print_header()

def temp(k):
    return np.kron(con.s0, con.Cy) + 1j*np.kron(con.s3, con.Cx)
gt.print_table(temp, name="A1g")

def temp(k):
    return np.kron(con.s3, con.Cy) + 1j*np.kron(con.s0, con.Cx)
gt.print_table(temp, name="A2g")

def temp(k):
    return np.kron(con.s0, con.Cy) - 1j*np.kron(con.s3, con.Cx)
gt.print_table(temp, name="B1g")

def temp(k):
    return np.kron(con.s3, con.Cy) - 1j*np.kron(con.s0, con.Cx)
gt.print_table(temp, name="B2g")

gt.print_header()

def temp(k):
    return np.kron(con.s0, con.By) + 1j*np.kron(con.s3, con.Bx)
gt.print_table(temp, name="A1g")

def temp(k):
    return np.kron(con.s3, con.By) + 1j*np.kron(con.s0, con.Bx)
gt.print_table(temp, name="A2g")

def temp(k):
    return np.kron(con.s0, con.By) - 1j*np.kron(con.s3, con.Bx)
gt.print_table(temp, name="B1g")

def temp(k):
    return np.kron(con.s3, con.By) - 1j*np.kron(con.s0, con.Bx)
gt.print_table(temp, name="B2g")