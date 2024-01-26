import numpy as np

dpi = 2*np.pi

# parameter definition (not general and should be done better)
ea, eb, t, tp, tbx = -131.8, -132.22, -81.62, -36.73, -109.37
tby, tbp, tabp, tacp, tbc = -6.56, 0.262, -1.05, -1.05, -8.75
tbcp, c = -1.05, 3.279
kp = 16  # 16 is 0.2*80 where 0.2 is the value in OS

def t00(k):
    pre = eb + 2*(tby*np.cos(dpi*k[0]) + tbx*np.cos(dpi*k[1]))
    deu = 8*tbp*np.cos(dpi*k[0]/2)*np.cos(dpi*k[1]/2)*np.cos(c*dpi*k[2]/2)
    return pre + deu
#ham.set_term([[0, 0], [3, 3]], t00, t00_mk, 1)

def t01(k):
    pre = 4*tbc*np.sin(dpi*k[0])*np.sin(dpi*k[1])
    deu = 8*tbcp*np.sin(dpi*k[0]/2)*np.sin(dpi*k[1]/2)*np.cos(c*dpi*k[2]/2)
    return pre + deu
#ham.set_term([[0, 1], [1, 0], [3, 4], [4, 3]], t01, t01_mk, 1)

def t02(k):
    return 8*tacp*np.sin(dpi*k[0]/2)*np.cos(dpi*k[1]/2)*np.sin(c*dpi*k[2]/2)
#ham.set_term([[0, 2], [2, 0], [3, 5], [5, 3]], t02, t02_mk, 1)

def t11(k):
    pre = eb + 2*(tbx*np.cos(dpi*k[0]) + tby*np.cos(dpi*k[1]))
    deu = 8*tbp*np.cos(dpi*k[0]/2)*np.cos(dpi*k[1]/2)*np.cos(c*dpi*k[2]/2)
    return pre + deu
#ham.set_term([[1, 1], [4, 4]], t11, t11_mk, 1)

def t12(k):
    return 8*tabp*np.cos(dpi*k[0]/2)*np.sin(dpi*k[1]/2)*np.sin(c*dpi*k[2]/2)
#ham.set_term([[1, 2], [2, 1], [4, 5], [5, 4]], t12, t12_mk, 1)

def t22(k):
    pre = ea + 2*t*(np.cos(dpi*k[0]) + np.cos(dpi*k[1]))
    deu = 4*tp*np.cos(dpi*k[0])*np.cos(dpi*k[1])
    return pre + deu
#ham.set_term([[2, 2], [5, 5]], t22, t22_mk, 1)
def zero(k): return 0
# redo this so zero is 0
funcs_GWAG = [t00, t01, t02, t11, t12, t22, zero]
index_GWAG = (
    0, 1, 2, 6, 6, 6,
    1, 3, 4, 6, 6, 6,
    2, 4, 5, 6, 6, 6,
    6, 6, 6, 0, 1, 2,
    6, 6, 6, 1, 3, 4,
    6, 6, 6, 2, 4, 5
)

# GWAGF
def t01pk(k):
    pre = 4*tbc*np.sin(dpi*k[0])*np.sin(dpi*k[1])
    deu = 8*tbcp*np.sin(dpi*k[0]/2)*np.sin(dpi*k[1]/2)*np.cos(c*dpi*k[2]/2)
    return pre + deu + 0.5j*kp

def t01mk(k):
    pre = 4*tbc*np.sin(dpi*k[0])*np.sin(dpi*k[1])
    deu = 8*tbcp*np.sin(dpi*k[0]/2)*np.sin(dpi*k[1]/2)*np.cos(c*dpi*k[2]/2)
    return pre + deu - 0.5j*kp

def kappa(k):
    return 0.5*kp

def mkappa(k):
    return -0.5*kp

def ikappa(k):
    return 0.5j*kp

def mikappa(k):
    return -0.5j*kp

funcs_GWAGF = [
    zero, t00, t01pk, t01mk, t02, t11, t12, t22,
    kappa, mkappa, ikappa, mikappa
]

index_GWAGF = (
    1, 2, 4, 0, 0, 9,
    3, 5, 6, 0, 0, 10,
    4, 6, 7, 8, 11, 0,
    0, 0, 8, 1, 3, 4,
    0, 0, 10, 2, 5, 6,
    9, 11, 0, 4, 6, 7
)


t1 , t2, t3, t4, t5, mu = 88, 9, 80, 40, 5, 109
soc = 40

def exz(k):
    return -2*t1*np.cos(dpi*k[0]) - 2*t2*np.cos(dpi*k[1]) - mu

def eyz(k):
    return -2*t2*np.cos(dpi*k[0]) - 2*t1*np.cos(dpi*k[1]) - mu

def exy(k):
    pre = -2*t3*(np.cos(dpi*k[0]) + np.cos(dpi*k[1]))
    deu = -4*t4*np.cos(dpi*k[0])*np.cos(dpi*k[1])
    tro = -2*t5*(np.cos(2*dpi*k[0]) + np.cos(2*dpi*k[1]))
    return pre + deu + tro -mu

def SOC(k):
    return 0.5*soc

def mSOC(k):
    return -0.5*soc

def iSOC(k):
    return 0.5j*soc

def miSOC(k):
    return -0.5j*soc

funcs_RHA = [zero, exz, eyz, exy, SOC, mSOC, iSOC, miSOC]

index_RHA = (
    1, 6, 0, 0, 0, 5,
    7, 2, 0, 0, 0, 6,
    0, 0, 3, 4, 7, 0,
    0, 0, 4, 1, 7, 0,
    0, 0, 6, 6, 2, 0,
    5, 7, 0, 0, 0, 3
)


funcs_RHAnS = [zero, exz, eyz, exy]

index_RHAnS = (
    1, 0, 0, 0, 0, 0,
    0, 2, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 3
)


ham_funcs = {
    "GWAG": funcs_GWAG, "GWAGF": funcs_GWAGF, "RHA": funcs_RHA,
    "RHAnS": funcs_RHAnS
}
ham_index = {
    "GWAG": index_GWAG, "GWAGF": index_GWAGF, "RHA": index_RHA,
    "RHAnS": index_RHAnS
}