import numpy as np

dpi = 2*np.pi
# Definition of the pauli matrices
s0 = np.array([
    [1, 0],
    [0, 1]
])

s1 = np.array([
    [0, 1],
    [1, 0]
])

s2 = np.array([
    [0, -1j],
    [1j, 0]
])

s3 = np.array([
    [1, 0],
    [0, -1]
])
# Definition of the ABC matrices
Ax = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

Ay = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

Az = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
])

Bx = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

By = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0]
])

Bz = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
])

Cx = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

Cy = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]
])

Cz = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
])

def pnoham(k):
    return np.zeros((3, 3))

t1 , t2, t3, t4, t5, mu = 88, 9, 80, 40, 5, 109
soc = 40

s3Ax, s3Ay, s3Az = np.kron(s3, Ax), np.kron(s3, Ay), np.kron(s3, Az)
s3Cz, s0By, s0Cx = np.kron(s3, Cz), np.kron(s0, By), np.kron(s0, Cx)

def pRHAnS(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    kp = soc/2
    ham = (
        # bands
        eyz*s3Ax + exz*s3Ay + exy*s3Az
    )
    return ham

def pRHA(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    kp = soc/2
    ham = (
        # bands
        eyz*s3Ax + exz*s3Ay + exy*s3Az +
        # SOC
        # second term as a minus!
        1j*kp*s3Cz - kp*s0By + 1j*kp*s0Cx
    )
    return ham

def pdRHAdx(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], 0.#k[2]
    eyx = 2*t2*np.sin(kx)
    exz = 2*t1*np.sin(kx)
    exy = 2*t3*np.sin(kx) + 4*t4*np.sin(kx)*np.cos(ky) + 4*t5*np.sin(2*kx)
    ham = eyx*s3Ax + exz*s3Ay + exy*s3Az
    return ham

def pdRHAdy(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], 0.#k[2]
    eyx = 2*t1*np.sin(ky)
    exz = 2*t2*np.sin(ky)
    exy = 2*t3*np.sin(ky) + 4*t4*np.cos(kx)*np.sin(ky) + 4*t5*np.sin(2*ky)
    ham = eyx*s3Ax + exz*s3Ay + exy*s3Az
    return ham

def pdRHAdz(k):
    return 0*k[2]*np.zeros((6, 6))



def pnogap(k):
    return 0*k[2]*np.zeros((6, 6))

s1Ax, s1Ay, s1Az = np.kron(s1, Ax), np.kron(s1, Ay), np.kron(s1, Az)
s1Cz, s2Cy, s2Bx = np.kron(s1, Cz), np.kron(s2, Cy), np.kron(s2, Bx)
s2Cz = np.kron(s2, Cz)

def pB1g(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        f1*s1Az - f2*s1Ax + f2*s1Ay +
        # Inter-orbital terms
        f1*s2Cz + (1j*s2Cy - s2Bx)
    )
    return gap

def pdB1gx(k):
    kx = dpi*k[0]
    f1 = 0.5*np.sin(kx)
    f2 = -0.5*np.sin(kx)
    gap = (
        f1*s1Az - f2*s1Ax + f2*s1Ay + f1*s2Cz
    )
    return gap

def pdB1gy(k):
    ky = dpi*k[1]
    f1 = -0.5*np.sin(ky)
    f2 = -0.5*np.sin(ky)
    gap = (
        f1*s1Az - f2*s1Ax + f2*s1Ay + f1*s2Cz
    )
    return gap

def pdB1gz(k):
    kz = dpi*k[2]
    return 0*kz*np.zeros((6, 6))

def pB1giOU(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        f1*s1Az - f2*s1Ax + f2*s1Ay +
        # Inter-orbital terms
        f1*s2Cz + 1j*(1j*s2Cy - s2Bx)
    )
    return gap

def pdB1giOUx(k):
    kx = dpi*k[0]
    f1 = 0.5*np.sin(kx)
    f2 = -0.5*np.sin(kx)
    gap = (
        f1*s1Az - f2*s1Ax + f2*s1Ay + f1*s2Cz
    )
    return gap

def pdB1giOUy(k):
    ky = dpi*k[1]
    f1 = -0.5*np.sin(ky)
    f2 = -0.5*np.sin(ky)
    gap = (
        f1*s1Az - f2*s1Ax + f2*s1Ay + f1*s2Cz
    )
    return gap

def pdB1giOUz(k):
    kz = dpi*k[2]
    return 0*kz*np.zeros((6, 6))

# The one with Oli terms
def pB1g_O(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay +
        # Inter-orbital terms
        1/3*(1j*s2Cy - s2Bx)
    )
    return gap

def pdB1g_Ox(k):
    kx = dpi*k[0]
    f1 = 0.5*np.sin(kx)
    f2 = -0.5*np.sin(kx)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1g_Oy(k):
    ky = dpi*k[1]
    f1 = -0.5*np.sin(ky)
    f2 = -0.5*np.sin(ky)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1g_Oz(k):
    kz = dpi*k[2]
    return 0*kz*np.zeros((6, 6))

def pB1g_intra_O(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1g_intra_Ox(k):
    kx = dpi*k[0]
    f1 = 0.5*np.sin(kx)
    f2 = -0.5*np.sin(kx)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1g_intra_Oy(k):
    ky = dpi*k[1]
    f1 = -0.5*np.sin(ky)
    f2 = -0.5*np.sin(ky)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1g_intra_Oz(k):
    kz = dpi*k[2]
    return 0*kz*np.zeros((6, 6))


def pB1giOU_O(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay +
        # Inter-orbital terms
        1/3*1j*(1j*s2Cy - s2Bx)
    )
    return gap

def pdB1giOU_Ox(k):
    kx = dpi*k[0]
    f1 = 0.5*np.sin(kx)
    f2 = -0.5*np.sin(kx)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1giOU_Oy(k):
    ky = dpi*k[1]
    f1 = -0.5*np.sin(ky)
    f2 = -0.5*np.sin(ky)
    gap = (
        f1*s1Az - 0.5*f2*s1Ax + 0.5*f2*s1Ay
    )
    return gap

def pdB1giOU_Oz(k):
    kz = dpi*k[2]
    return 0*kz*np.zeros((6, 6))

# In between B1g and B1g_O
def pB1g_all(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        -(1-1/2)*f2*s1Ax + (1-1/2)*f2*s1Ay +
        # Inter-orbital terms
        f1*s2Cz + (1-1/3)*(1j*s2Cy - s2Bx)
    )
    return gap

def pB1g_present(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        -(1-1/2)*f2*s1Ax + (1-1/2)*f2*s1Ay +
        # Inter-orbital terms
        (1-1/3)*(1j*s2Cy - s2Bx)
    )
    return gap

def pB1g_intra(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        # the two last terms have a - sign compare to not pseudogap
        -(1-1/2)*f2*s1Ax + (1-1/2)*f2*s1Ay
    )
    return gap

def pB1g_inter(k):
    kx, ky, kz = dpi*k[0], dpi*k[1], dpi*k[2]
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Inter-orbital terms
        f1*s2Cz + (1-1/3)*(1j*s2Cy - s2Bx)
    )
    return gap