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
# Definition of the Hamiltonians
def noham(k):
    return np.zeros((6, 6))
# Ham parameters
t1 , t2, t3, t4, t5, mu = 88, 9, 80, 40, 5, 109
soc = 40
s0Ax, s0Ay, s0Az = np.kron(s0, Ax), np.kron(s0, Ay), np.kron(s0, Az)
def RHAnS(k):
    """Hamiltonian without SOC

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the hamiltonian at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    ham = eyz*s0Ax + exz*s0Ay + exy*s0Az
    return ham

s3Cz, s2Cy, s1Cx = np.kron(s3, Cz), np.kron(s2, Cy), np.kron(s1, Cx)
def RHA(k):
    """Hamiltonian with SOC

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the hamiltonian at the k-point of shape (6, 6)
    """
    kx, ky, kz = k[0], k[1], 0.#k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    kp = soc/2
    ham = (
        # the diagonal terms
        eyz*s0Ax + exz*s0Ay + exy*s0Az +
        # the SOC terms
        1j*kp*s3Cz - 1j*kp*s2Cy + 1j*kp*s1Cx
    )
    return ham

def dRHAdx(k):
    kx, ky, kz = k[0], k[1], 0.#k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    eyx = 2*t2*np.sin(kx)
    exz = 2*t1*np.sin(kx)
    exy = 2*t3*np.sin(kx) + 4*t4*np.sin(kx)*np.cos(ky) + 4*t5*np.sin(2*kx)
    ham = eyx*s0Ax + exz*s0Ay + exy*s0Az
    return ham

def dRHAdy(k):
    kx, ky, kz = k[0], k[1], 0.#k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    eyx = 2*t1*np.sin(ky)
    exz = 2*t2*np.sin(ky)
    exy = 2*t3*np.sin(ky) + 4*t4*np.cos(kx)*np.sin(ky) + 4*t5*np.sin(2*ky)
    ham = eyx*s0Ax + exz*s0Ay + exy*s0Az
    return ham

def dRHAdz(k):
    return np.zeros((6, 6))

def RHA20(k):
    """Hamiltonian with SOC/2

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the hamiltonian at the k-point of shape (6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    kp = soc/4
    ham = (
        # the diagonal terms
        eyz*s0Ax + exz*s0Ay + exy*s0Az +
        # the SOC terms
        1j*kp*s3Cz - 1j*kp*s2Cy + 1j*kp*s1Cx
    )
    return ham

def RHA80(k):
    """Hamiltonian with SOC*2

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the hamiltonian at the k-point of shape (6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu
    eyz = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - mu
    exy = (
        -2*t3*(np.cos(kx) + np.cos(ky)) - 4*t4*np.cos(kx)*np.cos(ky) -
        2*t5*(np.cos(2*kx) + np.cos(2*ky))
    )
    exy -= mu
    kp = soc
    ham = (
        # the diagonal terms
        eyz*s0Ax + exz*s0Ay + exy*s0Az +
        # the SOC terms
        1j*kp*s3Cz - 1j*kp*s2Cy + 1j*kp*s1Cx
    )
    return ham

def nogap(k):
    return np.zeros((6, 6))

s2Az, s2AypAx = np.kron(s2, Az), np.kron(s2, Ay + Ax)
s2Bz = np.kron(s2, Bz)
s1Cz = np.kron(s1, Cz)
s3Cx, s0Cy = np.kron(s3, Cx), np.kron(s0, Cy)

def A1g(k):
    """The gap with the A1g symmetry
    Note that there is no actual k dependence

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    gap = (
        # Intra-orbital terms
        -1j*s2Az - 1j*s2AypAx
        # Inter-orbital terms
        - s1Cz + s3Cx - 1j*s0Cy
    )
    return gap

s2AymAx = np.kron(s2, Ay - Ax)
s3Cz, s0Cz = np.kron(s3, Cz), np.kron(s0, Cz)

def A2g(k):
    """The gap with the A2g symmetry
    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    xy = np.sin(kx)*np.sin(ky)
    #zx = np.sin(kz)*np.sin(kx)
    #zy = np.sin(kz)*np.sin(ky)
    f2 = 0.5*(np.cos(kx) - np.cos(ky))
    f1 = xy*(f2)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 - 1j*s2AymAx*xy
        # Inter-orbital terms
        - 1j*s2Bz*f2
        # Problem terms
        - s0Cx + 1j*s3Cy
        # 0 at kz=0
        #+ s3Cz*zy + 1j*s0Cz*zx  # Oumar : - +
    )
    return gap

def A2g_intra(k):
    """The gap with the A2g symmetry
    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    xy = np.sin(kx)*np.sin(ky)
    f2 = 0.5*(np.cos(kx) - np.cos(ky))
    f1 = xy*(f2)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 - 1j*s2AymAx*xy
    )
    return gap

def A2g2(k):
    """The gap with the A2gi symmetry
    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    xy = np.sin(kx)*np.sin(ky)
    #zx = np.sin(kz)*np.sin(kx)
    #zy = np.sin(kz)*np.sin(ky)
    f2 = 0.5*(np.cos(kx) - np.cos(ky))
    f1 = xy*(f2)
    gap = (
        # Intra-orbital terms
        2*1j*s2Az*f1 - 1j*s2AymAx*xy
        # Inter-orbital terms
        - 1j*s2Bz*f2
        # Problem terms
        + (1j*s3Cy - s0Cx)  # there is a *1j from Oumar
        # 0 at kz=0
        #+ s3Cz*zy + 1j*s0Cz*zx  # Oumar : - +
    )
    return gap

s2AxmAy = np.kron(s2, Ax - Ay)
s1Cz = np.kron(s1, Cz)
s1Bz = np.kron(s1, Bz)
s0Cx, s3Cy = np.kron(s0, Cx), np.kron(s3, Cy)
def B1g(k):
    """The gap with the B1g symmetry
        This one respects TRS

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
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
        1j*s2Az*f1 + 1j*s2AxmAy*f2
        # Inter-orbital terms
        + 1j*s1Cz*f1
        # Problem terms
        + (1j*s3Cx - s0Cy)  # there is a *-1j from Oumar
        # 0 at kz=0
        #+ (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

def B1gi(k):
    """The gap with the B1g symmetry
        This one does not respect TRS

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    zx = np.sin(kz)*np.sin(kx)
    zy = np.sin(kz)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2 +
        # Inter-orbital terms
        s1Cz*f1 +
        # Problem terms
        1j*(s3Cx + 1j*s0Cy) +
        # 0 at kz=0
        (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

def B1g_intra(k):
    """The gap with the B1g symmetry
        This one respects TRS

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2
        #1j*np.kron(s2, Ay)*f2
    )
    return gap

def B1gi_li5(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    zx = np.sin(kz)*np.sin(kx)
    zy = np.sin(kz)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2 +
        # Inter-orbital terms
        .5*s1Cz*f1 +
        # Problem terms
        .5*1j*(s3Cx + 1j*s0Cy) +
        # 0 at kz=0
        (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

def B1gOU(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    zx = np.sin(kz)*np.sin(kx)
    zy = np.sin(kz)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2 +
        # Inter-orbital terms
        s1Cz*f1 +
        # Problem terms
        (s3Cx + 1j*s0Cy)
        # 0 at kz=0
        #+ (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

def B1giOU(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    zx = np.sin(kz)*np.sin(kx)
    zy = np.sin(kz)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2 +
        # Inter-orbital terms
        1j*s1Cz*f1 +
        # Problem terms
        (s3Cx + 1j*s0Cy)
        # 0 at kz=0
        #+ (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap

def B1g_intra(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    f2 = 0.5*(np.cos(kx) + np.cos(ky))
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxmAy*f2
    )
    return gap

s2AxpAy = np.kron(s2, Ax + Ay)
def B1gF(k):
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    f1 = 0.5*(np.cos(ky) - np.cos(kx))
    gap = (
        # Intra-orbital terms
        1j*s2Az*f1 + 1j*s2AxpAy*f1
        # Inter-orbital terms
        + 1j*s1Cz*f1
        # Problem terms
        + (1j*s3Cx - s0Cy)  # there is a *1j from Oumar
    )
    return gap

def B2g(k):
    """The gap with the B2g symmetry

    Args:
        k (list): the k-vector [kx, ky, kz]

    Returns:
        numpy array: the gap at the k-point of shape(6, 6)
    """
    kx, ky, kz = k[0], k[1], k[2]
    kx *= dpi
    ky *= dpi
    kz *= dpi
    xy = np.sin(kx)*np.sin(ky)
    gap = (
        # Intra-orbital terms
        1j*s2Az*xy + 1j*s2AypAx*xy
        # Inter-orbital terms
        + 1j*s2Bz + 1j*s0Cx - s3Cy
    )
    return gap

# other ham
# SOC parameter
kp2 = 50.7
# parameters for xz and yz so we put z as the first letter
z1, z2, z3, z4, z5, z6, zu = 296.2, -57.2, 52.6, -15.6, -15.1, -11.6, 315.6

# parameters for xy 
d1, d2, d3, d4 = 369.5, 123.2, 20.4, 13.9
d5, d6, d7, du = -6., 3.2, 2.8, 432.5

def WRFS(k):
    kx = dpi*k[0]
    ky = dpi*k[1]
    #kz = dpi*k[2]
    x = np.cos(kx)
    y = np.cos(ky)
    x2 = np.cos(2*kx)
    y2 = np.cos(2*ky)
    x3 = np.cos(3*kx)
    y3 = np.cos(3*ky)
    eyz = -2*z1*y - 2*z2*y2 - 2*z3*x - 4*z4*y*x - 4*z5*y2*x - 2*z6*y3 - zu
    exz = -2*z1*x - 2*z2*x2 - 2*z3*y - 4*z4*x*y - 4*z5*x2*y - 2*z6*x3 - zu
    exy = (
        -2*d1*(x + y) - 4*d2*x*y - 4*d3*(x2*y + x*y2) - 4*d4*x2*y2
        - 2*d5*(x2 + y2) - 4*d6*(x3*y + x*y3) - 4*d7*(x3 + y3) - du
    )
    ham = (
        eyz*s0Ax + exz*s0Ay + exy*s0Az 
        + 1j*kp2*s3Cz - 1j*kp2*s2Cy + 1j*kp2*s1Cx
    )
    return ham

# not actually B1g
def B1g_O(k):
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
        1j*s2Az*f1 + 0.5*1j*s2AxmAy*f2
        # Inter-orbital terms
        + 1/3*1j*s1Cz*f1
        # Problem terms
        + (1/3*1j*s3Cx - 1/3*s0Cy)  # there is a *-1j from Oumar
        # 0 at kz=0
        #+ (1j*s0Cz*zy + s3Cz*zx)
    )
    return gap