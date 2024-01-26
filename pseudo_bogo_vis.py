# %%
import numpy as np

import gap_pseudo as ps
# %%
# must be redone to work
RHA = (
    np.kron(ps.s0, ps.Ax) + np.kron(ps.s0, ps.Ay) + np.kron(ps.s0, ps.Az) +
    1j*np.kron(ps.s3, ps.Cz) + 1j*np.kron(ps.s2, ps.Cy) +
    1j*np.kron(ps.s1, ps.Cx)
)

#print(RHA)

B1g = (
    1j*np.kron(ps.s2, ps.Az) + 1j*np.kron(ps.s2, ps.Ax) -
    1j*np.kron(ps.s2, ps.Ay) +
    1j*np.kron(ps.s1, ps.Cz) + 1j*np.kron(ps.s3, ps.Cx) -
    np.kron(ps.s0, ps.Cy)
)

print(B1g)
# %%
# do the full bogo then do pseudo trans
Bogo = np.block([
    [RHA, np.conjugate(np.transpose(B1g))],
    [B1g, -np.transpose(RHA)]
])

A, B = [0, 1, 5, 3, 4, 2, 6, 7, 11, 9, 10, 8], np.arange(0, 12, 1)
Bogo[A,:] = Bogo[B,:]
Bogo[:,A] = Bogo[:,B]
print(np.matrix(Bogo))
C, D = [0, 1, 2, 9, 10, 11, 6, 7, 8, 3, 4, 5], np.arange(0, 12, 1)
Bogo[C,:] = Bogo[D,:]
Bogo[:,C] = Bogo[:,D]
print(np.shape(Bogo))
Bogo_m = np.matrix(Bogo)
print(Bogo_m)
# %%

# here we use the new matrices to do the trans in one go and compare to above
Bogo_p = (
    np.kron(ps.s3, ps.Ax) + np.kron(ps.s3, ps.Ay) + np.kron(ps.s3, ps.Az) +
    1j*np.kron(ps.s3, ps.Cz) + np.kron(ps.s0, ps.By) +
    1j*np.kron(ps.s0, ps.Cx) -
    np.kron(ps.s1, ps.Ax) + np.kron(ps.s1, ps.Ay) + np.kron(ps.s1, ps.Az) +
    1j*np.kron(ps.s1, ps.Cz) + 1j*np.kron(ps.s2, ps.Cy) -
    np.kron(ps.s2, ps.Bx)
)

print(Bogo_p)

print(ps.RHA + ps.B1g)