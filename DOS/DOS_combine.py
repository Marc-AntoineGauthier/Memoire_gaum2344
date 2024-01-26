import sys
import numpy as np

from march_params import params

Nx = params["Nx"]
Nx0 = params["Nx0"]
div = int(sys.argv[1])

C_tot = np.zeros((10))
C_l = []

data = np.loadtxt(
    f"{Nx0}_{Nx}_0.dat"
)
E = data[0]
DOS = data[1]

for i in range(1, div):
    data = np.loadtxt(
    f"{Nx0}_{Nx}_{i}.dat"
    )
    DOS += data[1]

np.savetxt(
    f"a{div}_{Nx0}_{Nx}.dat",
    [E, DOS]
)
