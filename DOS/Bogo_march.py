import sys

import numpy as np
import timeit as tt
#import matplotlib.pyplot as plt

#from Class_march import Bogo_March
#from march_params import params
from Class_pseudo_march import Bogo_March
from pseudo_march_params import params

#import psutil

genesis = tt.default_timer()

#energies = np.linspace(params["lowest_e"], params["highest_e"], params["NE"])
#DOS_list = np.zeros((params["NE"]))

f_e, l_e = np.linspace(-1., -0.3, 5), np.linspace(0.3, 1., 5)
m_e = np.linspace(-0.3, 0.3, 1001)
energies = np.concatenate((f_e, m_e, l_e))
DOS_list = np.zeros(len(energies))

# creating the object
march = Bogo_March()
# we initialize the BZ
s, N = int(sys.argv[1]), int(np.sqrt(int(sys.argv[2])))
interx = (params["Mx"] - params["mx"])/N
intery = (params["My"] - params["my"])/N
interz = (params["Mz"] - params["mz"])/N
# x
wx = s//N
march.mx, march.Mx = wx*interx, (wx + 1)*interx
# y
wy = s%N
march.my, march.My = wy*intery, (wy + 1)*intery
# z
wz = 0.
march.mz, march.Mz = wz*interz, (wz + 1)*interz
# The rest of the parameters
march.params_from_params()

for e, ener in enumerate(energies):
    time_start = tt.default_timer()
    DOS = 0
    march.set_energy(ener)

    march.pathfinder()
    march.precision()
    for i,contour in enumerate(march.contours):
        for j, c in enumerate(contour):
            c_shifted = np.roll(np.copy(c), -1, axis=0)

            pts_diff = np.add(c, -c_shifted)

            distances = 2*np.pi*np.sqrt(np.sum(pts_diff*pts_diff, axis=1))
            vec = march.eigen_vec[i][j]
            vec = np.expand_dims(vec, axis=(-1))
            
            c = np.expand_dims(c, axis=(-2, -1))
            c = np.transpose(c, (1, 0, 2, 3))

            vx = (
                np.conjugate(np.transpose(vec, (0, 2, 1))) @
                march.dBogox(c) @ vec
            )
            vy = (
                np.conjugate(np.transpose(vec, (0, 2, 1))) @
                march.dBogoy(c) @ vec
            )
            vz = (
                np.conjugate(np.transpose(vec, (0, 2, 1))) @
                march.dBogoz(c) @ vec
            )
            v_vec = np.transpose(np.squeeze(np.array([vx, vy, vz])))

            v = np.sqrt(vx*vx + vy*vy + vz*vz)
            
            sign = np.sign(np.cross(v_vec, pts_diff))[:,-1]
            #print(np.shape(sign))
            #print("positive : ", np.sum(sign >= 0))
            #print("negative : ", np.sum(sign <= 0))
            #print("total : ", len(sign))
            #exit()
            v = np.squeeze(v)

            # the last point is dubious
            DOS += np.sum(sign[:-1]*distances[:-1]/v[:-1])
        #print("number of paths : ", j)
    if np.abs(np.imag(DOS))>1e-10:
        print("problem : ", DOS)
    DOS_list[e] = np.abs(1/(4*np.pi*np.pi)*np.real(DOS))
    time_end = tt.default_timer()
    print("time : ", time_end - time_start)

#plt.scatter(energies, DOS_list, s=2, color="purple")
#plt.show()
np.savetxt(
    f"{params['Nx0']}_{sys.argv[1]}.dat", [energies, DOS_list]
)

demise = tt.default_timer()
print("total time : ", demise - genesis)
#print(psutil.Process().memory_info().rss / (1024 * 1024))
