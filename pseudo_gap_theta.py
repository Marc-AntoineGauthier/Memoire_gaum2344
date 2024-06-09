import numpy as np
from scipy.optimize import fminbound
from scipy import interpolate

import gap_pseudo as ps
save = False
save_path = False

kb = 8.617333262145e-2  # value in meV

# loading BCS T dependence
data_BCS = np.loadtxt("data/BCS_gap_norm_1000.dat")
BCS_T = interpolate.interp1d(
    data_BCS[0], data_BCS[1], fill_value="extrapolate"
)

class Bogo:
    def __init__(self) -> None:
        self.theta = 0
        self.cos = np.cos(self.theta)
        self.sin = np.sin(self.theta)
        self.ham = ps.pnoham
        self.gap = ps.pnogap
        self.base_gn = 0.32
        self.gn = self.base_gn
        self.T = 0.
        self.B = np.zeros((6, 6))
        self.eval = np.zeros(6)
        self.evec = np.zeros((6, 6))
        # not for usual use
        self.gap2 = ps.pnogap
        self.gn2 = 0.
    
    def set_theta(self, theta):
        self.theta = theta
        self.cos = np.cos(theta)
        self.sin = np.sin(theta)

    def set_ham(self, ham):
        self.ham = ham

    def set_ham_str(self, ham:str):
        self.ham = getattr(ps, ham)
    
    def set_gap(self, gap):
        self.gap = gap

    def set_gap_str(self, gap:str):
        self.gap = getattr(ps, gap)

    def set_gap2_str(self, gap:str):
        # not for usual use
        self.gap2 = getattr(ps, gap)
    
    def set_T(self, T):
        self.T = T
        self.gn = self.base_gn*BCS_T(T/1.5)

    def set_gap_norm(self, gn:float):
        self.base_gn = gn
        self.set_T(self.T)  # this is to ensure that the gn follows the current
                            # temperature

    def set_gap_norm2(self, gn:float):
        # not for usual use
        self.gn2 = gn

    def calc_bogo(self, r):
        k = np.array([r*self.cos, r*self.sin, 0.])
        H = self.ham(k)
        D = self.gn*self.gap(k)
        D2 = self.gn2*self.gap2(k)
        self.B = 0.5*H + D + D2 # D is already weighted
        self.eval, self.evec = np.linalg.eigh(self.B)
    
    def calc_gap(self, r):
        # function to be optimized
        k = np.array([r*self.cos, r*self.sin, 0.])
        H = self.ham(k)
        D = self.gn*self.gap(k)
        D2 = self.gn2*self.gap2(k)
        self.B = 0.5*H + D + D2 # D is already weighted
        eval, _ = np.linalg.eigh(self.B)
        return eval[3]
    
    def get_kx(self, r):
        return r*self.cos
    
    def get_ky(self, r):
        return r*self.sin
    


def bounds(dv):
    # need to explain this (unclear at first read)
    signdv = np.greater_equal(dv, 0)
    fmtp = np.logical_and(np.logical_not(signdv[:-1]), signdv[1:])
    index = np.where(fmtp)[0]
    # do not forget to have [i, i+1] so it gives the right things for fminbound
    bounds = np.array([[i, i+1] for i in index])
    return bounds


Ham = Bogo()
model = "pRHA"
model_save = model[1:]
Ham.set_ham_str(model)

gap_name = "pB1g_O"
gap_save_name = gap_name[1:]
Ham.set_gap_str(gap_name)

# not for ususal use
gap2_name = "pB1g_inter"
Ham.set_gap2_str(gap2_name)
Ham.set_gap_norm2(0.*Ham.gn)


Ham.set_T(0.)

# not for usual use
#Ham.set_gap_norm2(0.3)

xy_vec = np.array([0, 0, 1, 0, 0, 1])
xz_vec = np.array([0, 1, 0, 0, 1, 0])
yz_vec = np.array([1, 0, 0, 1, 0, 0])

n_angles = 750
fa = np.linspace(0, 0.57, n_angles)
ma = np.linspace(0.57, 0.61, 300)
sa = np.linspace(0.61, np.pi/4, n_angles)
angles = np.concatenate((fa, ma, sa))
#angles = np.linspace(0.58, 0.61, 2*n_angles+300)
gap_save = np.zeros((3, 2*n_angles+300, 6))
path_save = np.zeros((2*n_angles+300, 6))
r_len = 1200
for a, angle in enumerate(angles):
    Ham.set_theta(angle)
    # r goes from a to the max where a is a circle that can be drawn
    # without touching the FS and as the Gamma point as the origin
    ptk = np.linspace(0.25, 0.5/np.cos(angle), r_len)
    lg0 = np.zeros(r_len)  # kept in loop for safety reasons
    for i, r in enumerate(ptk):
        lg0[i] = Ham.calc_gap(r)
    # use derivative that goes from - to + to know how many min to find
    lg0dv = np.gradient(lg0, ptk)
    # need to explain this algorithm
    index = bounds(lg0dv)
    # fminbound in a loop over index
    for i, b in enumerate(index):
        mini_r = fminbound(Ham.calc_gap, ptk[b[0]], ptk[b[1]])
        Ham.calc_bogo(mini_r)
        avec = Ham.evec[:, 3]
        xy_w = np.real(np.dot(np.multiply(xy_vec, avec), np.conjugate(avec)))
        xz_w = np.real(np.dot(np.multiply(xz_vec, avec), np.conjugate(avec)))
        yz_w = np.real(np.dot(np.multiply(yz_vec, avec), np.conjugate(avec)))
        #print([xy_w, xz_w, yz_w])
        # The order is not the same as the HAM so the colors match the FS
        gap_save[i, a] = np.array(
            [mini_r, angle, Ham.eval[3], xz_w, yz_w, xy_w ]
        )
        path_save[a, 2*i] = Ham.get_kx(mini_r)
        path_save[a, 2*i+1] = Ham.get_ky(mini_r)

# second part but with a sin for the maximum r
fa = np.linspace(np.pi/4, np.pi/2 - 0.61, n_angles)
ma = np.linspace(np.pi/2 - 0.61, np.pi/2 - 0.57, 300)
sa = np.linspace(np.pi/2 - 0.57, np.pi/2, n_angles)
angles2 = np.concatenate((fa, ma, sa))
gap_save2 = np.zeros((3, 2*n_angles + 300, 6))
path_save2 = np.zeros((2*n_angles+300, 6))

for a, angle in enumerate(angles2):
    Ham.set_theta(angle)
    # r goes from a to the max where a is a circle that can be drawn
    # without touching the FS and as the Gamma point as the origin
    ptk = np.linspace(0.25, 0.5/np.sin(angle), r_len)
    lg0 = np.zeros(r_len)  # kept in loop for safety reasons
    for i, r in enumerate(ptk):
        lg0[i] = Ham.calc_gap(r)
    # use derivative to goes from - to + to know how many min to find
    lg0dv = np.gradient(lg0, ptk)
    # need to explain this algorithm
    index = bounds(lg0dv)
    # fminbound in a loop over index
    for i, b in enumerate(index):
        mini_r = fminbound(Ham.calc_gap, ptk[b[0]], ptk[b[1]])
        Ham.calc_bogo(mini_r)
        avec = Ham.evec[:, 3]
        xy_w = np.real(np.dot(np.multiply(xy_vec, avec), np.conjugate(avec)))
        xz_w = np.real(np.dot(np.multiply(xz_vec, avec), np.conjugate(avec)))
        yz_w = np.real(np.dot(np.multiply(yz_vec, avec), np.conjugate(avec)))
        #print([xy_w, xz_w, yz_w])
        # The order is not the same as the HAM so the colors match the FS
        gap_save2[i, a] = np.array(
            [mini_r, angle, Ham.eval[3], xz_w, yz_w, xy_w ]
        )
        path_save2[a, 2*i] = Ham.get_kx(mini_r)
        path_save2[a, 2*i+1] = Ham.get_ky(mini_r)

# making sure all the values are positive
# r, angle and eval[6] should be positive anyways
gap_save = np.abs(gap_save)
gap_save2 = np.abs(gap_save2)
gap_file = np.concatenate((gap_save, gap_save2), axis=2)
gap_file = np.reshape(gap_file, (3*(4*n_angles + 600), 6))
# saving the data
if save:
    print("we saved")
    np.savetxt(
        f"gap_theta_save/{model_save}/{gap_save_name}/GT_{4*n_angles}", gap_file
    )
if save_path:
    path_file = np.concatenate((path_save, path_save2), axis=0)
    np.savetxt(
        f"gap_theta_save/{model_save}/{gap_save_name}/path", path_file
    )
    print("path saved")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

for i in gap_save:
    ax.scatter(angles, i[:, 2], c=i[:,3:], s=1)
for i in gap_save2:
    ax.scatter(angles2, i[:, 2], c=i[:,3:], s=1)
# Temperature line
#ax.axhline(y=kb*Ham.T, color="red", alpha=0.75)

ax.set_xlabel(r"$\theta$", fontsize=24)
ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
ax.set_xticklabels(
    ["0", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$",
    r"$\frac{\pi}{2}$"]
)

ax.set_ylabel(r"$\Delta$ [meV]", fontsize=24)

ax.tick_params(axis="both", labelsize=20)

#ax.set_title("{}".format(gap_name), fontsize=24)
#ax.set_title("B1g_O zoom", fontsize=24)
#ax.set_title("{} T={:.2f}".format(gap_name, Ham.T), fontsize=24)

# totally custom legend :D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# The color is white so we do not see the line
legend_elements = [Line2D([0], [0], marker="o", color="white",
                    markerfacecolor="blue", markersize=10, label="XY"
                    ),
                    Line2D([0], [0], marker="o", color="white", 
                    markerfacecolor="red", markersize=10, label="YZ"
                    ),
                    Line2D([0], [0], marker="o", color="white",
                    markerfacecolor="green", markersize=10, label="XZ"
                    ),
                    Line2D([0], [0], marker="o", color="white",
                    markerfacecolor="black", markersize=10, label="no band"
                    )
]

ax.legend(handles=legend_elements)
#ax.set_ylim(0, 0.12)

plt.show()
