import numpy as np
import os

from skimage import measure

import gap_pseudo as ps

from pseudo_march_params import params

class Bogo_March:
    def __init__(self):
        # This is the base constructor to keep it somewhat general
        # setting all parameters to some base balue
        # BZ parameters
        self.mx, self.Mx = 0., 0.5
        self.my, self.My = 0., 0.5
        self.mz, self.Mz = 0., 0.
        # March parameters
        self.Nx0, self.Ny0, self.Nz0 = 0, 0, 0
        # Bogo parameters
        self.dim = 6  # there is no *2 because pseudo already has it
        self.ngap = 0.32
        self.H_model = "pnoham"
        self.H = ps.pnogap # nogap is just a 6x6 matrice of zeros
        self.D_model = "pnogap"
        self.D = ps.pnogap
        self.dHx, self.dHy, self.dHz = ps.pnogap, ps.pnogap, ps.pnogap
        self.dDx, self.dDy, self.dDz = ps.pnogap, ps.pnogap, ps.pnogap
        # variable that will be computed
        self.Bg = np.zeros((1, 1, 1, self.dim, self.dim))  # for B_grid
        self.Bp = np.zeros((1, self.dim, self.dim))  # for B_point
        self.ham_ev_grid = np.zeros((self.dim, 1, 1))  # to have something
        self.contours = []  # to have something
        self.new_contours = [] # to find the better paths
        self.eigen_vec = []  # we want to save the eigenvectors
        self.new_eigen_vec = []  # for the better paths
        # the energy we will be working on
        self.energy_ind = -1
        self.lv = 0.
        # keep track of which iteration we are at
        self.iteration = 0
    
    def set_energy(self, lv):
        self.lv = lv
        self.energy_ind += 1
        print(f"Level : {self.lv}    (ind : {self.energy_ind})")
    
    def Bogo_grid(self, k):
        """This function is used to evaluate Bogo on a grid

        Args:
            k (numpy array): The shape should be (3, Nx, Ny, Nz, 1, 1), where
            the last two 1 are expanded dimensions.
        """
        Hk = self.H(k)
        Dk = self.D(k)
        # The returned Hk and Dk are shape (Nx, Ny, Nz, dim, dim)
        self.Bg = 0.5*Hk + self.ngap*Dk

    def Bogo_pts(self, k):
        """This function is used to evaluate Bogo on a list of k points

        Args:
            k (numpy array): The shape shuold be (3, Nk, 1, 1), where the
            last two 1 are expanded dimensions.
        """
        Hk = self.H(k)
        Dk = self.D(k)
        # The returned Hk and Dk are shape (Nk, dim, dim)
        self.Bp = 0.5*Hk + self.ngap*Dk
    
    def dBogox(self, k):
        Hk = self.dHx(k)
        Dk = self.dDx(k)
        bogo_pt = 0.5*Hk + self.ngap*Dk
        return bogo_pt

    def dBogoy(self, k):
        Hk = self.dHy(k)
        Dk = self.dDy(k)
        bogo_pt = 0.5*Hk + self.ngap*Dk
        return bogo_pt

    def dBogoz(self, k):
        Hk = self.dHz(k)
        Dk = self.dDz(k)
        bogo_pt = 0.5*Hk + self.ngap*Dk
        return bogo_pt

    
    def params_from_params(self):
        """
        This method will set the parameters of the object to the prameters
        of the parameter file.
        """
        # March parameters
        self.Nx0 = params["Nx0"]
        self.Ny0 = params["Ny0"]
        self.Nz0 = params["Nz0"]
        # Bogo parameters
        self.dim = params["dim"]  # no 2* because of pseudo
        self.ngap = params["ngap"]

        self.H_model = params["H"]
        self.H = getattr(ps, self.H_model)

        self.D_model = params["D"]
        self.D = getattr(ps, self.D_model)

        self.dHx = getattr(ps, params["dHx"])
        self.dHy = getattr(ps, params["dHy"])
        self.dHz = getattr(ps, params["dHz"])

        self.dDx = getattr(ps, params["dDx"])
        self.dDy = getattr(ps, params["dDy"])
        self.dDz = getattr(ps, params["dDz"])
        # We compute the Bogo grid now that we have the parameters
        kx, ky, kz = np.ogrid[
            self.mx:self.Mx:(self.Nx0+1)*1j,
            self.my:self.My:(self.Ny0+1)*1j,
            self.mz:self.Mz:(self.Nz0+1)*1j
        ]
        k = np.meshgrid(kx, ky, kz)
        # expand dims to be coherent with the matrix
        k = np.expand_dims(k, axis=(-2, -1))
        self.Bogo_grid(k)
        # we want the eigenvalues for the grid saved to loop over energies
        # easily
        self.ham_ev_grid, _ = np.linalg.eigh(self.Bg)
        # making the eigenvalues in the right shape for marching square
        self.ham_ev_grid = np.transpose(self.ham_ev_grid, (3, 0, 1, 2))
        self.ham_ev_grid = np.reshape(
            self.ham_ev_grid, (self.dim, self.Nx0+1, self.Ny0+1)
        )    

    def pathfinder(self):
        # we reset the iteration counter because it is like doing another case
        self.iteration = 0
        # likewise for saved_stats
        self.contours = []  # we reset the contours
        for ev in self.ham_ev_grid:
            temp_contour = []
            contour = measure.find_contours(ev, self.lv)
            for c in contour:
                temp_pts = []
                for p in c:
                    temp_pts.append([
                        # x is 1 and y is 0
                        (self.Mx - self.mx)*p[1]/self.Nx0 + self.mx,
                        (self.My - self.my)*p[0]/self.Ny0 + self.my,
                        0.
                    ])
                temp_contour.append(np.array(temp_pts))
            self.contours.append(temp_contour)
        

    def precision(self, save=False, verbose=False):
        """
        This function will print some statistic on the precision of the current
        contours. Since this requires to compute the eigenvalues, the 
        eigenvectors will also e computed and saved.
        Args:
            save (Bool): if True saves the stats and contours of the iteration.
            Default is False
            verbose (Bool): if True print a lot of stuff. Default is False
        """
        # we reset the eigen_vector save
        self.eigen_vec = []
        for i, contour in enumerate(self.contours):
            
            temp_vec_list = []
            for indc, c in enumerate(contour):
                pts_t = np.copy(c)
                pts_t = np.expand_dims(pts_t, axis=(-2, -1))
                # We want (3, Nk, dim, dim)
                pts_t = np.transpose(pts_t, (1, 0, 2, 3))
                self.Bogo_pts(pts_t)
                temp_e, temp_vec = np.linalg.eigh(self.Bp)
                #print("temp_vec : ", len(temp_vec))
                temp_vec_list.append(temp_vec[:,:,i])
                #print(
                #    "shapes : ", np.shape(pts_t), np.shape(self.Bp),
                #    np.shape(temp_e), np.shape(temp_vec)
                #)
                stat = np.mean(np.abs(temp_e[:,i]))

                if save:
                    folder = f"data/e{self.energy_ind}/i{self.iteration}"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    # saving the precision

                    # saving the paths
                    file = folder + f"/{i}_{indc}_path.dat"
                    np.savetxt(file, c)
                if verbose:
                    print(f"{i}_{indc} : {(stat - np.abs(self.lv))/self.lv}")
                
            #print("the appended : ", len(temp_vec_list))
            self.eigen_vec.append(temp_vec_list)
