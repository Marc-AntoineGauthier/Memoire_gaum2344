params = {
    # The Brillouin zone
    "mx": 0.,  # minimum in x
    "Mx": 0.5, # maximum in x
    "my": 0.,  # minimum in y
    "My": 0.5, # maximum in y
    "mz": 0.,  # minimum in z
    "Mz": 0.,  # maximum in z
    # Pathfinder
    "Nx0": 100,  # number of points along x for the initial grid
    "Ny0": 100,  # number of points along y for the initial grid
    "Nz0": 0,    # number of points along z for the initial grid
    # Energies
    "lowest_e": -1.,  # lowest energy
    "highest_e": 1.,  # highest eergy
    "NE": 50,  # number of energy points
    # For the model
    "dim": 6,  # dimension of the matrix that will go into Bogo
    "ngap": 0.32,  # the norm of the gap for Bogo
    "H": "pRHA",    # name of the normal state hamiltonian in gap_con.py
    "D": "pB1g",    # name of the SCOP in gap_con.py
    "dHx": "pdRHAdx",  # name of dH/dx in gap_con.py
    "dHy": "pdRHAdy",  # name of dH/dy in gap_con.py
    "dHz": "pdRHAdz",  # name of dH/dz in gap_con.py
    "dDx": "pdB1gx",  # name of dD/dx in gap_con.py
    "dDy": "pdB1gy",  # name of dD/dy in gap_con.py
    "dDz": "pdB1gz"   # name of dD/dz in gap_con.py
}