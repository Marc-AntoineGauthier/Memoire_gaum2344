from numpy import pi
params = {
    "taille": 50000,  # number of points on one side of the BZ
    "n_group": 100,  # number of division of the BZ
    "model": "RHA",  # name of the model to use
    "gap_name": "B1g",  # a name given to the gap for the save file
    "n_gap": 1,  # number of gap
    "gap1": "B1g",  # name of the first gap
    "gap_n1": 0.32,  # norm for the first gap
    "Tc1": 1.5,  # Tc for the first gap
    "gap2": "A2g",  # name of the second gap
    "gap_n2": 0.32,  # norm for the second gap
    "Tc2": 1.5,  # Tc for the second gap
    "phase": pi/2,  # phase between the two gaps (applied to gap2)
    "approx": "BCS",  # name of the approximation for the temperature
    # dependence
    
}
