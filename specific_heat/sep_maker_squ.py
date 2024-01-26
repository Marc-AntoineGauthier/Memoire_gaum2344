from itertools import product as itp

import numpy as np

from param import params

"""[summary]

sys.argv[1] (int): number of k point on one side of the BZ
sys.argv[2] (int): number of groups to divide the BZ into. 
"""

def is_square(number):
    """Look if a number is a perfect square

    Args:
        number (float): number that we want to look if it is a square

    Returns:
        Bool: True for a perfect square, False otherwise.
    source : 
        https://djangocentral.com/python-program-to-check-if-a-number-is-perfect-square/
    """
    if int(np.sqrt(number)+0.5)**2 == number:
        return True
    else:
        return False

def square_group_maker(taille, n_group):
    """
    Separate an axis into "n_group". This gives the separations for both
    axis.

    Args:
        taille (int): Number of point to be separated
        n_group (int): Number of desired separations

    Returns:
        List [Tuples]: List containing Tuples (starting index, number of
        points).
    """
    if taille%n_group == 0:
        t_group = int(taille/n_group)
        seperators = [(i*t_group, t_group) for i in range(n_group)]
    
    elif taille%(n_group-1) == 0:
        t_group = int(taille/(n_group-1))
        seperators = [(i*t_group, t_group) for i in range(n_group-1)]
        seperators.append((0, 0))
        print("WARNING : Not optimable, some groups will have no k-point.")
    
    else:
        t_group = int(taille/(n_group-1))
        seperators = [(i*t_group, t_group) for i in range(n_group-1)]
        seperators.append((t_group*(n_group-1), taille%(n_group-1)))
    return seperators

def rectangle_group_maker(taille, n_group):
    """
    Separates a square k-grid into n_groups of rectangles

    Args:
        taille (int): Number of points on ne side of the k-grid
        n_group (int): Number of groups to divide the k-grid

    Returns:
        List [Tuple]: A list of Tuples (start index x, number of point
        x, start index y, number of point y)
    """
    sep_square = []
    sep_reste = []
    if is_square(n_group):
        n_div = int(np.sqrt(n_group))
        sep_square = square_group_maker(taille, n_div)
    
    elif is_square(n_group-1):
        # the idea is to make the other group a complete row on top of
        # the grid of a square number. This means that the grid will
        # not be squares for all of its parts.
        group_squ = n_group - 1
        N = taille*taille
        hauteur = 1
        Trouve = False
        pre_diff = np.inf
        while not(Trouve):
            y = hauteur*taille
            x = (N-y)/group_squ
            diff = np.abs(x-y)
            if diff < pre_diff:
                pre_diff = diff
                hauteur += 1
            else:
                Trouve = True
                hauteur -= 1  # we go back to the optimal one
        # making the seperations in x
        n_sep_x = int(np.sqrt(group_squ))
        n_x = int(taille/n_sep_x)
        sep_x = [[0, n_x] for i in range(n_sep_x)]
        for i in range(taille%n_sep_x):
            sep_x[i][1] += 1
        before = 0
        for i in sep_x:
            i[0] = before
            before += i[1]
        # making the seperations in y
        taille_y = taille - hauteur
        n_sep_y = n_sep_x
        n_y = int(taille_y/n_sep_y)
        sep_y = [[0, n_y] for i in range(n_sep_y)]
        for i in range(taille_y%n_sep_y):
            sep_y[i][1] += 1
        before = 0
        for i in sep_y:
            i[0] = before
            before += i[1]
        # making the groups
        for x, y in itp(sep_x, sep_y):
            sep_reste.append(
                (x[0], x[1], y[0], y[1])
            )
        # adding the last group
        sep_reste.append(
            (0, taille, taille - hauteur, hauteur)
        )
        pass

    else:
        N = taille*taille
        n_div = int(np.sqrt(n_group))
        group_squ = n_div*n_div
        surplus = taille%n_div
        Trouve = False
        pre_diff = np.inf
        while not(Trouve):
            x = ((taille-surplus)/(n_div))**2
            y = (N-x*group_squ)/(n_group-group_squ)
            diff = np.abs(x-y)
            if diff < pre_diff:
                pre_diff = diff
                surplus += n_div
            else:
                Trouve = True
                surplus -= n_div  # we go back to the optimal one
        # now we can do the squares
        sep_square = square_group_maker(taille-surplus, n_div)
        # we now seprate what is left
        # here up refer to the upper rectangle and down to the rectangle
        # on the right; we are not dealing with spins.
        taille_up_x = taille - surplus
        n_reste = n_group - group_squ
        print("number of groups not in the square : ", n_reste)
        n_sep_up = int(n_reste/2)
        n_up = int(taille_up_x/n_sep_up)
        sep_up = [[0, n_up] for i in range(n_sep_up)]
        for i in range(taille_up_x%n_sep_up):
            sep_up[i][1] += 1
        before = 0
        for i in sep_up:
            i[0] = before
            before += i[1]
        
        n_sep_down = int(n_reste/2) + n_reste%2
        n_down = int(taille/n_sep_down)
        sep_down = [[0, n_down] for i in range(n_sep_down)]
        for i in range(taille%n_sep_down):
            sep_down[i][1] += 1
        before = 0
        for i in sep_down:
            i[0] = before
            before += i[1]

        for sep in sep_up:
            sep_reste.append(
                (sep[0], sep[1], taille - surplus, surplus)
            )

        for sep in sep_down:
            sep_reste.append(
                (taille - surplus, surplus, sep[0], sep[1])
            )
        print("groups not in the square : ", sep_reste)
    # here we make the delimetation for the square group
    all_sep = []
    for i,j in itp(sep_square, repeat=2):
        all_sep.append((i[0], i[1], j[0], j[1]))
    # putting the two groups together
    for sep in sep_reste:
        all_sep.append(sep)
    return all_sep

# making the limits to be used with the generators and saving into a
# file
taille = params["taille"]
n_groups = params["n_group"]

groups = rectangle_group_maker(taille, n_groups)
#print(groups)
inter = ((0.5+0.5)/(taille))  # from k_gen_cut of "k_gen.py"
output = []
for g in groups:
    start_x = g[0]*inter - (0.5-inter)
    start_y = g[2]*inter - (0.5-inter)
    output.append(
        [
            start_x, start_x + g[1]*inter-inter, g[1],
            start_y, start_y + g[3]*inter-inter, g[3]
        ]
    )
# need some confirmation that everything works and saving into a file
valide = 0
for g in output:
    valide += g[2]*g[5]
N = taille*taille
print(
    "Same number of point : {} (given : {}, summed : {})".format(
        valide == N, N, valide
        )
    )

np.savetxt(
    f"sep_for_{taille}_{n_groups}_BZ", output
)
