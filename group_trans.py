import numpy as np
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=2)
dpi = 2*np.pi
z = np.zeros((6, 6))
C4k = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])

sxk = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

syk = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

szk = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

C4s = np.array([
    [(1+1j)/np.sqrt(2), 0],
    [0, (1-1j)/np.sqrt(2)]
])

sxs = np.array([
    [0, 1j],
    [1j, 0]
])

sys = np.array([
    [0, 1],
    [-1, 0]
])

szs = np.array([
    [1j, 0],
    [0, -1j]
])

C4l = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, -1]
])

sxl = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

syl = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

szl = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

# Defining the transformations with the generator for the l and s space

El = [C4l @ C4l @ sxl @ syl]
C4ll = [C4l, C4l@ C4l @ C4l]  # the l is for list so the two variables are distinct
C2l = [C4l @ C4l]
C2pl = [C4l @ C4l @ szl @ sxl, C4l @ C4l @ szl @ syl]
C2ppl = [C4l @ sxl @ szl, C4l @ syl @ szl]
invl = [sxl @ syl @ szl]
S4l = [C4l @ szl, C4l @ C4l @ C4l @ szl]  # the second one can be verified
szll = [szl]  # the l is for list so the two variables are distinct
sxyl = [sxl, syl]
sdl = [C4l @ sxl, C4l @ syl]

Es = [C4s @ C4s @ sxs @ sys]
C4sl = [C4s, C4s @ C4s @ C4s]  # the l is for list so the two variables are distinct
C2s = [C4s @ C4s]
C2ps = [C4s @ C4s @ szs @ sxs, C4s @ C4s @ szs @ sys]
C2pps = [C4s @ sxs @ szs, C4s @ sys @ szs]
invs = [sxs @ sys @ szs]
S4s = [C4s @ szs, C4s @ C4s @ C4s @ szs]  #the second one can be verified
szsl = [szs]
sxys = [sxs, sys]
sds = [C4s @ sxs, C4s @ sys]

# Defining the transformation for k
Ek = [C4k @ C4k @ sxk @ syk]
C4kl = [C4k, C4k @ C4k @ C4k]  # the l is for list so the two variables are distinct
C2k = [C4k @ C4k]
C2pk = [C4k @ C4k @ szk @ sxk, C4k @ C4k @ szk @ syk]
C2ppk = [C4k @ sxk @ szk, C4k @ syk @ szk]
invk = [sxk @ syk @ szk]
S4k = [C4k @ szk, C4k @ C4k @ C4k @ szk]  # the second one can be verified
szkl = [szk]  # the l is for list so the two variables are distinct
sxyk = [sxk, syk]
sdk = [C4k @ sxk, C4k @ syk]

header = [
    "E", "2C4", "C2", "2C2\'", "2C2\'\'", "i", "2S4", "sz", "sx/sy", "sd/d\'"
]
def print_header():
    line = "|"
    for h in header:
        line += "{:^5}|".format(h)
    line += "name"
    print(line)

def charals(term, trans):
    new_term = trans @ term @ np.transpose(trans)
    if np.all(np.isclose(new_term - term, z)):
        return 1
    elif np.all(np.isclose(new_term + term, z)):
        return -1
    else:
        return 999

def print_tablels(term, name=":)"):
    trans = [
        [Es, El], [C4sl, C4ll], [C2s, C2l], [C2ps, C2pl], [C2pps, C2ppl],
        [invs, invl], [S4s, S4l], [szsl, szll], [sxys, sxyl], [sds, sdl]
    ]
    line = "|"
    for tran in trans:
        c = set()
        for t, _ in enumerate(tran):
            c.add(charals(term, np.kron(tran[0][t], tran[1][t])))
        if len(c) == 1:
            line += "{:^5d}|".format(list(c)[0])
        else:
            line += "{:^5d}|".format(888)
    line += name
    print(line)
    return 0

def charak(func, trans):
    p1 = [0.2, 0.1, 0.17]
    p2 = np.linalg.inv(trans) @ p1
    fp1 = func(p1)
    fp2 = func(p2)
    if np.isclose((fp1 - fp2)/fp1, 0):
        return 1
    elif np.isclose((fp1 + fp2)/fp1, 0):
        return -1

def print_tablek(func, name=":)"):
    trans = [
        Ek, C4kl, C2k, C2pk, C2ppk, invk, S4k, szkl, sxyk, sdk
    ]
    line = "|"
    for tran in trans:
        c = set()
        for t in tran:
            c.add(charak(func, t))
        if len(c) == 1:
            line += "{:^5d}|".format(list(c)[0])
        else:
            line += "{:^5d}|".format(888)
    line += name
    print(line)
    return 0
   
def chara(mat, tk, tls):
    p1 = [0.2, 0.1, 0.2]
    p2 = np.linalg.inv(tk) @ p1
    term = mat(p1)
    fp2 = mat(p2)
    new_term = tls @ fp2 @ np.transpose(tls)
    if np.all(np.isclose( new_term - term, z )):
        return 1
    elif np.all(np.isclose( new_term + term, z )):
        return -1
    else:
        #print("-")
        #print(new_term - term)
        #print("+")
        #print(new_term + term)
        return 999

def print_table(mat, name=":)"):
    trans = [
        [Es, El, Ek], [C4sl, C4ll, C4kl], [C2s, C2l, C2k],
        [C2ps, C2pl, C2pk], [C2pps, C2ppl, C2ppk],
        [invs, invl, invk], [S4s, S4l, S4k], [szsl, szll, szkl],
        [sxys, sxyl, sxyk], [sds, sdl, sdk]
    ]
    line = "|"
    for tran in trans:
        c = set()
        for t, _ in enumerate(tran[0]):
            c.add(chara(mat, tran[2][t], np.kron(tran[0][t], tran[1][t])))
        if len(c) == 1:
            line += "{:^5d}|".format(list(c)[0])
        else:
            line += "{:^5d}|".format(888)
    line += name
    print(line)
    return 0
