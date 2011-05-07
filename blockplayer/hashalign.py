import numpy as np
import preprocess
import opencl
import lattice
import config
import os
import dataset
import stencil
import config


def find_features(A):
    """Find all the features in a block grid, .e. horizontal 'corners'
    Args:
        A: an (x,y,z) binary array
    Returns:
        [(x,y,z,r)] that can be used to apply a correction
    """
    import scipy.ndimage
    global inds
    inds = []
    for y in range(A.shape[1]):
        for r in range(4):
            Ar = np.rot90(A[:,y,:], r).astype('f')
            kernel = np.array([[0,-1],[-1,1]],'f')

            cr = scipy.ndimage.convolve(Ar, kernel)
            xz = np.array(np.nonzero(cr==1)).transpose()
            inds += [(x,y,z,r) for x,z in xz]

    global mask_grid
    mask_grid = A*0
    for x,y,z,r in inds:
        for _ in range(r):
            x,z = z, A.shape[0]-x-1
        mask_grid[x,y,z]=r
    return inds


def diff_coord(a, b, ashape):
    xa,ya,za,ra = a
    x,y,z,r = b
    x,y,z = xa-x,ya-y,za-z
    for _ in range(ra):
        x,z = z,-x
    return x,y,z,(r-ra)%4


def apply_correction(grid, bx, by, bz,rot):
    return np.roll(np.roll(\
            np.swapaxes(np.rot90(np.swapaxes(grid,1,2),rot),1,2),
        bx, 0), bz, 2)


def match_features(A, B, sq):
    matches = []
    ymax = max([y for _,y,_,_ in A])
    for y in range(ymax):
        for a in [a for a in A if a[1]==y]:
            for b in [b for b in B if b[1] == y]:
                matches += [tuple(diff_coord(a,b, sq))]
    d = {}
    for m in matches:
        d.setdefault(m,0)
        d[m] += 1
    return d


def correction2modelmat(R_aligned, x, y, z, r):
    import expmap
    R = expmap.axis2rot(np.array([0,-r*np.pi/2,0]))

    R_correct = R_aligned.copy()
    R_correct[:3,:] = np.dot(R, R_correct[:3,:])
    R_correct[:3,3] += [x*config.LW, 0, z*config.LW]
    return R_correct


def find_best_alignment(occA, vacA, occB, vacB,
                        R_aligned=None, prev_R_Correct=None):
    A,B = occA, occB
    assert A.shape[0] == A.shape[2] == B.shape[0] == B.shape[2]
    featureA = find_features(A)
    featureB = find_features(B)
    matches = match_features(featureA, featureB, A.shape[0])
    matches = sorted(matches, key=lambda m: matches[m], reverse=True)

    def error(occA, vacA, occB, vacB):
        return np.sum(np.minimum(vacB,occA) + np.minimum(occB,vacA) -
                      np.minimum(occB,occA)/2.)+100

    # Store a measure of how each candidate rotation lines up with
    # the current estimate
    dotscores = [1, 1, 1, 1]
    if 1 and not R_aligned is None and not prev_R_Correct is None:
        for r in range(4):
            R = correction2modelmat(R_aligned, 0, 0, 0, r)
            # The score is the angle between the forward vectors of the
            # proposed matrix and the previous estimated matrix
            dotscores[r] = 1 + 3*(1 - (np.dot(R[:3,0], prev_R_Correct[:3,0])))

    bestmatch = 4*[None]
    besterror = 4*[np.inf]
    for match in matches:
        _,_,_,r = match
        oB = apply_correction(occB, *match)
        vB = apply_correction(vacB, *match)
        err = error(occA, vacA, oB, vB)
        if err < besterror[r]:
            bestmatch[r] = match
            besterror[r] = err

    #print dotscores
    bestind = np.argmin(np.array(besterror)+10*np.array(dotscores))
    return bestmatch[bestind], besterror[bestind]
