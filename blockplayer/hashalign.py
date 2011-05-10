import numpy as np
import preprocess
import opencl
import lattice
import config
import os
import dataset
import stencil
import config
import speedup_cy


def features_weave(Ar):
    assert Ar.flags['C_CONTIGUOUS']
    assert Ar.shape[0] == Ar.shape[2]
    code = """
    int WYWZ = (int)WY*(int)WZ;
    int maxlen = (int)(WX-1)*(int)WY*(int)WZ - 1;
    for (int i = 0; i < maxlen; i++) {
        int a0 = Ar[i];
        int a1 = Ar[i+1];
        int a2 = Ar[i+WYWZ];
        if (a0 && !a1 && !a2) {
            output[i] = 1;
        }
    }
    """
    WX, WY, WZ = Ar.shape
    output = np.zeros(Ar.shape, 'i')
    import scipy.weave
    scipy.weave.inline(code, ['Ar','output','WX','WY','WZ'])
    return output


def find_features(A):
    """Find all the features in a block grid, .e. horizontal 'corners'
    Args:
        A: an (x,y,z) binary array
    Returns:
        [(x,y,z,r)] that can be used to apply a correction
    """
    global inds
    inds = []
    #import scipy.ndimage
    #kernel = np.array([[0,-1],[-1,1]],'f')
    #for r in range(4):
    #    Ary = np.swapaxes(np.rot90(np.swapaxes(A, 1, 2), r), 1, 2).astype('f')
    #    Ary = np.ascontiguousarray(Ary)
    #    for y in range(A.shape[1]):
    #        Ar = Ary[:,y,:]
    #        cr = scipy.ndimage.convolve(Ar, kernel)
    #        xz = np.array(np.nonzero(cr==1)).transpose()
    #        inds += [(x,y,z,r) for x,z in xz]

    inds_weave = []
    for r in range(4):
        Ary = np.swapaxes(np.rot90(np.swapaxes(A, 1, 2), r), 1, 2).astype('f')
        Ary = np.ascontiguousarray(Ary)
        cr = features_weave(Ary)
        xz = np.nonzero(cr)
        inds_weave += [(x,y,z,r) for x,y,z in zip(*xz)]

    inds = inds_weave
    #assert set(inds_weave) == set(inds)

    global mask_grid
    mask_grid = A*0
    for x,y,z,r in inds_weave:
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
    ymax = max([y for _,y,_,_ in A])
    yA = [[_ for _ in A if _[1] == y] for y in range(ymax)]
    yB = [[_ for _ in B if _[1] == y] for y in range(ymax)]
    d = {}
    for y in range(ymax):
        for a in yA[y]:
            for b in yB[y]:
                m = diff_coord(a,b, sq)
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

    def error(occA, vacA, occB, vacB):
        return np.sum(np.minimum(vacB,occA) + np.minimum(occB,vacA) -
                      np.minimum(occB,occA)/2.)+100

    def error_weave(occA, vacA, occB, vacB, bx, bz, term):
        code = """
        int term_ = ((int)term-100)*2;
        int total_ = 0;
        for (int i = (int)lower; i < (int)upper; i++) {
            int oA = occA[i];
            int oB = occB[i+(int)offset];
            if (!oA && !oB) continue;
            int vA = vacA[i];
            int vB = vacB[i+(int)offset];
            total_ += (int)(2*(vB&&oA) + 2*(oB&&vA) - (oB&&oA));
            if (total_ > term_) break;
        }
        total[0] = (float)total_/2 + 100;
        """
        offset = -(bz + bx*occA.shape[2]*occA.shape[1])
        lower = 0 if offset>0 else -offset
        upper = occA.shape[0]*occA.shape[1]*occA.shape[2]
        upper = upper if offset<0 else upper-offset
        lower, upper, offset = map(np.int32, (lower, upper, offset))
        total = np.array([0],'f')
        #print lower, upper

        assert occA.flags['C_CONTIGUOUS']
        assert vacA.flags['C_CONTIGUOUS']
        assert occB.flags['C_CONTIGUOUS']
        assert vacB.flags['C_CONTIGUOUS']

        import scipy.weave
        scipy.weave.inline(code, ['occA', 'vacA', 'occB', 'vacB',
                                  'bx', 'bz', 'total', 'term',
                                  'upper', 'lower', 'offset'])
        return total[0]

    A,B = occA, occB
    assert A.shape[0] == A.shape[2] == B.shape[0] == B.shape[2]
    featureA = find_features(A)
    featureB = find_features(B)
    featureA_cy = speedup_cy.find_features(A.astype('u1'))
    featureB_cy = speedup_cy.find_features(B.astype('u1'))
    assert np.all(np.array(featureA) == featureA_cy)
    featureA = speedup_cy.find_features(A.astype('u1'))
    featureB = speedup_cy.find_features(B.astype('u1'))

    matches = speedup_cy.match_features(featureA, featureB, A.shape[0])
    #matches_ = speedup_cy.match_features(featureA, featureB, A.shape[0])

    # Sort the matches by number of corroborating features
    matches = sorted(matches, key=lambda m: matches[m], reverse=True)

    # Group the matches by the rotation
    matches = [[m for m in matches if m[3] == r] for r in range(4)]

    # Store a measure of how each candidate rotation lines up with
    # the current estimate
    dotscores = [1, 1, 1, 1]
    if not R_aligned is None and not prev_R_Correct is None:
        for r in range(4):
            R = correction2modelmat(R_aligned, 0, 0, 0, r)
            # The score is the angle between the forward vectors of the
            # proposed matrix and the previous estimated matrix
            dotscores[r] = 1 + 3*(1 - (np.dot(R[:3,0], prev_R_Correct[:3,0])))

    bestmatch = 4*[None]
    besterror = 4*[10000]

    # Only consider the top 8 most corroborated feature matches
    for r in range(4):
        oBr = np.ascontiguousarray(np.swapaxes(
            np.rot90(np.swapaxes(occB, 1, 2), r), 1, 2))
        vBr = np.ascontiguousarray(np.swapaxes(
            np.rot90(np.swapaxes(vacB, 1, 2), r), 1, 2))
        for match in matches[r][:4]:
            #oB = apply_correction(occB, *match)
            #vB = apply_correction(vacB, *match)
            #err_np = error(occA, vacA, oB, vB)

            bx,_,bz,_ = match
            #err = error_weave(occA, vacA, oBr, vBr, bx, bz, term=besterror[r])
            err = speedup_cy.grid_error(occA.astype('u1'),
                                        vacA.astype('u1'),
                                        oBr.astype('u1'),
                                        vBr.astype('u1'), bx, bz,
                                        term=besterror[r])
            #assert err_ == err
            #print [bx, bz, r], err, err_np
            #err = err_np
            #assert err == err_weave

            if err < besterror[r]:
                bestmatch[r] = match
                besterror[r] = err

    # Use the previouse_estimate score with a weight to select the best
    bestind = np.argmin(np.array(besterror)+10*np.array(dotscores))
    return bestmatch[bestind], besterror[bestind]
