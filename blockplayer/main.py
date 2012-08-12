# Andrew Miller <amiller@cs.ucf.edu> 2011
#
# BlockPlayer - 3D model reconstruction using the Lattice-First algorithm
# See: 
#    "Interactive 3D Model Acquisition and Tracking of Building Block Structures"
#    Andrew Miller, Brandyn White, Emiko Charbonneau, Zach Kanzler, and Joseph J. LaViola Jr.
#    IEEE VR 2012, IEEE TVGC 2012
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import config
import opencl
import lattice
import grid
import spacecarve
import stencil
import occvac
import dataset
import hashalign

from rtmodel.rangeimage import RangeImage
from rtmodel.camera import kinect_camera

R_display = None


def initialize():
    grid.initialize()
    global R_display, S1, S2
    R_display = S1 = S2 = None

def update_frame(depth, rgb=None):
    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global modelmat, rimg
    bg = config.bg[0]
    cam = config.cameras[0]
    rimg = RangeImage(depth, cam)
    try:
        rimg.threshold_and_mask(bg)
    except IndexError:
        grid.initialize()
        modelmat = None
        return
    rimg.filter()

    # Compute the surface normals
    opencl.set_rect(rimg.rect)
    opencl.load_filt(rimg.depth_filtered)
    opencl.load_raw(rimg.depth_recip)
    opencl.load_mask(from_rect(rimg.mask, rimg.rect).astype('u1'))
    opencl.compute_normals().wait()
    
    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()

    # Use a preferred initial location
    LW = config.LW
    LH = config.LH
    modelmat = R_oriented
    modelmat = np.linalg.inv(modelmat)
    px,py,pz = config.center
    modelmat[:3,3] += [np.round(px/LW)*LW, np.round(py/LH)*LH, np.round(pz/LW)*LW]
    modelmat = np.linalg.inv(modelmat).astype('f')
    R_oriented = modelmat

    R_aligned = lattice.translation_opencl(R_oriented)

    # Use occvac to estimate the voxels from just the current frame
    occ, vac = occvac.carve_opencl()

    # Further carve out the voxels using spacecarve
    warn = np.seterr(invalid='ignore')
    try:
        vac = vac | spacecarve.carve(depth, R_aligned, bg)
    except np.linalg.LinAlgError:
        return
    np.seterr(divide=warn['invalid'])

    if grid.has_previous_estimate() and np.any(grid.occ):
        try:
            c,err = hashalign.find_best_alignment(grid.occ, grid.vac,
                                                  occ, vac,
                                                  R_aligned,
                                                  grid.previous_estimate['R_correct'])
        except ValueError:
            print 'could not align previous'
            return None

        R_correct = hashalign.correction2modelmat(R_aligned, *c)
        occ = occvac.occ = hashalign.apply_correction(occ, *c)
        vac = occvac.vac = hashalign.apply_correction(vac, *c)

    elif np.any(occ):
        # If this is the first estimate (bootstrap) then try to center the grid
        if np.any(grid.occ):
            # Initialize with ground truth
            try:
                c,err = hashalign.find_best_alignment(grid.occ, grid.vac,
                                                      occ, vac, R_aligned)
                R_correct = hashalign.correction2modelmat(R_aligned, *c)
                occ = occvac.occ = hashalign.apply_correction(occ, *c)
                vac = occvac.vac = hashalign.apply_correction(vac, *c)
            except ValueError:
                #print 'could not align bootstrap'
                return None
        else:
            R_correct, occ, vac = grid.center(R_aligned, occ, vac)
            occvac.occ, occvac.vac = occ, vac
    else:
        #print 'nothing happened'
        return

    def matrix_slerp(matA, matB, alphaT=0.01, alphaR=0.01):
        # alpha = 1 means no smoothing is applied
        # alpha = 0 means the value never changes
        if matA is None:
            return matB
        import transformations
        qA = transformations.quaternion_from_matrix(matA)
        qB = transformations.quaternion_from_matrix(matB)
        qC =transformations.quaternion_slerp(qA, qB, alphaR)
        mat = matB.copy()
        mat[:3,:3] = transformations.quaternion_matrix(qC)[:3,:3]
        mat = np.linalg.inv(mat)
        mat[:3,3] = (1-alphaT)*np.linalg.inv(matA)[:3,3] + (alphaT)*np.linalg.inv(matB)[:3,3]
        mat = np.linalg.inv(mat).astype('f')
        return mat

    global R_display
    global S1, S2

    # Double exponential smoothing
    # See: LaViola, 2003
    # http://www.cs.brown.edu/~jjl/pubs/kfvsexp_final_laviola.pdf
    alphaT = 0.2
    alphaR = 0.2
    S1 = matrix_slerp(S1, R_correct, alphaT, alphaR)
    S2 = matrix_slerp(S2, S1, alphaT, alphaR)
    R_display = matrix_slerp(S1, S2, (3*alphaT-1)/(1-alphaT), (3*alphaR-1)/(1-alphaR))
    #R_display = matrix_slerp(R_display, R_correct, alphaT, alphaR)

    occ_stencil, vac_stencil = grid.stencil_carve(depth, rimg.rect,
                                                  R_correct, occ, vac, bg,
                                                  rgb)
    if lattice.is_valid_estimate():
        # Run stencil carve and merge
        color = stencil.RGB if not rgb is None else None
        grid.merge_with_previous(occ, vac, occ_stencil, vac_stencil, color)
    grid.update_previous_estimate(R_correct)
