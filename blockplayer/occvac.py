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
import opencl
import config
import speedup_cy

def setup_grid():
    global gridmin, gridmax
    gridmin = np.zeros((4,),'f')
    gridmax = np.zeros((4,),'f')
    gridmin[:3] = config.bounds[0]
    gridmax[:3] = config.bounds[1]


def occvac_numpy(gridinds):
    """Params:
    """
    inds = gridinds[gridinds[:,0,3]!=0,:,:3]
    bins = [np.arange(0,gridmax[i]-gridmin[i]+1) for i in range(3)]
    occH,_ = np.histogramdd(inds[:,0,:], bins)
    vacH,_ = np.histogramdd(inds[:,1,:], bins)
    vac = vacH>30
    occ = occH>30
    #print 'occvac_numpy', hash(occ.tostring()), hash(vac.tostring())
    return occ, vac

def occvac_cython(gridinds):
    shape = [gridmax[i]-gridmin[i] for i in range(3)]
    occ = np.zeros(shape, 'u1')
    vac = np.zeros(shape, 'u1')
    speedup_cy.occvac(gridinds, occ, vac,
                      gridmin.astype('i'),
                      gridmax.astype('i'))
    occ, vac = occ.astype('bool'), vac.astype('bool')
    #print 'occvac_cython', hash(occ.tostring()), hash(vac.tostring())
    return occ, vac


def carve_numpy(xfix, zfix, P_aligned, cxyz):
    setup_grid()

    global cx,cy,cz
    cx,cy,cz = cxyz
    cxyz = np.dstack(cxyz)
    xyz = P_aligned
    f1 = cxyz*0.5

    mod = np.array([config.LW, config.LH, config.LW])
    gridinds = np.floor(-gridmin[:3] + xyz/mod + f1)
    gridinds = np.array((gridinds, gridinds - cxyz), 'i1')
    gridinds = np.rollaxis(gridinds.reshape(2, -1, 3), 1)
    inds = np.zeros((gridinds.shape[0], 2, 4), 'i1')
    inds[:,:,:3] = gridinds
    inds[:,0,3] = np.dot(cxyz.reshape(-1,3),(4,2,1))
    inds[:,1,3] = np.dot(cxyz.reshape(-1,3),(4,2,1))
    assert len(inds.shape) == 3 and inds.shape[1] == 2 and inds.shape[2] == 4
    gi = inds[:,:,3]; print gi.dtype, gi.shape, gi.sum(), gi.min(), gi.max()
    #print 'carve_numpy', hash(inds.tostring())
    if len(inds) == 0: return None
    occvac_cython(inds)
    return occvac_numpy(inds)

def carve_opencl(xfix, zfix):
    setup_grid()

    opencl.compute_gridinds(xfix,zfix, config.LW, config.LH, 
                            gridmin, gridmax)
    inds = opencl.get_gridinds()
    gi = inds[:,:,3]; print gi.dtype, gi.shape, gi.sum(), gi.min(), gi.max()
    assert len(inds.shape) == 3 and inds.shape[1] == 2 and inds.shape[2] == 4
    #print 'carve_opencl', hash(inds.tostring())
    if len(inds) == 0: return None
    occvac_cython(inds)
    return occvac_numpy(inds)
