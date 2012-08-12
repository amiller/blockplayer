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

cimport numpy as np
import numpy as np
import calibkinect


cdef convert_(np.uint16_t *depth,
              np.float32_t *X,
              np.float32_t *Y,
              np.float32_t *mat,
              np.float32_t *lut,
              np.float32_t *x,
              np.float32_t *y,
              np.float32_t *z,
              int N):
    cdef int i
    cdef float Z, x_, y_, z_, w
    for i in range(N):
        Z = lut[depth[i]];
        x_ = X[i]*mat[0] + Y[i]*mat[1] + Z*mat[2] + mat[3];
        y_ = X[i]*mat[4] + Y[i]*mat[5] + Z*mat[6] + mat[7];
        z_ = X[i]*mat[8] + Y[i]*mat[9] + Z*mat[10] + mat[11];
        w = X[i]*mat[12] + Y[i]*mat[13] + Z*mat[14] + mat[15];
        w = 1/w;
        x[i] = x_*w;
        y[i] = y_*w;
        z[i] = z_*w;


def convertOpenNI2Real(depth, u=None, v=None,
                       mat=calibkinect.projection()):

    assert mat.dtype == np.float32, "mat must be np.float32"
    assert mat.shape == (4,4), "mat must be 4x4"
    assert mat.flags['C_CONTIGUOUS'], "mat must be contiguous"
    assert depth.dtype == np.uint16, "depth must be np.uint16"

    if u is None or v is None: v,u = calibkinect.full_vu
    assert depth.shape == u.shape == v.shape, "depth and v,u must match"

    cdef np.ndarray[np.float32_t,ndim=2] X = u
    cdef np.ndarray[np.float32_t,ndim=2] Y = v
    cdef np.ndarray[np.float32_t,ndim=2] x = np.empty(depth.shape, 'f')
    cdef np.ndarray[np.float32_t,ndim=2] y = np.empty(depth.shape, 'f')
    cdef np.ndarray[np.float32_t,ndim=2] z = np.empty(depth.shape, 'f')

    N = np.prod(depth.shape)

    cdef np.ndarray[np.uint16_t,ndim=2] depth_ = depth
    cdef np.ndarray[np.float32_t,ndim=2] mat_ = mat
    cdef np.ndarray[np.float32_t] lut = calibkinect.lut
    
    convert_(<np.uint16_t *>depth_.data,
             <np.float32_t *>X.data,
             <np.float32_t *>Y.data,
             <np.float32_t *>mat_.data,
             <np.float32_t *>lut.data,
             <np.float32_t *>x.data,
             <np.float32_t *>y.data,
             <np.float32_t *>z.data,
             N)
    return x,y,z

