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
import expmap
import scipy
import scipy.optimize
import scipy.ndimage
import pylab
from OpenGL.GL import *
import calibkinect
import opencl
import config

import os
import ctypes
speedup = np.ctypeslib.load_library('speedup_ctypes.so',
                                    os.path.dirname(__file__))

matarg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
speedup.normals.argtypes = [matarg, matarg, matarg,
                            matarg, matarg, matarg,
                            matarg, ctypes.c_int, ctypes.c_int]

def normals_opencl(depth, mask=None, rect=((0,0),(640,480)), win=6):
    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    opencl.set_rect(rect)
    depth = from_rect(depth,rect)
    (l,t),(r,b) = rect
    assert depth.dtype == np.uint16
    assert depth.shape == (b-t, r-l)
    #depth[depth==0] = -1e8  # 2047 this is taken care of by recip_depth

    if mask is None:
        mask = np.ones((b-t,r-l),'bool')
    else:
        mask = np.array(from_rect(mask,rect))

    global filt
    depth = np.ascontiguousarray(depth)
    depth = calibkinect.recip_depth_openni(depth)
    filt = scipy.ndimage.uniform_filter(depth,win)
    opencl.load_filt(filt)
    opencl.load_raw(depth)
    opencl.load_mask(mask)
    return opencl.compute_normals().wait()


def normal_show(nx,ny,nz):
    return np.dstack((nx/2+.5,ny/2+.5,nz/2+.5))


def normals_numpy(depth, rect=((0,0),(640,480)), win=7, mat=None):
    assert depth.dtype == np.float32
    from scipy.ndimage.filters import uniform_filter
    (l,t),(r,b) = rect
    v,u = np.mgrid[t:b,l:r]
    depth = depth[v,u]
    depth[depth==0] = -1e8  # 2047
    depth = calibkinect.recip_depth_openni(depth)
    depth = uniform_filter(depth, win)
    global duniform
    duniform = depth

    dx = (np.roll(depth,-1,1) - np.roll(depth,1,1))/2
    dy = (np.roll(depth,-1,0) - np.roll(depth,1,0))/2
    #dx,dy = np.array(depth),np.array(depth)
    #speedup.gradient(depth.ctypes.data, dx.ctypes.data,
    # dy.ctypes.data, depth.shape[0], depth.shape[1])

    X,Y,Z,W = -dx, -dy, 0*dy+1, -(-dx*u + -dy*v + depth).astype(np.float32)

    mat = calibkinect.projection().astype('f').transpose()
    mat = np.ascontiguousarray(mat)
    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]
    w = np.sqrt(x*x + y*y + z*z)

    x,y,z = (_ / w for _ in x,y,z)

    w[z<0] *= -1
    weights = z*0+1
    weights[depth<-1000] = 0
    weights[z<.1] = 0

    x_ = x*mat[0,0] + y*mat[0,1] + z*mat[0,2]
    y_ = x*mat[1,0] + y*mat[1,1] + z*mat[1,2]
    z_ = x*mat[2,0] + y*mat[2,1] + z*mat[2,2]

    #return x/w, y/w, z/w
    return np.dstack((x_,y_,z_)), weights


def normals_c(depth, rect=((0,0),(640,480)), win=7):
    assert depth.dtype == np.uint16
    from scipy.ndimage.filters import uniform_filter
    (l,t),(r,b) = rect
    v,u = np.mgrid[t:b,l:r]
    depth = depth[v,u]
    depth = calibkinect.recip_depth_openni(depth)
    output_ = np.empty(depth.shape, 'f')
    uniform_filter(depth, win, output=output_)
    depth = output_

    x,y,z = [np.empty_like(depth) for i in range(3)]

    mat = calibkinect.projection().astype('f').transpose()
    mat = np.ascontiguousarray(mat)

    speedup.normals(depth.astype('f'), u.astype('f'), v.astype('f'), x, y, z,
                    mat, depth.shape[0], depth.shape[1])

    weights = z*0+1
    weights[depth<-1000] = 0
    weights[z<.1] = 0

    return np.dstack((x,y,z)), weights


def color_axis(normals,d=0.1):
    #n = np.log(np.power(normals,40))
    X,Y,Z = [normals[:,:,i] for i in range(3)]
    cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
    cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
    return [c * 0.8 + 0.2 for c in cx]
