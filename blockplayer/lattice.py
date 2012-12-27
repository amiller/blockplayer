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
from config import LW
import opencl

from functools import partial
def from_rect(m,rect):
  (l,t),(r,b) = rect
  return m[t:b,l:r]

def color_axis(X,Y,Z,w,d=0.3):
  X2,Y2,Z2 = X*X,Y*Y,Z*Z
  d = 1/d
  cc = Y2+Z2, Z2+X2, X2+Y2
  # I changed this, so it's possible the threshold should be adjusted
  #cx = [w*np.maximum(1.0-(c*d)**2),0*c) for c in cc]
  cx = [w*np.maximum(1.0-(c*(d**2)),0*c) for c in cc]
  return [c for c in cx]

# Find the circular mean, using weights
def cmean(sum_i, sum_j, count, modulo):
  i, j = sum_i / count, sum_j / count
  q = np.arctan2(j,i) / (2*np.pi) * modulo
  if np.isnan(q): q = 0
  return q, np.sqrt(i**2 + j**2), count

# Another form of circular mean
def circular_mean(data, modulo):
  angle = data / modulo * np.pi * 2
  j = np.sin(angle)
  i = np.cos(angle)
  return cmean(i.sum(), j.sum(), len(angle), modulo)

def project(X, Y, Z, mat):
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w


def orientation_opencl(noshow=None):
  """Find the orientation of the lattice using the (labeled) surface normals.
      (see normals.normals_opencl)
  """
  # we may use any arbitrary vector v_ as a basis in XZ
  v_ = -np.array([0,0,1])
  v1 = np.array([0,1,0])
  v2 = np.cross(v1,v_); v2 = (v2 / np.sqrt(np.dot(v2,v2)))
  v0 = np.cross(v1,v2)
  mat = np.hstack((np.vstack((v0,v1,v2)),[[0],[0],[0]]))

  opencl.compute_flatrot(mat.astype('f'))
  sq = opencl.reduce_flatrot()

  qqx = sq[0] / sq[3]
  qqz = sq[2] / sq[3]
  angle = np.arctan2(qqz,qqx)/4
  q0 = np.cos(angle) * v0 + np.sin(angle) * v2
  q0 /= np.sqrt(np.dot(q0,q0))
  q2 = np.cross(q0,v1)

  # Build an output matrix out of the components
  mat = np.eye(4,4,dtype='f')
  mat[:3,:3] = np.vstack((q0,v1,q2))
  return mat


def orientation_numpy(normals,weights):

  def flatrot():
    dx,dy,dz = np.rollaxis(normals,2)

    # Use the quadruple angle formula to push everything around the
    # circle 4 times faster, like doing mod(x,pi/2)
    qz = 4*dz*dx*dx*dx - 4*dz*dz*dz*dx
    qx = dx*dx*dx*dx - 6*dx*dx*dz*dz + dz*dz*dz*dz

    # Build the weights using a threshold, finding the normals lying on
    # the XZ plane
    global cx, qqx, qqz
    #cx = np.max((1.0-dy*dy/(d*d), 0*dy),0)
    cx = dy < 0.3
    w = weights * cx

    sq = map(np.nansum, (w*qx, 0, w*qz, w))
    return sq

  sq = flatrot()

  # Project the normals against the XZ plane
  v_ = -np.array([0,0,1])
  v1 = np.array([0,1,0])
  v2 = np.cross(v1,v_); v2 = (v2 / np.sqrt(np.dot(v2,v2)))
  v0 = np.cross(v1,v2)
  mat = np.hstack((np.vstack((v0,v1,v2)),[[0],[0],[0]]))

  qqx = sq[0] / sq[3]
  qqz = sq[2] / sq[3]
  angle = np.arctan2(qqz,qqx)/4
  q0 = np.cos(angle) * v0 + np.sin(angle) * v2
  q0 /= np.sqrt(np.dot(q0,q0))
  q2 = np.cross(q0,v1)

  # Build an output matrix out of the components
  mat = np.eye(4,4,dtype='f')
  mat[:3,:3] = np.vstack((q0,v1,q2))
  return mat


def translation_opencl(R_oriented):
  """
  Params:
      mat:
  Returns:
      modelmat: a 4x4 matrix

  global state:
      meanx,meany are the 'fixes' used to transform opencl.get_modelxyz()
      into true model coordinates. The values of opencl.get_modelxyz() do
      not reflect the correct value of modelmat. The modelmat includes a
      correction by [-meanx, 0, -meany] that must be applied to modelxyz,
      and passed as a parameter to opencl.computegridinds.
  """
  assert R_oriented.dtype == np.float32
  global modelmat
  modelmat = np.array(R_oriented)

  # Returns warped coordinates, and sincos values for the lattice
  opencl.compute_lattice2(modelmat[:3,:4], LW)

  global meanx,meanz,cxyz_,qx2qz2, dmx, dmz, countx, countz
  cxyz_,qx2qz2 = opencl.reduce_lattice2()
  meanx,dmx,countx = cmean(*qx2qz2[:2],count=cxyz_[0],modulo=LW)
  meanz,dmz,countz = cmean(*qx2qz2[2:],count=cxyz_[2],modulo=LW)
  modelmat[:,3] -= np.array([meanx, 0, meanz, 0])

  return modelmat


def is_valid_estimate():
  global dmx, dmz, countx, countz
  return (dmx >= 0.7 and dmz >= 0.7 and
          countx > 100 and countz >= 100)


def translation_numpy(rimg, R_oriented):
  """
  Assuming we know the tableplane, find the rotation and translation in
  the XZ plane.
  """
  assert R_oriented.shape == (4,4)
  assert R_oriented.dtype == np.float32

  # Project normals from camera space to model space (axis-aligned)
  global dx,dy,dz
  n,w = rimg.normals, rimg.weights
  dx,dy,dz = np.rollaxis(np.dot(n, R_oriented[:3,:3].T),2)

  # Threshold the axis-aligned normals to label them (P_oriented)
  global cx,cy,cz
  CLIM = 0.9486
  step = lambda edge, x: (x >= edge).astype('i1')
  thresh = lambda x: (step(x, -CLIM) + step(x, CLIM) - 1) * (w>0)
  cx,cy,cz = map(thresh, (dx,dy,dz))

  # Transform points P_camera to P_oriented
  global P_oriented
  XYZ_ = np.dot(rimg.xyz, R_oriented[:3,:3].T) + R_oriented[:3,3]
  X,Y,Z = map(partial(from_rect, rect=rimg.rect),
              np.rollaxis(XYZ_,2))
  P_oriented = np.dstack((X,Y,Z))
  
  # Find the circular mean and solve for R_aligned
  global meanx, meany, meanz
  global dmx, dmy, dmz
  global countx, county, countz
  meanx,dmx,countx = circular_mean(X[cx!=0],LW)
  meanz,dmz,countz = circular_mean(Z[cz!=0],LW)

  R_aligned = np.copy(R_oriented)
  R_aligned[:,3] -= np.array([meanx, 0, meanz, 0])

  # Compute P_aligned
  global P_aligned
  P_aligned = P_oriented.copy()
  P_aligned[:,:,0] -= meanx
  P_aligned[:,:,2] -= meanz

  # Stacked data in model space
  global XYZ, dXYZ, cXYZ
  XYZ = np.rollaxis(P_aligned, 2)
  dXYZ = ((dx, dy, dz))
  cXYZ = ((cx, cy, cz))

  # XYZo are just points for show
  # They are intended to match the get_xyz points from opencl
  global XYZo
  XYZo = map(partial(from_rect, rect=rimg.rect), 
             np.rollaxis(rimg.xyz,2))

  return R_aligned
