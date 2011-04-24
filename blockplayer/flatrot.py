import numpy as np
import opencl
import expmap
import pylab
from OpenGL.GL import *


def lattice_orientation_opencl(noshow=None):
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

  global dm
  dm = np.sqrt(qqx**2 + qqz**2)

  # Build an output matrix out of the components
  mat = np.eye(4,4,dtype='f')
  mat[:3,:3] = np.vstack((q0,v1,q2))

  return mat


def lattice_orientation_numpy(normals,weights):

  # Project the normals against the plane
  dx,dy,dz = np.rollaxis(normals,2)

  # Use the quadruple angle formula to push everything around the
  # circle 4 times faster, like doing mod(x,pi/2)
  qz = 4*dz*dx*dx*dx - 4*dz*dz*dz*dx
  qx = dx*dx*dx*dx - 6*dx*dx*dz*dz + dz*dz*dz*dz

  # Build the weights using a threshold, finding the normals lying on
  # the XZ plane
  d=0.3
  global cx, qqx, qqz
  cx = np.max((1.0-dy*dy/(d*d), 0*dy),0)
  w = weights * cx

  qqx = np.nansum(w*qx) / w.sum()
  qqz = np.nansum(w*qz) / w.sum()
  angle = np.arctan2(qqz,qqx)/4

  q0 = np.array([np.cos(angle), 0, np.sin(angle)])
  q0 /= np.sqrt(np.dot(q0,q0))
  q2 = np.cross(q0,np.array([0,1,0]))

  # Build an output matrix out of the components
  mat = np.vstack((q0,np.array([0,1,0]),q2))
  axes = expmap.rot2axis(mat)

  return axes
