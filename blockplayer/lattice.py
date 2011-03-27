import numpy as np
import expmap
import scipy
import pylab
from OpenGL.GL import *
import normals
import calibkinect
import preprocess
from pylab import *
import colors
import opencl
import config


def circular_mean(data, modulo):
  """Given data sampled from a periodic function (with known period: modulo),
  find the phase by converting to cartesian coordinates.
  """
  angle = data / modulo * np.pi * 2
  y = np.sin(angle)
  x = np.cos(angle)
  a2 = np.arctan2(y.mean(),x.mean()) / (2*np.pi)
  if np.isnan(a2): a2 = 0
  return a2*modulo


def color_axis(X,Y,Z,w,d=0.3):
  X2,Y2,Z2 = X*X,Y*Y,Z*Z
  d = 1/d
  cc = Y2+Z2, Z2+X2, X2+Y2
  cx = [w*np.maximum(1.0-(c*d)**2,0*c) for c in cc]
  return [c for c in cx]


def project(X, Y, Z, mat):
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w


def lattice2_opencl(mat, init_t=None):
  """
  Returns:
      modelmat: a 4x4 matrix

  global state:
      meanx,meany are the 'fixes' used to transform opencl.get_modelxyz()
      into true model coordinates. The values of opencl.get_modelxyz() do
      not reflect the correct value of modelmat. The modelmat includes a
      correction by [-meanx, 0, -meany] that must be applied to modelxyz,
      and passed as a parameter to opencl.computegridinds.
  """
  from config import LW
  assert mat.shape == (4,4)
  assert mat.dtype == np.float32
  global modelmat
  modelmat = np.array(mat)

  # Returns warped coordinates, and sincos values for the lattice
  opencl.compute_lattice2(modelmat[:3,:4], LW)

  global cx,cz,face
  # If we don't have a good initialization for the model space translation,
  # use the centroid of the surface points.
  if init_t:
    global face
    X,Y,Z,face = np.rollaxis(opencl.get_modelxyz(),1)
    cx,_,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                          dtype='i1').reshape(-1,2),1)

    modelmat[:,3] -= np.round([X[cz!=0].mean()/LW,
                               0,
                               Z[cx!=0].mean()/LW,
                               0])*LW
    opencl.compute_lattice2(modelmat[:3,:4], LW)

  # Find the circular mean, using weights
  def cmean(mxy,c):
    x,y = mxy / c
    a2 = np.arctan2(y,x) / (2*np.pi) * LW
    if np.isnan(a2): a2 = 0
    return a2, np.sqrt(x**2 + y**2), c

  global meanx,meanz,cxyz_,qx2qz2, dmx, dmy, countx, county
  cxyz_,qx2qz2 = opencl.reduce_lattice2()
  meanx,dmx,countx = cmean(qx2qz2[:2],cxyz_[0])
  meanz,dmy,county = cmean(qx2qz2[2:],cxyz_[2])
  modelmat[:,3] -= np.array([meanx, 0, meanz, 0])

  return modelmat


def lattice2(n,w,depth,mat,matxyz,rect,init_t=None):
  """
  Assuming we know the tableplane, find the rotation and translation in
  the XZ plane.
  - init_angle, init_t:
      if None, the rotation (4-way 90 degree ambiguity) is defined arbitrarily
      and the translation is set to the centroid of the detected points.
  """
  from config import LW

  (l,t),(r,b) = rect
  assert mat.shape == (4,4)
  assert matxyz.shape == (4,4)
  assert mat.dtype == np.float32
  assert matxyz.dtype == np.float32

  global modelmat
  modelmat = np.array(mat)

  # Build a matrix that can project the depth image into model space
  v,u = np.mgrid[t:b,l:r]
  X,Y,Z = project(u.astype('f'),v.astype('f'),depth[t:b,l:r].astype('f'),
                  np.dot(modelmat, matxyz))

  # Project normals from camera space to model space (axis aligned)
  global dx,dy,dz
  global cx,cy,cz
  dx,dy,dz = project(*np.rollaxis(n,2), mat=modelmat)
  cx,cy,cz = color_axis(dx,dy,dz,w)

  # If we don't have a good initialization for the model space translation,
  # use the centroid of the surface points.
  if init_t:
    modelmat[:,3] -= [X[cx>0].mean(), 0, Z[cz>0].mean(), 0]
    v,u = np.mgrid[t:b,l:r]
    X,Y,Z = project(u.astype('f'),v.astype('f'),depth[t:b,l:r].astype('f'),
                    np.dot(modelmat, matxyz))

  global meanx, meany, meanz
  meanx = circular_mean(X[cx>0],LW)
  meanz = circular_mean(Z[cz>0],LW)

  ax,az = np.sum(cx>0),np.sum(cz>0)
  ax,az = [np.minimum(_/30.0,1.0) for _ in ax,az]
  modelmat[:,3] -= np.array([ax*meanx, 0, az*meanz, 0])

  X -= (ax)*meanx
  Z -= (az)*meanz

  # Stacked data in model space
  global XYZ, dXYZ, cXYZ
  XYZ = ((X,Y,Z))
  dXYZ = ((dx, dy, dz))
  cXYZ = ((cx, cy, cz))

  return modelmat
