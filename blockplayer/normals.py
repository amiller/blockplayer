import numpy as np
import expmap
import scipy
import scipy.ndimage
import scipy.optimize
import pylab
from OpenGL.GL import *
import calibkinect
import opencl

import os
import ctypes
print __file__
speedup = np.ctypeslib.load_library('speedup_ctypes.so',
                                    os.path.dirname(__file__))

matarg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
speedup.normals.argtypes = [matarg, matarg, matarg,
                            matarg, matarg, matarg,
                            matarg, ctypes.c_int, ctypes.c_int]


def normals_opencl(depth, mask=None, rect=((0,0),(640,480)), win=7):
    (l,t),(r,b) = rect
    assert depth.dtype == np.float32
    depth = depth[t:b,l:r]
    #depth[depth==2047] = -1e8

    if mask is None:
        mask = np.ones((b-t,r-l),'bool')

    global filt
    filt = scipy.ndimage.uniform_filter(depth,win)
    opencl.load_filt(filt)
    opencl.load_mask(mask)
    return opencl.compute_normals().wait()


def normal_show(nx,ny,nz):
    return np.dstack((nx/2+.5,ny/2+.5,nz/2+.5))


def normals_numpy(depth, rect=((0,0),(640,480)), win=7):
    assert depth.dtype == np.float32
    from scipy.ndimage.filters import uniform_filter
    (l,t),(r,b) = rect
    v,u = np.mgrid[t:b,l:r]
    depth = depth[v,u]
    depth[depth==2047] = -1e8
    depth = uniform_filter(depth, win)

    dx = (np.roll(depth,-1,1) - np.roll(depth,1,1))/2
    dy = (np.roll(depth,-1,0) - np.roll(depth,1,0))/2
    #dx,dy = np.array(depth),np.array(depth)
    #speedup.gradient(depth.ctypes.data, dx.ctypes.data, dy.ctypes.data, depth.shape[0], depth.shape[1])

    X,Y,Z,W = -dx, -dy, 0*dy+1, -(-dx*u + -dy*v + depth).astype(np.float32)

    mat = np.linalg.inv(calibkinect.xyz_matrix()).astype('f').transpose()
    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]
    w = np.sqrt(x*x + y*y + z*z)
    w[z<0] *= -1
    weights = z*0+1
    weights[depth<-1000] = 0
    weights[(z/w)<.1] = 0
    #return x/w, y/w, z/w
    return np.dstack((x/w,y/w,z/w)), weights


def normals_c(depth, rect=((0,0),(640,480)), win=7):
    assert depth.dtype == np.float32
    from scipy.ndimage.filters import uniform_filter, convolve
    (l,t),(r,b) = rect
    v,u = np.mgrid[t:b,l:r]
    depth = depth[v,u]
    depth[depth==2047] = -1e8
    depth = uniform_filter(depth, win)

    x,y,z = [np.empty_like(depth) for i in range(3)]
    mat = np.linalg.inv(calibkinect.xyz_matrix()).astype('f').transpose()

    speedup.normals(depth.astype('f'), u.astype('f'), v.astype('f'), x, y, z, mat, depth.shape[0], depth.shape[1])

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


"""
Everything below here needs to be moved to a demo_normalsphere.py
"""

import freenect
import pylab
def go_():
  global depth
  while 1:
    depth,_ = freenect.sync_get_depth()
    show_opencl()
    pylab.draw();
    pylab.waitforbuttonpress(0.1)


def project(depth, u=None, v=None, mat=calibkinect.xyz_matrix().astype('f')):
  if u is None or v is None: v,u = np.mgrid[:480,:640].astype('f')
  X,Y,Z = u,v,depth
  assert mat.dtype == np.float32
  assert mat.shape == (4,4)
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w


def update(X,Y,Z,COLOR=None,AXES=None):
  from visuals.pointwindow import PointWindow
  global window
  if not 'window' in globals(): window = PointWindow(size=(640,480))

  @window.event
  def on_draw_axes():

    # Draw some axes
    glLineWidth(3)
    glPushMatrix()
    glMultMatrixf(axes_rotation.transpose())

    glScalef(1.5,1.5,1.5)
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
    glEnd()
    glPopMatrix()

  xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
  mask = Z.flatten()<10
  xyz = xyz[mask,:]
  window.XYZ = xyz

  global axes_rotation
  axes_rotation = np.eye(4)
  if not AXES is None:
    # Rotate the axes
    axes_rotation[:3,:3] = expmap.axis2rot(-AXES)
  window.upvec = axes_rotation[:3,1]

  if not COLOR is None:
    R,G,B,A = COLOR
    color = np.vstack((R.flatten(), G.flatten(), B.flatten(), A.flatten())).transpose()
    color = color[mask,:]

  window.update_points(xyz)
  window.update_points(xyz, color)


def show_normals(depth, rect, win=7):
   global axes_rotation
   axes_rotation = np.eye(4)

   r0 = [-0.63, 0.68, 0.17]
   n,weights = normals_opencl(depth,rect,win=win)
   R,G,B = color_axis(n)
   update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+.5,G+.5,B+.5,weights*(R+G+B)))
   window.Refresh()


if __name__ == "__main__":  
  rgb, depth = [x[1].astype('f') for x in np.load('data/block2.npz').items()]
  rect =((264,231),(434,371))
  (l,t),(r,b) = rect
  v,u = np.mgrid[t:b,l:r]
  r0 = [-0.63, 0.68, 0.17]
  
  x,y,z = project(depth[v,u], u.astype(np.float32), v.astype(np.float32))
  
  show_normals(depth,rect)
