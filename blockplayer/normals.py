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
speedup = np.ctypeslib.load_library('speedup_ctypes.so', os.path.dirname(__file__))

matarg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
speedup.normals.argtypes = [matarg,  matarg, matarg,  matarg, matarg, matarg, matarg, ctypes.c_int, ctypes.c_int]


def normals_opencl2(depthL, maskL, rectL,
                    depthR, maskR, rectR, win=7):
  assert depthL.dtype == np.float32
  assert depthR.dtype == np.float32

  global filtL, filtR

  opencl.load_mask(maskL,'LEFT')
  opencl.load_mask(maskR,'RIGHT')

  filtL = scipy.ndimage.uniform_filter(depthL,win)
  opencl.load_filt(filtL,'LEFT')                    
  opencl.compute_normals('LEFT')  

  filtR = scipy.ndimage.uniform_filter(depthR,win)  
  opencl.load_filt(filtR,'RIGHT')                    
  opencl.compute_normals('RIGHT').wait()

  
def normals_opencl(depth, mask, rect=((0,0),(640,480)), win=7, offset=0):
  (l,t),(r,b) = rect
  assert depth.dtype == np.float32
  depth = depth[t:b,l:r]
  #depth[depth==2047] = -1e8
  global filt
  filt = scipy.ndimage.uniform_filter(depth,win)  # 2ms?
  opencl.load_filt(filt,rect)                     # 329us
  opencl.load_mask(mask,rect)
  opencl.compute_normals(rect)                # 1.51ms
  #n = opencl.get_normals(rect=rect)          # 660us
  #return n
 
  
def normal_show(nx,ny,nz):
  return np.dstack((nx/2+.5,ny/2+.5,nz/2+.5))


def normals_numpy(depth, rect=((0,0),(640,480)), win=7):
  assert depth.dtype == np.float32
  from scipy.ndimage.filters import uniform_filter, convolve
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
  

# Color score
def color_axis(normals,d=0.1):
  #n = np.log(np.power(normals,40))
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
  return [c * 0.8 + 0.2 for c in cx]
  
def meanshift_iter_opencl(mat, rect=((0,0),(640,480))):
  opencl.compute_meanshift(mat, rect)
  ax,aw = opencl.get_meanshift(rect=rect)
  xm,ym,zm = aw[:,:,0].sum(), aw[:,:,1].sum(), aw[:,:,2].sum()
  dX = ax[:,:,0].sum() / (ym + zm)
  dY = ax[:,:,1].sum() / (zm + xm)
  dZ = ax[:,:,2].sum() / (xm + ym)
  if np.isnan(dX): dX = 0
  if np.isnan(dY): dY = 0
  if np.isnan(dZ): dZ = 0
  return (dX,dY,dZ)
  
def meanshift_iter_numpy(normals, mat, d, weights):
  n = apply_rot(mat, normals)
  X,Y,Z = [n[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  xm,ym,zm = [(c <= d*d)*weights for c in cc] # threshold mask
  dX = (Z*ym - Y*zm).sum() / (ym.sum() + zm.sum()); 
  dY = (X*zm - Z*xm).sum() / (zm.sum() + xm.sum()); 
  dZ = (Y*xm - X*ym).sum() / (xm.sum() + ym.sum()); 
  if np.isnan(dX): dX = 0
  if np.isnan(dY): dY = 0
  if np.isnan(dZ): dZ = 0
  return (dX,dY,dZ)
  
# Returns the optimal rotation, and the per-axis error weights
def mean_shift_optimize(normals, weights, r0=np.array([0,0,0]), rect=((0,0),(640,480))):
  (L,T),(R,B) = rect
  assert normals.shape == (B-T,R-L,3)
  assert weights.shape == (B-T,R-L)
  # Don't worry about convergence for now!
  mat = expmap.axis2rot(r0)
  d = 0.2 # E(p) = (x/d)^2 + (y/d)^2
  perr = 0; derr = 10
  iters = 0
  def compute_cc(mat,normals):
    n = apply_rot(mat, normals)
    X,Y,Z = [n[:,:,i] for i in range(3)]
    cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
    return cc
    
  while iters <= 40 and (derr > 0.0001):
    iters += 1
    
    if 0: # Compute the current error. Compare it to the previous
      cc = compute_cc(mat,normals)
      err = np.sum(weights*[np.max((1.0-(c/d*c/d),0*c),0) for c in cc])
      derr = np.abs(perr-err)
      perr = err
    
    if 1 or iters==4: # Compute and display the current rotated normals
      n = apply_rot(mat, normals)
      R,G,B = color_axis(n,d)
      #update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+0.5,G+0.5,B+0.5,weights))
      update(normals[:,:,0],normals[:,:,1],normals[:,:,2], 
           COLOR=(R+0.5,G+0.5,B+0.5,weights), AXES=expmap.rot2axis(mat))
      window.Refresh()
      pylab.waitforbuttonpress(0.01)
      
    dX,dY,dZ = meanshift_iter_numpy(normals, mat, d, weights)
    #dX,dY,dZ = meanshift_iter_opencl(mat, rect)
    m = expmap.euler2rot([dX, dY, dZ])
    mat = np.dot(m.transpose(), mat)
  cc = compute_cc(mat,normals)
  return expmap.rot2axis(mat), 1#weights*[np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
    
  
def score(normals, weights):
  d = 0.1
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  return np.sum([np.max((1.0-(c/d*c/d),0*c),0) for c in cc])
  
def apply_rot(rot, xyz):
  flat = np.rollaxis(xyz,2).reshape((3,-1))
  xp = np.dot(rot, flat)
  return xp.transpose().reshape(xyz.shape)


def surface(normals, weights, r0=np.array([-0.7,-0.2,0])):
  import mpl_toolkits.mplot3d.axes3d as mp3
  rangex = np.arange(-0.6,0.6,0.06)
  rangey = np.arange(-0.6,0.6,0.06)
  
  def err(x):
    rot = expmap.axis2rot(x)
    n = apply_rot(rot, normals)
    R,G,B = color_axis(n)
    update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+0.4,G+0.4,B+0.4,(R+G+B)))
    #window.Refresh()
    #pylab.waitforbuttonpress(0.001)
    window.processPaintEvent()
    return score(n, weights)
  x,y = np.meshgrid(rangex, rangey)
  z = [err(r0+np.array([xp,yp,0])) for xp in rangex for yp in rangey]
  z = np.reshape(z, (len(rangey),len(rangex)))
  fig = pylab.figure(1)
  fig.clf()
  global ax
  ax = mp3.Axes3D(fig)

  ax.plot_surface(x,y,-z, rstride=1,cstride=1, cmap=pylab.cm.jet,
  	linewidth=0,antialiased=False)
  fig.show()

def go(iter=1000):
  normals_opencl(depth,rect)
  for i in range(iter):
    x0 = (np.random.rand(3)*2-1)*0.3 + np.array(r0)
    x,w = mean_shift_optimize(n, weights, x0, rect)
    print x
    #print optimize_normals(n, weights, x0)

  
def play_movie():
  from pylab import gca, imshow, figure
  import pylab
  foldername = 'data/movies/block2'
  frame = 0
  r0 = np.array([-0.7626858,   0.28330218,  0.17082515])
  while 1:
    
    depth = np.load('%s/depth_%05d.npz'%(foldername,frame)).items()[0][1]
    #v,u = np.mgrid[135:332,335:485]
    v,u = np.mgrid[231:371,264:434]
    d = np.load('%s/normals_%05d.npz'%(foldername,frame))
    n,weights,u,v = d['n'],d['weights'],d['u'],d['v']
    figure(1)
    gca().clear()
    imshow(depth)
    x,y,z = project(depth[v,u], u, v)
    #n, weights = normal_compute(x,y,z)
    #np.savez('%s/normals_%05d'%(foldername,frame),v=v,u=u,n=n,weights=weights)
    r0 = mean_shift_optimize(n, weights, r0)
    print r0 
    frame += 1

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
