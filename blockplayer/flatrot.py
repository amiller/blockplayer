import numpy as np
import opencl
import expmap
import pylab
from OpenGL.GL import *


def flatrot_opencl(plane, guessmat=None, noshow=None):
  """Find the orientation of the lattice using the (labeled) surface normals.
  Assumes we know the table plane. If a guess is provided, then we use it.
  Otherwise it is chosen arbitrarily for initialization.
  """
  if guessmat is None:
    # Pick a random vector in the plane
    v1 = plane[:3]
    #v_ = np.random.random(3)
    v_ = -np.array([0,0,1])
    v2 = np.cross(v1,v_); v2 = (v2 / np.sqrt(np.dot(v2,v2)))
    v0 = np.cross(v1,v2)
    mat = np.hstack((np.vstack((v0,v1,v2)),[[0],[0],[0]]))
  else:
    v0,v1,v2 = guessmat[:3,:3]
    mat = np.array(guessmat)
    mat[:,3] = [0,0,0]

  opencl.compute_flatrot(mat.astype('f'))
  sq = opencl.reduce_flatrot()

  qqx = sq[0] / sq[3]
  qqz = sq[2] / sq[3]
  angle = np.arctan2(qqz,qqx)/4
  q0 = np.cos(angle) * v0 + np.sin(angle) * v2
  q0 /= np.sqrt(np.dot(q0,q0))
  q2 = np.cross(q0,v1)

  # Build an output matrix out of the components
  mat = np.vstack((q0,v1,q2))

  if not noshow:
    axes = expmap.rot2axis(mat)

    nwL,nwR = opencl.get_normals()
    from main import LW
    mm = np.eye(4)
    mm[:3,:3] = mat
    opencl.compute_lattice2(mm[:3,:4].astype('f'), LW)
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    cx,cz = np.rollaxis(np.frombuffer(np.array(face).data,
                                      dtype='i2').reshape(-1,2),1)
    R,G,B = np.abs(cx),cx*0,np.abs(cz)

    nL,wL = nwL[:,:,:3],nwL[:,:,3]
    nR,wR = nwR[:,:,:3],nwR[:,:,3]

    # Reproject using the basis vectors for display
    #X,Y,Z = np.rollaxis(nL.reshape(-1,3),1)
    #w = wL.reshape(-1)
    #X,Y,Z = np.rollaxis(nR.reshape(-1,3),1)
    #w = wR.reshape(-1)

    X,Y,Z = np.rollaxis(np.vstack((nL.reshape(-1,3),nR.reshape(-1,3))),1)
    #w = np.vstack((wL.reshape(-1,1),wR.reshape(-1,1)))

    update(X,Y,Z, COLOR=(R,G,B,R+G+B+.5), AXES=axes)
    window.Refresh()
    pylab.waitforbuttonpress(0.001)

  return mat


def flatrot_numpy(normals,weights):

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
  #mat = np.vstack((v0,v1,v2))
  mat = np.vstack((q0,np.array([0,1,0]),q2))
  axes = expmap.rot2axis(mat)

  if 1:
    # Reproject using the basis vectors for display
    if 1:
      X = dx
      Y = dy
      Z = dz
    else:
      pass

    update(X,Y,Z, COLOR=(w+.7,w*0+.7,w*0+.7,w*0+.5), AXES=axes)
    window.Refresh()
    pylab.waitforbuttonpress(0.001)

  return axes


def update(X,Y,Z,COLOR=None,AXES=None):
  from visuals.pointwindow import PointWindow
  global window
  if not 'window' in globals():
      window = PointWindow(size=(640,480))

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
    color = np.vstack((R.flatten(), G.flatten(), B.flatten(),
                       A.flatten())).transpose()
    color = color[mask,:]

  window.update_points(xyz)
  window.update_points(xyz, color)
