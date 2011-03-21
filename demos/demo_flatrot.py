import numpy as np
import pylab
from OpenGL.GL import *
import cv

from blockplayer import expmap
from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import flatrot
from blockplayer import lattice


def ipbreak():
    import IPython.Shell
    if IPython.Shell.KBINT:
        IPython.Shell.KBINT = False
        raise SystemExit


def show_flatrot():
    pass


def show_normals(n, w, name='normals'):
    im = cv.CreateImage((n.shape[1],n.shape[0]), 32, 3)
    cv.SetData(im, np.dstack(3*[w]) * n[:,:,::-1]/2+.5)
    cv.ShowImage(name, im)


def once():
    dataset.advance()
    global depthL,depthR
    depthL,depthR = dataset.depthL,dataset.depthR

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global maskL, rectL
    global maskR, rectR

    (maskL,rectL) = preprocess.threshold_and_mask(depthL,config.bgL)
    (maskR,rectR) = preprocess.threshold_and_mask(depthR,config.bgR)

    opencl.set_rect(rectL,rectR)
    normals.normals_opencl2(from_rect(depthL,rectL).astype('f'), 
                            np.array(from_rect(maskL,rectL)), rectL, 
                            from_rect(depthR,rectR).astype('f'),
                            np.array(from_rect(maskR,rectR)), rectR,
                            6)

    mat = np.eye(4)
    mat[:3,:3] = flatrot.flatrot_opencl()
    axes = expmap.rot2axis(mat[:3,:3])

    # Get the normals in 'table space'
    nwL,nwR = opencl.get_normals()
    nL,wL = nwL[:,:,:3],nwL[:,:,3]
    nR,wR = nwR[:,:,:3],nwR[:,:,3]

    show_normals(nL, wL, 'normalsL')
    show_normals(nR, wR, 'normalsR')

    X,Y,Z = np.rollaxis(np.vstack((nL.reshape(-1,3),nR.reshape(-1,3))),1)
    w = np.vstack((wL.reshape(-1,1),wR.reshape(-1,1)))

    # Perform the 'labeling' by rotating the normals using the output from
    # flatrot, then threshold with color_axis
    global dx,dy,dz
    global cx,cy,cz
    dx,dy,dz = lattice.project(X,Y,Z, mat)
    cx,cy,cz = lattice.color_axis(dx,dy,dz,w.flatten())

    R,G,B = np.abs(cx).astype('f'),cy.astype('f')*0,np.abs(cz).astype('f')

    # Render the points in 'table space' but colored with the axes from flatrot
    update(X,Y,Z, COLOR=(R,G,B,R+G+B+.5), AXES=axes)
    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.005)
    ipbreak()


    if 0:  # I think this is from lattice
        _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
        Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)
        
        cx,cz = np.rollaxis(np.frombuffer(np.array(face).data,
                                          dtype='i2').reshape(-1,2),1)
        R,G,B = np.abs(cx),cx*0,np.abs(cz)
        update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))
    
        modelmat = lattice.lattice2_opencl(mat)
        window.Refresh()


def resume():
    while 1: once()


def go():
    dataset.load_random_dataset()
    resume()


def blah():
  # Stacked data in model space
  global XYZ, dXYZ, cXYZ
  XYZ = ((X,Y,Z))
  dXYZ = ((dx, dy, dz))
  cXYZ = ((cx, cy, cz))

  if 1:
    Xo,Yo,Zo = project(u,v,depth[t:b,l:r], matxyz)
    cany = (cx>0)|(cz>0)  
    R,G,B = cx[cany],cy[cany],cz[cany]
    update(Xo[cany],Yo[cany],Zo[cany],COLOR=(R,G,B,R+G+B))

    window.Refresh()
    
  return modelmat


def update(X,Y,Z,COLOR=None,AXES=None):
  from blockplayer.visuals.pointwindow import PointWindow
  global window
  if not 'window' in globals():
      window = PointWindow(title='flatrot_opencl', size=(640,480))

  @window.event
  def post_draw():
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

  #window.update_points(xyz)
  window.update_points(xyz, color)

if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    pass
    #go()
