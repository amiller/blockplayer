import numpy as np
import pylab
from OpenGL.GL import *
import freenect

import blockplayer.visuals.pointwindow
from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import flatrot


if not 'FOR_REAL' in globals():
    FOR_REAL = False


from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_lattice', size=(640,480))


def once():
    if not FOR_REAL:
        dataset.advance()
        global depth
        depth = dataset.depth
    else:
        depth,_ = freenect.sync_get_depth()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global mask, rect

    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    opencl.set_rect(rect)
    normals.normals_opencl(from_rect(depth,rect).astype('f'),
                           np.array(from_rect(mask,rect)), rect,
                           6)

    mat = np.eye(4,dtype='f')
    mat[:3,:3] = flatrot.flatrot_opencl()

    mat = lattice.lattice2_opencl(mat)
    #print 'flatrot.dm:', flatrot.dm,\
    #      'lat.dmx:', lattice.dmx, lattice.countx,\
    #      'lat.dmy:', lattice.dmy, lattice.county

    global face, Xo, Yo, Zo
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]

    update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))

    global modelmat
    modelmat = mat

    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.005)


def resume():
    while 1: once()


def go(forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    if not FOR_REAL:
        dataset.load_random_dataset()
    resume()


def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
    global modelmat
    if not 'modelmat' in globals():
        return

    xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
    mask = Z.flatten()<10
    xyz = xyz[mask,:]

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

    window.update_points(xyz, color)

    @window.event
    def post_draw():
        from blockplayer.config import LW,LH
        #return
        glPolygonOffset(1.0,0.2)
        glEnable(GL_POLYGON_OFFSET_FILL)

        # Draw the gray table
        if 1:
            glBegin(GL_QUADS)
            glColor(0.6,0.7,0.7,1)
            for x,y,z in config.bg['boundptsM']:
                glVertex(x,y,z)
            glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)

        glPushMatrix()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

        if 1:
            # Draw the axes for the model coordinate space
            glLineWidth(3)
            glMultMatrixf(np.linalg.inv(modelmat).transpose())
            glScalef(LW,LH,LW)
            glBegin(GL_LINES)
            glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
            glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
            glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
            glEnd()

        #from blockplayer import grid
        # Draw a grid for the model coordinate space
        glLineWidth(1)
        glBegin(GL_LINES)
        #GR = grid.GRIDRAD*2
        GR = 30
        glColor3f(0.2,0.2,0.4)
        for j in range(0,1):
            for i in range(-GR,GR+1):
                glVertex(i,j,-GR); glVertex(i,j,GR)
                glVertex(-GR,j,i); glVertex(GR,j,i)
        glEnd()
        glPopMatrix()


if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
