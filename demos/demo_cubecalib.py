import numpy as np
from OpenGL.GL import *

from blockplayer import table_calibration
from blockplayer import expmap
from blockplayer import lattice
from blockplayer import config
from blockplayer import opencl
from blockplayer import flatrot


def cubecalib_opencl():
    """Run the cube calibration, but then look at the points
    after running normals2_opencl"""
    table_calibration.finish_cube_calib('L')
    cXYZL = np.array(lattice.cXYZ)
    table_calibration.finish_cube_calib('R')
    cXYZR = np.array(lattice.cXYZ)
    cXYZ = np.hstack((cXYZL.reshape(3,-1), cXYZR.reshape(3,-1)))

    opencl.setup_kernel((config.bgL['KK'],config.bgL['Ktable']),
                        (config.bgR['KK'],config.bgR['Ktable']))

    opencl.compute_normals('LEFT')
    opencl.compute_normals('RIGHT')

    global modelmat
    modelmat = np.eye(4,dtype='f')
    modelmat[:3,:3] = flatrot.flatrot_opencl()
    modelmat = lattice.lattice2_opencl(modelmat)

    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)
    cx,cy,cz = cXYZ

    cany = (cx>0)|(cz>0)
    R,G,B = cx[cany],cy[cany],cz[cany]
    update(Xo[cany],Yo[cany],Zo[cany],COLOR=(R,G,B,R+G+B))
    window.mode = 'ortho'
    window.clearcolor = [1,1,1,0]
    window.Refresh()


def cubecalib():
    table_calibration.finish_cube_calib('L')
    XYZL = np.array(lattice.XYZ)
    cXYZL = np.array(lattice.cXYZ)

    table_calibration.finish_cube_calib('R')
    XYZR = np.array(lattice.XYZ)
    cXYZR = np.array(lattice.cXYZ)

    XYZ = np.hstack((XYZL.reshape(3,-1), XYZR.reshape(3,-1)))
    cXYZ = np.hstack((cXYZL.reshape(3,-1), cXYZR.reshape(3,-1)))

    Xo,Yo,Zo = XYZ
    cx,cy,cz = cXYZ

    cany = (cx>0)|(cz>0)
    R,G,B = cx[cany],cy[cany],cz[cany]
    update(Xo[cany],Yo[cany],Zo[cany],COLOR=(R,G,B,R+G+B))
    window.mode = 'ortho'
    window.clearcolor = [1,1,1,0]
    window.Refresh()


def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
    from blockplayer.visuals.pointwindow import PointWindow
    global window
    if not 'window' in globals():
        window = PointWindow(title='lattice2_opencl', size=(640,480))

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
        global modelmat
        if not 'modelmat' in globals():
            modelmat = np.eye(4)

        from blockplayer.config import LW,LH
        #return
        glPolygonOffset(1.0,0.2)
        glEnable(GL_POLYGON_OFFSET_FILL)

        # Draw the gray table
        if 1:
            glBegin(GL_QUADS)
            glColor(0.6,0.7,0.7,1)
            for x,y,z in config.bgL['boundptsM']:
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
