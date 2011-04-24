import numpy as np
import pylab
from OpenGL.GL import *
import freenect

from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import spacecarve
from blockplayer import stencil
from blockplayer import occvac
from blockplayer import blockdraw

import cv

from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_grid', size=(640,480))
    window.Move((0,0))


if not 'FOR_REAL' in globals():
    FOR_REAL = False


depth_cache = []


def once():
    global depth, rgb
    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depth
    else:
        depth,_ = freenect.sync_get_depth()
        rgb,_ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
        cv.NamedWindow('RGB',0)
        #cv.ShowImage('RGB',rgb)

    global mask, rect, modelmat

    try:
        (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)
    except IndexError:
        grid.initialize()
        modelmat = None
        window.Refresh()
        pylab.waitforbuttonpress(0.01)
        return

    # Compute the surface normals
    normals.normals_opencl(depth, mask, rect)

    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()
    R_aligned = lattice.translation_opencl(R_oriented)

    # Use occvac to estimate the voxels from just the current frame
    occ, vac = occvac.carve_opencl()

    if grid.has_previous_estimate():
        R_aligned, c = grid.nearest(grid.previous_estimate[2], R_aligned)
        print c
        occ = occvac.occ = grid.apply_correction(occ, *c)
        vac = occvac.vac = grid.apply_correction(vac, *c)

    # Further carve out the voxels using spacecarve
    vac = vac | spacecarve.carve(depth, R_aligned)

    if 1 and grid.has_previous_estimate():
        # Align the new voxels with the previous estimate
        R_correct, occ, vac = grid.align_with_previous(R_aligned, occ, vac)
    else:
        # Otherwise try to center it
        R_correct, occ, vac = grid.center(R_aligned, occ, vac)

    if lattice.is_valid_estimate():
        # Run stencil carve and merge
        occ_stencil, vac_stencil = grid.stencil_carve(depth, rect,
                                                      R_correct, occ, vac)
        grid.merge_with_previous(occ, vac, occ_stencil, vac_stencil)

    if 1:
        blockdraw.clear()
        blockdraw.show_grid('occ', grid.occ&~occvac.occ,
                            color=np.array([1,0.6,0.6,1]))
        blockdraw.show_grid('occ1', grid.occ&occvac.occ,
                            color=np.array([1,0,0,1]))

        if 1 and lattice.is_valid_estimate():
            if 1:
                blockdraw.show_grid('occvac', occ_stencil,
                                    color=np.array([1,1,0,1]))
            if 0:
                blockdraw.show_grid('o1', occvac.occ&~occ_stencil,
                                    color=np.array([0,0,1,1]))
            if 1:
                blockdraw.show_grid('o2', occvac.occ&vac_stencil,
                                    color=np.array([0,1,0,1]))

        #blockdraw.show_grid('vac', grid.vac,
        #                    color=np.array([0.6,1,0.6,0]))
        update_display()
        pylab.waitforbuttonpress(0.01)
        sys.stdout.flush()
    grid.previous_estimate = grid.occ, grid.vac, R_correct


def resume():
    while 1: once()


def start(dset=None, frame_num=0):
    global modelmat
    modelmat = None
    grid.initialize()
    if not FOR_REAL:
        if dset is None:
            dataset.load_random_dataset()
        else:
            dataset.load_dataset(dset)
        while dataset.frame_num < frame_num:
            dataset.advance()
    else:
        config.load('data/newest_calibration')
        dataset.setup_opencl()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()


def update_display():
    global face, Xo, Yo, Zo
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)-1
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]
    update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))


def update(X,Y,Z,COLOR=None,AXES=None):
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
    window.clearcolor = [1,1,1,0]
    window.Refresh()

    @window.event
    def post_draw():
        from blockplayer.config import LW,LH

        if not 'R_correct' in grid.__dict__: return
        modelmat = R_correct

        glPolygonOffset(1.0,0.2)
        glEnable(GL_POLYGON_OFFSET_FILL)

        # Draw the gray table
        if 1:
            glBegin(GL_QUADS)
            glColor(0.6,0.7,0.7,1)
            for x,y,z in config.bg['boundptsM']:
                glVertex(x,y,z)
            glEnd()

        glPushMatrix()
        glMultMatrixf(np.linalg.inv(modelmat).transpose())
        glScale(LW,LH,LW)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glPushMatrix()
        glTranslate(*config.bounds[0])
        blockdraw.draw()
        glPopMatrix()

        # Draw the shadow blocks (occlusions)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Draw the axes for the model coordinate space
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
        glEnd()

        # Draw a grid for the model coordinate space
        if 1:
            glLineWidth(1)
            glBegin(GL_LINES)
            GR = config.GRIDRAD
            glColor3f(0.2,0.2,0.4)
            for j in range(0,1):
                for i in range(-GR,GR+1):
                    glVertex(i,j,-GR); glVertex(i,j,GR)
                    glVertex(-GR,j,i); glVertex(GR,j,i)
            glEnd()
            glPopMatrix()
            pass

if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
