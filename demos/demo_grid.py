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
from blockplayer import flatrot
from blockplayer import grid
import cv

from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_grid', size=(640,480))


if not 'FOR_REAL' in globals():
    FOR_REAL = False


depth_cache = []


def once():
    if not FOR_REAL:
        dataset.advance()
        global depth
        depth = dataset.depth
    else:
        depth,_ = freenect.sync_get_depth()
        rgb,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)
        cv.NamedWindow('RGB',0)
        #cv.ShowImage('RGB',rgb)

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global depth_cache
    #depth_cache.append(np.array(depth))
    #depth_cache = depth_cache[-5:]
    #for d in depth_cache[:-1]:
    #    depth[d==2047]=2047

    global mask, rect
    global modelmat

    try:
        (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)
    except IndexError:
        grid.initialize()
        modelmat = None
        window.Refresh()
        pylab.waitforbuttonpress(0.01)
        return

    opencl.set_rect(rect)
    normals.normals_opencl(from_rect(depth,rect).astype('f'),
                           np.array(from_rect(mask,rect)), rect,
                           6)

    mat = np.eye(4,dtype='f')
    if modelmat is None:
        mat[:3,:3] = flatrot.flatrot_opencl()
        mat = lattice.lattice2_opencl(mat)
    else:
        mat = modelmat.copy()
        mat[:3,:3] = flatrot.flatrot_opencl(modelmat[:3,:])
        mat = lattice.lattice2_opencl(mat)

    #print 'flatrot.dm:', flatrot.dm, \
    #      'lat.dmx:', lattice.dmx, \
    #      'lat.dmy:', lattice.dmy

    if 1:
        global face, Xo, Yo, Zo
        _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
        Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

        global cx,cy,cz
        cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                               dtype='i1').reshape(-1,4),1)-1
        R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]
        update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))

    grid.add_votes(lattice.meanx, lattice.meanz, depth, rect, use_opencl=True)
    #grid.add_votes(lattice.meanx, lattice.meanz, depth, use_opencl=False)
    modelmat = lattice.modelmat

    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.01)
    import sys
    sys.stdout.flush()


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
        from blockplayer.grid import solid_blocks, shadow_blocks, wire_blocks
        if not 'modelmat' in lattice.__dict__: return
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
        glMultMatrixf(np.linalg.inv(lattice.modelmat).transpose())
        glScale(LW,LH,LW)

        #glEnable(GL_LINE_SMOOTH)
        #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        #for (x,y,z),cind in zip(legos,legocolors):

        glPushMatrix()
        glTranslate(*grid.bounds[0])

        # Draw the carved out pixels
        glColor(1.0,0.1,0.1,0.2)
        glEnableClientState(GL_VERTEX_ARRAY)

        if 0 and wire_blocks:
            carve_verts, _, line_inds, _ = wire_blocks
            glVertexPointeri(carve_verts)
            glDrawElementsui(GL_LINES, line_inds)

        if 0 and shadow_blocks:
            carve_verts, _, line_inds, quad_inds = shadow_blocks
            glVertexPointeri(carve_verts)
            #glDrawElementsui(GL_QUADS, quad_inds)
            glColor(1,1,0)
            glDrawElementsui(GL_LINES, line_inds)
            glDisableClientState(GL_VERTEX_ARRAY)

        #  Draw the filled in surface faces of the legos
        # verts, norms, line_inds, quad_inds =
            #grid_vertices((vote_grid>30)&(carve_grid<30))
        if 1 and solid_blocks:
            blocks = solid_blocks
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointeri(blocks['vertices'])
            # glColor(0.3,0.3,0.3)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointerf(np.abs(blocks['normals']))
            glDrawElementsui(GL_QUADS, blocks['quad_inds'])
            glDisableClientState(GL_COLOR_ARRAY)
            glColor(1,1,1,1)
            #glColor(0.1,0.1,0.1)
            glDrawElementsui(GL_LINES, blocks['line_inds'])
            glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

        # Draw the shadow blocks (occlusions)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Draw the outlines for the lego blocks
        glColor(1,1,1,0.8)

        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)

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
            GR = grid.GRIDRAD
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
