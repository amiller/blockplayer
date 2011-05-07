import cv
import pylab
import numpy as np
import timeit
import opennpy
from OpenGL.GL import *

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_normals', size=(640,480))
    window.Move((0,0))

from blockplayer import dataset
from blockplayer import normals
from blockplayer import config
from blockplayer import preprocess


def color_axis(normals,d=0.1):
    #n = np.log(np.power(normals,40))
    X,Y,Z = [normals[:,:,i] for i in range(3)]
    cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
    cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
    return [c * 0.8 + 0.2 for c in cx]


def show_normals(n, w, name='normals'):
    im = cv.CreateImage((n.shape[1],n.shape[0]), 32, 3)
    cv.SetData(im, np.dstack(3*[w]) * n[:,:,::-1]/2+.5)
    cv.ShowImage(name, im)


def show_normals_sphere(n, w):
    global axes_rotation
    axes_rotation = np.eye(4)
    window.upvec = axes_rotation[:3,1]

    R,G,B = color_axis(n)
    window.update_xyz(n[:,:,0], n[:,:,1], n[:,:,2], (R+.5,G+.5,B+.5,w*(R+G+B)))
    window.Refresh()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()


def resume():
    while 1: once()


def start(dset=None, frame_num=0):
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


def once():
    global depth
    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depth
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global mask, rect, modelmat
    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    global n,w
    if 0:
        n,w = normals.normals_numpy(depth)
        show_normals(n, w, 'normals_numpy')

    if 0:
        n,w = normals.normals_c(depth)
        show_normals(n, w, 'normals_c')

    if 1:
        normals.opencl.set_rect(rect)
        dt = timeit.timeit(lambda:
                           normals.normals_opencl(depth, mask, rect).wait(),
                           number=1)

        #print dt
        nw = normals.opencl.get_normals()
        n,w = nw[:,:,:3], nw[:,:,3]
        #show_normals(n, w, 'normals_opencl')
        show_normals_sphere(n, w)

    pylab.waitforbuttonpress(0.01)


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
