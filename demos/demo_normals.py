# Andrew Miller <amiller@cs.ucf.edu> 2011
#
# BlockPlayer - 3D model reconstruction using the Lattice-First algorithm
# See: 
#    "Interactive 3D Model Acquisition and Tracking of Building Block Structures"
#    Andrew Miller, Brandyn White, Emiko Charbonneau, Zach Kanzler, and Joseph J. LaViola Jr.
#    IEEE VR 2012, IEEE TVGC 2012
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import cv
import pylab
import numpy as np
import timeit
import opennpy
from OpenGL.GL import *

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from wxpy3d.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_normals', size=(640,480))
    window.Move((0,0))

from blockplayer import dataset
from blockplayer import config
from blockplayer import opencl

from rtmodel.rangeimage import RangeImage
from rtmodel.camera import kinect_camera

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
    window.update_xyz(n[:,:,0], n[:,:,1], n[:,:,2], (w*(R+.5),w*(G+.5),w*(B+.5),w*(R+G+B)))
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
    cam = 1

    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depths[cam]
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global n, w, mask, rect, modelmat

    
    camera = kinect_camera()
    camera.RT = config.cameras[0]['Ktable']
    rimg = RangeImage(depth, camera)
    rimg.threshold_and_mask(config.cameras[0])
    rimg.filter(win=6)
        
    if 1:
        rimg.compute_normals()
        n,w = rimg.normals, rimg.weights
        show_normals(n, w, 'normals_numpy')

    if 0:
        opencl.set_rect(rimg.rect)
        opencl.load_filt(rimg.depth_filtered)
        opencl.load_raw(rimg.depth_recip)
        opencl.load_mask(from_rect(rimg.mask, rimg.rect).astype('u1'))

        dt = timeit.timeit(lambda:
                               opencl.compute_normals().wait(),
                           number=1)

        #print dt
        nw = opencl.get_normals()
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
