# Andrew Miller <amiller@cs.ucf.edu> 2011
# 
#

import numpy as np
import pylab
from OpenGL.GL import *
import opennpy
import os

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from wxpy3d import PointWindow
from wxpy3d.opengl_state import opengl_state
global window
if not 'window' in globals():
    window = PointWindow(title='demo_blockmodel', size=(640,480))
    window.Move((0,0))

from rtmodel.rangeimage import RangeImage
from rtmodel.camera import kinect_camera, Camera
from blockplayer import config
from blockplayer import grid
from blockplayer import dataset
from blockplayer import colormap
from blockplayer import blockmodel
reload(blockmodel)


def load(dset=None):
    global bm, GT

    if dset is None:
        dataset.load_random_dataset()
    else:
        dataset.load_dataset(dset)

    name = dset
    name = os.path.split(name)[1]
    custom = os.path.join('data/sets/', name, 'gt.txt')
    try:
        if os.path.exists(custom):
            # Try dataset directory first
            fname = custom
        else:
            import re
                # Fall back on generic ground truth file
            match = re.match('.*_z(\d)m_(.*)', name)
            number = int(match.groups()[0])
            fname = 'data/experiments/gt/gt%d.txt' % number

        print 'Initializing with groundtruth'
        with open(fname) as f:
            GT = grid.gt2grid(f.read())

        bm = blockmodel.BlockModel(GT)
    except AttributeError: # re.match failed
        print 'No groundtruth'
        raise
    window.Refresh()

def draw_axes():
    with opengl_state():
        glEnable(GL_DEPTH_TEST)
        glScale(.2, .2, .2)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(3, 0xAAAA)
        glBegin(GL_LINES)
        glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
        glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
        glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
        glEnd()


def once():
    global cam, rimg, color, bm, normals
    cam = kinect_camera()
    cam.RT = window.RT
    cam.RT[:3,3] /= 2
    bm = blockmodel.BlockModel(GT)
    rimg, color, normals = bm.render(cam)
    figure(1); clf(); imshow((np.abs(color))*10)
    figure(2); clf(); imshow(rimg.depth)
    figure(3); clf(); imshow(np.abs(normals))
    window.Refresh()

@window.event
def post_draw():
    # Draw the model
    window.set_camera()
    if 'bm' in globals() and 1:
        bm.draw()
        draw_axes()
    if 'cam' in globals():
        cam.render_frustum() 

@window.eventx
def EVT_KEY_DOWN(evt):
    if evt.GetKeyCode() == ord(' '):
        once()
    window.Refresh()

if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    load('data/sets/6dof_cube_0')
