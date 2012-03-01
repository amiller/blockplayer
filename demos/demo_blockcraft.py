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

import numpy as np
import pylab
from OpenGL.GL import *
import zmq

from blockplayer import config
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import stencil
from blockplayer import blockdraw
from blockplayer import dataset
from blockplayer import main
from blockplayer import blockcraft
import opennpy

from blockplayer.visuals.blockwindow import BlockWindow
global window
if not 'window' in globals():
    window = BlockWindow(title='demo_grid', size=(640,480))
    window.Move((0,0))

if not 'FOR_REAL' in globals():
    FOR_REAL = False


def once():
    global depth, rgb
    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depth
        rgb = dataset.rgb
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()
        rgb,_ = opennpy.sync_get_video()

    main.update_frame(depth, rgb)

    blockdraw.clear()
    if 'RGB' in stencil.__dict__:
        blockdraw.show_grid('occ', grid.occ, color=grid.color)
    else:
        blockdraw.show_grid('occ', grid.occ, color=np.array([1,0.6,0.6,1]))

    window.clearcolor=[0,0,0,0]
    window.flag_drawgrid = True

    if 'R_correct' in main.__dict__:
        window.modelmat = main.R_display
    
    g = blockcraft.translated_rotated(main.R_correct, grid.occ)
    talk_to_minecraft(g)
    
    window.Refresh()
    pylab.waitforbuttonpress(0.005)
    import sys
    sys.stdout.flush()


def resume():
    try:
        while 1: once()
    except IOError:
        return


def start(dset=None, frame_num=0):
    main.initialize()
    if not FOR_REAL:
        if dset is None:
            dataset.load_random_dataset()
        else:
            dataset.load_dataset(dset)
        while dataset.frame_num < frame_num:
            dataset.advance()
    else:
        config.load('data/newest_calibration')
        opennpy.align_depth_to_rgb()
        dataset.setup_opencl()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()

@window.event
def post_draw():
    pass

if 'window' in globals():
    window.Refresh()


if 'socket' in globals():
    socket.close()

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect('tcp://*:8134')


def talk_to_minecraft(voxels=None):
    global context,socket
    if voxels is None:
        voxels = grid.occ
    socket.send(voxels)

if __name__ == '__main__':
    pass
