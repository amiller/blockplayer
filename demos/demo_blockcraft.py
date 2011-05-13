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

    #blockdraw.show_grid('vac', grid.vac,
    #                    color=np.array([0.6,1,0.6,0]))
    if 0 and lattice.is_valid_estimate():
        window.clearcolor=[0.9,1,0.9,0]
    else:
        window.clearcolor=[0,0,0,0]
        #window.clearcolor=[1,1,1,0]
        window.flag_drawgrid = True

    if 0:
        update_display()

    if 'R_correct' in main.__dict__:
        window.modelmat = main.R_display

    #show_rgb(rgb)
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


def update_display():
    global face, Xo, Yo, Zo

    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)-1
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]

    window.Refresh()


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


def talk_to_minecraft():
    global context,socket
    voxels = grid.occ
    socket.send(voxels)

if __name__ == '__main__':
    pass
