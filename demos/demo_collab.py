import numpy as np
import pylab
from OpenGL.GL import *
import opennpy

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from blockplayer.visuals.blockwindow import BlockWindow
global window
if not 'window' in globals():
    window = BlockWindow(title='demo_grid', size=(640,480))
    window.Move((0,0))

from blockplayer import config
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import stencil
from blockplayer import blockdraw
from blockplayer import dataset
from blockplayer import main
from blockplayer import colormap
from blockplayer import hashalign
import cv


def show_rgb(rgb):
    rgb = rgb[::2,::2,::-1]
    im = cv.CreateImage((rgb.shape[1],rgb.shape[0]), 8, 3)
    cv.SetData(im, ((rgb*3.).clip(0,255).astype('u1')).tostring())
    cv.NamedWindow('rgb', 0)
    cv.ShowImage('rgb', im)


def show_depth(name, depth):
    im = cv.CreateImage((depth.shape[1],depth.shape[0]), 8, 3)
    cv.SetData(im, colormap.color_map(depth))
    cv.ShowImage(name, im)


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
    try:
        c,_ = hashalign.find_best_alignment(grid.occ,0*grid.occ,
                                        target_model,~target_model)
    except ValueError:
        pass
    else:
        tm = hashalign.apply_correction(target_model, *c)
        tm = np.ascontiguousarray(tm)

        not_filled = tm & ~grid.occ
        correct = tm & grid.occ
        incorrect = ~tm & grid.occ

        try:
            next_layer = np.min(np.nonzero(not_filled)[1])
        except ValueError:
            blockdraw.show_grid('0', grid.occ, color=np.array([0.2,1,0.2,1]))
        else:
            blockdraw.show_grid('1', incorrect,
                                color=np.array([1,1,0.1,1]))
            nf = not_filled*0
            nf[:,next_layer,:] = 1
            nf = nf & not_filled
            blockdraw.show_grid('2', nf,
                                color=np.array([1,0.2,1.0,1]))
            blockdraw.show_grid('3', correct, color=np.array([0.1,0.3,0.1,1]))

    window.clearcolor=[0,0,0,0]
    window.flag_drawgrid = False

    if 'R_correct' in main.__dict__:
        window.modelmat = main.R_display

    #show_rgb(rgb)
    window.Refresh()
    pylab.waitforbuttonpress(0.005)
    sys.stdout.flush()


def resume():
    try:
        while 1: once()
    except IOError:
        return


def start(dset=None, frame_num=0):
    main.initialize()

    #with open('data/experiments/collab/2011.txt') as f:
    global target_model
    with open('data/experiments/collab/block.txt') as f:
        target_model = grid.gt2grid(f.read())
    #grid.initialize_with_groundtruth(GT)

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
    blockdraw.clear()
    start(dset, frame_num)
    resume()


@window.event
def post_draw():
    pass

if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    pass
    #go()
