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
import opennpy
import os

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from blockplayer.blockwindow import BlockWindow
global window
if not 'window' in globals():
    window = BlockWindow(title='demo_grid', size=(640,480))
    window.Move((0,0))
    pointmodels = []

from rtmodel.rangeimage import RangeImage
from blockplayer import config
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import stencil
from blockplayer import blockdraw
from blockplayer import dataset
from blockplayer import main
from blockplayer import colormap
from blockplayer import blockcraft
from blockplayer import blockmodel
import cv


def show_rotated():
    g = main.grid.occ
    #g = blockcraft.centered_rotated(main.R_correct, g)
    g = blockcraft.translated_rotated(main.R_correct, g)    
    marginal = g.sum(1).astype('u1')*255
    cv.NamedWindow('scale_test', 0)
    cv.ShowImage('scale_test', cv.fromarray(marginal))
    cv.ResizeWindow('scale_test', 300, 300)


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
        depth = dataset.depths[0]
        rgb = dataset.rgbs[0] if dataset.rgbs else None
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()
        rgb,_ = opennpy.sync_get_video()

    global rimgs, pointmodels
    rimgs = []
    pointmodels = []
    for i,cam in enumerate(config.cameras):
        rimg = RangeImage(dataset.depths[i], cam)
        rimg.threshold_and_mask(config.bg[i])
        rimg.filter(win=6)
        rimg.compute_normals()
        rimgs.append(rimg)
        #rimg.compute_points()
        #pointmodels.append(rimg.point_model())

    main.update_frame(depth, rgb)
    #print main.R_aligned

    blockdraw.clear()
    blockdraw.show_grid('o1', main.occvac.occ, color=np.array([1,1,0,1]))
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

    if 1:
        update_display()

    if 'R_correct' in main.__dict__:
        window.modelmat = main.R_display

    #show_rgb(rgb)
    window.lookat = np.array(config.center)
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

    if not FOR_REAL:
        if dset is None:
            dataset.load_random_dataset()
        else:
            dataset.load_dataset(dset)
        while dataset.frame_num < frame_num:
            dataset.advance()

        dataset.load_gt()
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

    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)-1
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]

    if 1:
        window.update_xyz(Xo, Yo, Zo, COLOR=(R,G,B,R*0+1))

    #show_rotated()
    window.Refresh()


@window.event
def post_draw():
    for c in config.cameras:
        c.render_frustum()
    for pm in pointmodels:
        pm.draw()

if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    pass
    #go()
