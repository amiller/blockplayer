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
    window = BlockWindow(title='demo_lattice', size=(640,480))
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

def once(use_opencl=True,hold=False):
    if not FOR_REAL:
        if not hold:
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
        rimg.compute_normals()
        rimgs.append(rimg)
        #rimg.compute_points()
        #pointmodels.append(rimg.point_model())

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    # Step 1. Preprocess the input images (smoothing, normals, bgsub)
    global modelmat, rimg
    bg = config.bg[0]
    cam = config.cameras[0]
    rimg = RangeImage(depth, cam)
    try:
        rimg.threshold_and_mask(bg)
    except IndexError:
        grid.initialize()
        modelmat = None
        return
    rimg.filter()

    if use_opencl:
        opencl.set_rect(rimg.rect)
        opencl.load_filt(rimg.depth_filtered)
        opencl.load_raw(rimg.depth_recip)
        opencl.load_mask(from_rect(rimg.mask, rimg.rect).astype('u1'))
        opencl.compute_normals().wait()
    else:
        rimg.compute_normals()

    # Step 2. Find the lattice orientation (modulo 90 degree rotation)
    global R_oriented, R_aligned, R_correct
    if use_opencl:
        R_oriented = lattice.orientation_opencl()
    else:
        R_oriented = lattice.orientation_numpy(rimg.normals, rimg.weights)
    assert R_oriented.shape == (4,4)

    # Apply a correction towards config.center
    LW = config.LW
    LH = config.LH
    modelmat = R_oriented
    modelmat = np.linalg.inv(modelmat)
    px,py,pz = config.center
    modelmat[:3,3] += [np.round(px/LW)*LW, np.round(py/LH)*LH, np.round(pz/LW)*LW]
    modelmat = np.linalg.inv(modelmat).astype('f')
    R_oriented = modelmat

    # Step 3. Find the lattice translation (modulo (LW,LH,LW))
    if use_opencl:
        R_aligned = lattice.translation_opencl(R_oriented)
    else:
        rimg.RT = R_oriented
        rimg.compute_points()
        rimg.compute_normals()
        R_aligned = lattice.translation_numpy(rimg, R_oriented)

    global modelmat
    if modelmat is None:
        modelmat = R_aligned.copy()
    else:
        modelmat,_ = grid.nearest(modelmat, R_aligned)

    global face, Xo, Yo, Zo
    global cx,cy,cz
    if use_opencl:
        _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
        Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)
        cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                               dtype='i1').reshape(-1,4),1)-1

    else:
        cx,cy,cz = lattice.cXYZ
        Xo,Yo,Zo = lattice.XYZ

    assert cx.shape == Xo.shape
    assert cy.shape == Yo.shape
    assert cz.shape == Zo.shape

    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]
    window.update_xyz(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))

    #show_rotated()
    window.flag_drawgrid = True
    window.modelmat = modelmat
    window.lookat = np.array(config.center)
    window.Refresh()
    pylab.waitforbuttonpress(0.005)


def resume():
    try:
        a = True
        while 1: 
            a = not a
            once(False)
            
    except IOError:
        return


def start(dset=None, frame_num=0):
    global modelmat
    modelmat = None

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
    for c in config.cameras:
        c.render_frustum()
    for pm in pointmodels:
        pm.draw()


if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
