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
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import spacecarve
from blockplayer import stencil
from blockplayer import colors
from blockplayer import occvac
from blockplayer import blockdraw
from blockplayer import dataset
from blockplayer import classify
from blockplayer import hashalign

import cv

from blockplayer import colormap


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

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

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

    # Classify with rfc
    #classmask = classify.predict(depth)
    #mask &= (classmask[0]==1)

    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()
    R_aligned = lattice.translation_opencl(R_oriented)

    # Use occvac to estimate the voxels from just the current frame
    occ, vac = occvac.carve_opencl()

    # Further carve out the voxels using spacecarve
    warn = np.seterr(invalid='ignore')
    try:
        vac = vac | spacecarve.carve(depth, R_aligned)
    except np.LinAlgError:
        return
    np.seterr(divide=warn['invalid'])

    if 1 and grid.has_previous_estimate() and np.any(grid.occ):
        if 0:
            R_aligned, c = grid.nearest(grid.previous_estimate['R_correct'],
                                        R_aligned)
            occ = occvac.occ = grid.apply_correction(occ, *c)
            vac = occvac.vac = grid.apply_correction(vac, *c)

            # Align the new voxels with the previous estimate
            R_correct, occ, vac = grid.align_with_previous(R_aligned, occ, vac)
        else:
            c,err = hashalign.find_best_alignment(grid.occ, grid.vac, occ, vac,
                                                  R_aligned,
                                                  grid.previous_estimate['R_correct'])
            R_correct = hashalign.correction2modelmat(R_aligned, *c)
            grid.R_correct = R_correct
            occ = occvac.occ = hashalign.apply_correction(occ, *c)
            vac = occvac.vac = hashalign.apply_correction(vac, *c)
        #print R_aligned, R_correct
    elif np.any(occ):
        # If this is the first estimate (bootstrap) then try to center the grid
        R_correct, occ, vac = grid.center(R_aligned, occ, vac)
    else:
        return

    occ_stencil, vac_stencil = grid.stencil_carve(depth, rect,
                                                  R_correct, occ, vac,
                                                  rgb)
    if lattice.is_valid_estimate():
        # Run stencil carve and merge
        color = stencil.RGB if not rgb is None else None
        grid.merge_with_previous(occ, vac, occ_stencil, vac_stencil, color)

    if 1:
        blockdraw.clear()
        if 'RGB' in stencil.__dict__:
            blockdraw.show_grid('occ', grid.occ, color=grid.color)
        else:
            blockdraw.show_grid('occ', grid.occ, color=np.array([1,0.6,0.6,1]))

        if 0 and lattice.is_valid_estimate():
            if 1:
                blockdraw.show_grid('occ_stencil', occ_stencil,
                                    color=np.array([1,1,0,1]))
            if 1:
                blockdraw.show_grid('vac_stencil', vac_stencil,
                                    color=np.array([1,0,1,1]))

            if 0:
                blockdraw.show_grid('o1', occvac.occ&occ_stencil,
                                    color=np.array([0,0,1,1]))
            if 0:
                blockdraw.show_grid('o2', occvac.vac,
                                    color=np.array([0,1,0,0.2]))
            if 1:
                blockdraw.show_grid('spacecarve',
                                    spacecarve.vac&grid.occ,
                                    color=np.array([0,1,0,1]))

        #blockdraw.show_grid('vac', grid.vac,
        #                    color=np.array([0.6,1,0.6,0]))
        if lattice.is_valid_estimate():
            window.clearcolor=[0.9,1,0.9,0]
        else:
            window.clearcolor=[1,1,1,0]
        update_display()
        pylab.waitforbuttonpress(0.01)
        sys.stdout.flush()
    grid.update_previous_estimate(R_correct)


def resume():
    try:
        while 1: once()
    except IOError:
        return


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

        import re
        number = int(re.match('.*_z(\d)m_.*', dset).groups()[0])
        with open('data/experiments/gt/gt%d.txt' % number) as f:
            GT = grid.gt2grid(f.read())
        grid.initialize_with_groundtruth(GT)

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
    window.update_xyz(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))
    if 'R_correct' in globals():
        window.modelmat = R_correct
    window.Refresh()


@window.event
def post_draw():
    pass

if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
