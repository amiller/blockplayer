import os
import shutil
import numpy as np
import time
import simplejson as json
import glob
import cPickle as pickle

from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid
from blockplayer import spacecarve
from blockplayer import stencil
from blockplayer import occvac

out_path = os.path.join('data/experiments','output')

from blockplayer import glxcontext
glxcontext.init()
import sys
sys.stdout.write("GL Version String: ")
glxcontext.printinfo()


R_correct = None


def once():
    depth = dataset.depth
    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    global R_correct
    # Compute the surface normals
    normals.normals_opencl(depth, mask, rect)

    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()
    R_aligned = lattice.translation_opencl(R_oriented)

    # Use occvac to estimate the voxels from just the current frame
    occ, vac = occvac.carve_opencl()

    if grid.has_previous_estimate():
        R_aligned, c = grid.nearest(grid.previous_estimate[2], R_aligned)
        print c
        occ = occvac.occ = grid.apply_correction(occ, *c)
        vac = occvac.vac = grid.apply_correction(vac, *c)

    # Further carve out the voxels using spacecarve
    vac = vac | spacecarve.carve(depth, R_aligned)

    if 1 and grid.has_previous_estimate():
        # Align the new voxels with the previous estimate
        R_correct, occ, vac = grid.align_with_previous(R_aligned, occ, vac)
    else:
        # Otherwise try to center it
        R_correct, occ, vac = grid.center(R_aligned, occ, vac)

    if lattice.is_valid_estimate():
        # Run stencil carve and merge
        occ_stencil, vac_stencil = grid.stencil_carve(depth, rect,
                                                      R_correct, occ, vac)
        grid.merge_with_previous(occ, vac, occ_stencil, vac_stencil)
    grid.previous_estimate = grid.occ, grid.vac, R_correct


def run_grid():

    datasets = glob.glob('data/sets/*')
    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)

    for name in datasets:
    #for name in ('data/sets/cube',):
        dataset.load_dataset(name)
        name = os.path.split(name)[1]

        d = dict(name=name)
        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        global modelmat
        modelmat = None
        grid.initialize()

        total = 0
        output = []
        while 1:
            try:
                dataset.advance()
            except (IOError, ValueError):
                break
            t1 = time.time()
            once()
            t2 = time.time()
            total += t2-t1

            output.append((R_correct.copy(), grid.occ.copy()))

        d['frames'] = dataset.frame_num
        d['time'] = total
        d['output'] = output
        with open(os.path.join(folder, 'output.pkl'),'w') as f:
            pickle.dump(d, f)

        with open(os.path.join(folder, 'final_output.txt'),'w') as f:
            f.write(grid.grid2gt(grid.occ))


if __name__ == "__main__":
    run_grid()
