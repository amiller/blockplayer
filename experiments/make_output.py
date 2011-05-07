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
from blockplayer import classify
from blockplayer import hashalign

out_path = os.path.join('data/experiments','output')

from blockplayer import glxcontext
glxcontext.init()
import sys
sys.stdout.write("GL Version String: ")
glxcontext.printinfo()


R_correct = None


def once():
    depth = dataset.depth
    rgb = dataset.rgb
    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    global R_correct

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
    elif np.any(occ):
        # Otherwise try to center it
        R_correct, occ, vac = grid.center(R_aligned, occ, vac)
    else:
        return

    if lattice.is_valid_estimate():
        # Run stencil carve and merge
        occ_stencil, vac_stencil = grid.stencil_carve(depth, rect,
                                                      R_correct, occ, vac, rgb)
        color = stencil.RGB if not rgb is None else None
        grid.merge_with_previous(occ, vac, occ_stencil, vac_stencil, color)
    grid.update_previous_estimate(R_correct)        


def run_grid():

    datasets = glob.glob('data/sets/study_*')
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

        import re
        number = int(re.match('.*_z(\d)m_.*', name).groups()[0])
        with open('data/experiments/gt/gt%d.txt' % number) as f:
            GT = grid.gt2grid(f.read())
        grid.initialize_with_groundtruth(GT)
        

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
