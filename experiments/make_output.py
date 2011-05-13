import os
import shutil
import numpy as np
import time
import simplejson as json
import glob
import cPickle as pickle

from blockplayer import dataset
from blockplayer import grid
from blockplayer import main

out_path = os.path.join('data/experiments','output')

from blockplayer import glxcontext
glxcontext.init()
import sys
sys.stdout.write("GL Version String: ")
glxcontext.printinfo()


def once():
    depth = dataset.depth
    rgb = dataset.rgb
    main.update_frame(depth, rgb)


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
        main.initialize()

        import re
        number = int(re.match('.*_z(\d)m_.*', name).groups()[0])
        with open('data/experiments/gt/gt%d.txt' % number) as f:
            GT = grid.gt2grid(f.read())
        grid.initialize_with_groundtruth(GT)

        total = 0
        output = []
        try:
            while 1:
                try:
                    dataset.advance()
                except (IOError, ValueError):
                    break
                if dataset.frame_num % 30 == 0:
                    print name, dataset.frame_num
                t1 = time.time()
                once()
                t2 = time.time()
                total += t2-t1

                output.append((main.R_correct.copy(), grid.occ.copy()))
        except Exception as e:
            print e

        d['frames'] = dataset.frame_num
        d['time'] = total
        d['output'] = output
        with open(os.path.join(folder, 'output.pkl'),'w') as f:
            pickle.dump(d, f)

        with open(os.path.join(folder, 'final_output.txt'),'w') as f:
            f.write(grid.grid2gt(grid.occ))


if __name__ == "__main__":
    run_grid()
