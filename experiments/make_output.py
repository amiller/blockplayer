import os
import shutil
import numpy as np
import time
import json
import glob
import cPickle as pickle
import glxcontext

from blockplayer import dataset
from blockplayer import grid
from blockplayer import main


out_path = os.path.join('data/experiments','output')


# Create an offscreen opengl context
glxcontext.makecurrent()
from OpenGL.GL import glGetString, GL_VERSION
print("GL Version String: ", glGetString(GL_VERSION))


def once():
    depth = dataset.depth
    rgb = dataset.rgb
    main.update_frame(depth, rgb)


def run_grid():
    datasets = glob.glob('data/sets/study_*')
    try:
        os.mkdir(out_path)
    except OSError:
        print "Couldn't make create output directory [%s], it may already exist." % out_path
        print "Remove it and try again."
        return False

    for name in datasets:
        dataset.load_dataset(name)
        name = os.path.split(name)[1]

        d = dict(name=name)
        folder = os.path.join(out_path, name)
        os.mkdir(folder)

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
