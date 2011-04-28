import os
import shutil
import numpy as np
import time
import simplejson as json
import glob
import cPickle as pickle

from blockplayer import dataset
from blockplayer import config
from blockplayer import grid

out_path = os.path.join('data/experiments','gterr')


def run_dataset(name):
    dataset.load_dataset(name)
    name = os.path.split(name)[1]

    folder = os.path.join(out_path, name)
    os.mkdir(folder)

    with open(os.path.join('data/experiments/output',
                           name,'output.pkl'),'r') as f:
        d = pickle.load(f)

    global output
    output = np.array([_[1] for _ in d['output']])

    errors = []
    for occ in output:
        A,B,err,_,_ = grid.xcorr_correction(occ, config.GT)
        errors.append((A,B,err))
    return errors


def run_grid():
    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)

    datasets = glob.glob('data/sets/*')

    global errors
    errors = []

    for name in datasets:
    #for name in ('data/sets/cube',):
        output = run_dataset(name)
        errors.append([_[2] for _ in output])

    with open(os.path.join(out_path,'gterr.pkl'),'w') as f:
        pickle.dump(dict(errors=errors,names=datasets), f)


def plot_errors():
    with open(os.path.join(out_path,'gterr.pkl'),'r') as f:
        d = pickle.load(f)

    errors = d['errors']

    import pylab
    pylab.figure(1)
    pylab.clf()
    for err in errors:
        pylab.plot(np.arange(len(err))/30., err)
    pylab.xlabel('Time (s)')
    pylab.ylabel('Relative error (ground truth)')
    pylab.title('Relative error convergence over time')


if __name__ == "__main__":
    print "This is suppose to be run with IPython, otherwise it won't do anything"
    #run_grid()
    pass
