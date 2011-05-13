import glob
import os
import numpy as np
import cPickle as pickle
import pylab

from blockplayer import dataset
from blockplayer import config
from blockplayer import hashalign
from blockplayer import grid


def run_error():

    datasets = glob.glob('data/sets/study*')
    d = {}
    for name in datasets:
    #for name in ('data/sets/cube',):
        dataset.load_dataset(name)
        name = os.path.split(name)[1]
        print name

        import re
        number, kind = re.match('.*_z(\d)m_(.*)$', name).groups()
        number = int(number)

        with open('data/experiments/gt/gt%d.txt' % number) as f:
            config.GT = grid.gt2grid(f.read())

        with open(os.path.join('data/experiments/output/',name,
                               'final_output.txt'),'r') as f:
            final_output = f.read()

        out = grid.gt2grid(final_output)
        gt = config.GT

        try:
            c,_ = hashalign.find_best_alignment(out, 0*out, gt, 0*gt)
        except ValueError:
            err = gt.sum()
        else:
            cgt = hashalign.apply_correction(gt, *c)
            err = np.sum(cgt != out)

        import re
        re.match('dfsdf', name)

        #_,_,err1,_,_ = grid.xcorr_correction(out, config.GT)
        #assert err == err1/float(gt.sum())
        d[name] = {'err': err,
                   'tot': gt.sum(),
                   'rel': err/float(gt.sum()),
                   'num': number,
                   'type': kind,}
    with open('data/experiments/error_output.pkl','w') as f:
        pickle.dump(d, f)




def bar_chart(B, xitems, gitems):
    """
    B should be (len(gitems))x(len(xitems)) array
    """
    width = 0.35


    for i in range(len(B)):
        xind = np.arange(len(B[i]))
        pylab.bar(xind+width, B[i])


def chart_error():
    global d
    with open('data/experiments/error_output.pkl') as f:
        d = pickle.load(f)

    types = ('none', 'add', 'remove')
    nums = np.arange(1,6)

    rel = [[np.mean([_['rel']
                     for _ in d.values() if _['type']==t and _['num']==n])
            for t in types]
           for n in nums]

    width = 0.35

    pylab.figure(1)
    pylab.clf()

    
    for i in range(len(types)):
        pylab.bar(i*width, rel ,
        
    

    p = {}
    for n in nums:
         p[n-1] = pylab.bar(range(3), rel[n-1])

    pylab.xlabel('shape')
    pylab.ylabel('relative error')
    pylab.draw()
    pylab.savefig('www/experiment.png')


if __name__ == '__main__':
    #run_error()
    chart_error()
