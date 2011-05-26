import glob
import os
import numpy as np
import cPickle as pickle
import pylab
import re

from blockplayer import dataset
from blockplayer import config
from blockplayer import hashalign
from blockplayer import grid


def run_error():

    datasets = glob.glob('data/sets/study*')
    datasets.sort()
    d = {}
    for name in datasets:
        if 'user1' in name: continue
        dataset.load_dataset(name)
        name = os.path.split(name)[1]
        print name

        # Open the ground truth file
        custom = os.path.join('data/sets/', name, 'gt.txt')
        if os.path.exists(custom):
            # Try dataset directory first
            fname = custom
        else:
            import re
            # Fall back on generic ground truth file
            match = re.match('.*_z(\d)m_(.*)', name)
            if match is None:
                continue
            
            number = int(match.groups()[0])
            fname = 'data/experiments/gt/gt%d.txt' % number
        
        # Load the ground truth file
        removed = None
        added = None
        with open(fname) as fp:
            s = fp.read()
            config.GT = grid.gt2grid(s, chars='*')
            removed = grid.gt2grid(s, chars='rR')
            added = grid.gt2grid(s, chars='aA')

        import re
        number, kind = re.match('.*_z(\d)m_(.*)$', name).groups()
        number = int(number)

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
            gt = hashalign.apply_correction(gt, *c)
            added = hashalign.apply_correction(added, *c)
            removed = hashalign.apply_correction(removed, *c)
            should_be_there = gt | added 
            shouldnot_be_there = (~gt & ~added) | removed
            # Sanity check
            assert np.all(shouldnot_be_there ^ should_be_there)

            red = ~out & should_be_there
            yellow = out & shouldnot_be_there # Incorrect add
            purple = ~out & removed # Correct remove
            blue = out & added # Correct add
            green = out & gt # Correct block place

            # Sanity checks
            assert np.all(red+yellow+purple+blue+green <= 1)
            assert np.sum(red) + np.sum(blue) + np.sum(green) == np.sum(should_be_there)

            err = np.sum(yellow | red) 
            print err, gt.sum(), err/float(gt.sum())

        import re
        re.match('dfsdf', name)

        #_,_,err1,_,_ = grid.xcorr_correction(out, config.GT)
        #assert err == err1/float(gt.sum())
        d[name] = {'err': err,
                   'tot': (red+green+blue).sum(),
                   'rel': err/float((red+green+blue).sum()),
                   'num': number,
                   'type': kind,}
        d[name].update(dict(green=green,
                            red=red,
                            yellow=yellow,
                            purple=purple,
                            blue=blue))
    with open('data/experiments/error_output.pkl','w') as f:
        pickle.dump(d, f)


def bar_chart(B, types, std=None, xitems=[]):
    """
    B should be (len(gitems))x(len(xitems)) array
    """
    width = 1.0/(len(types)+0.5)
    colors = ['b','y','g','r']
    d = []
    for i in range(len(B)):
        xind = np.arange(len(B[i]))
        stdi = None if std is None else std[i]
        d.append(pylab.bar(xind+i*width, B[i], width,
                           color=colors[i]))
        pylab.errorbar(xind+(i+0.5)*width, B[i], fmt=None,
                       yerr=stdi, ecolor='k')
    pylab.xticks(xind+0.5, xitems)
    return d


def chart_error():
    global d
    with open('data/experiments/error_output.pkl') as f:
        d = pickle.load(f)

    types = ('none', 'add', 'remove')
    nums = np.arange(1,6)

    def unfold(key):
        r = [[[_[key]
               for _ in d.values() if _['type']==t and _['num']==n]
              for n in nums]
             for t in types]
        return np.array(r)

    std = unfold('rel').std(2)

    global red, blue, green, purple, yellow
    red = unfold('red').sum(3).sum(3).sum(3)
    green = unfold('green').sum(3).sum(3).sum(3)
    purple = unfold('purple').sum(3).sum(3).sum(3)
    yellow = unfold('yellow').sum(3).sum(3).sum(3)
    blue = unfold('blue').sum(3).sum(3).sum(3)

    rel = ((red+yellow)/(green+blue+red).astype('f')).mean(2)

    N = unfold('rel').shape[2]

    print 'Total mean: ', np.mean(rel)
    std = 1.96* np.array(std) / np.sqrt(N)

    #inds = np.argsort(rel.mean(0))
    #rel = np.array(rel[:,inds[::-1]])
    #std = np.array(std[:,inds[::-1]])

    pylab.close(pylab.figure(1))
    pylab.figure(1, figsize=(6,3))
    p = bar_chart(rel, types)
    pylab.legend([_[0] for _ in p], types)
    pylab.ylabel('Relative Error')
    pylab.yticks(np.arange(0,0.41,0.1))
    pylab.draw()
    pylab.savefig('www/experiment.png')


if __name__ == '__main__':
    #run_error()
    chart_error()
