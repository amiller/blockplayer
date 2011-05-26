import os
import shutil
import numpy as np
import django.conf
import time
import re
try:
    django.conf.settings.configure()
except:
    pass
from django import template
import simplejson as json
import glob
import cPickle as pickle


from blockplayer import config
from blockplayer import dataset
from blockplayer import grid
from blockplayer import hashalign

out_path = os.path.join('www','grid')


if not 'ds' in globals(): ds = None


def write_grid(ds):
    runs = []
    for d in ds:
        run = dict(name=d['name'],
                   frames=d['frames'],
                   time='%.2f' % d['time'],
                   fps='%.2f' % (d['frames'] / d['time']),
                   total=d['total'],
                   incorrect=d['incorrect'],
                   error=d['error'])
        runs.append(run)

    with open('makewww/index.html','r') as f:
        t = template.Template(f.read())
    
    numruns = len(runs)
    for i in xrange(0, numruns, 3):
        name = '%d.html' % (i/3+1)
        if i == 0:
            name = "index.html"
        with open(os.path.join(out_path, name), 'w') as f:
            f.write(t.render(template.Context({
                "runs": runs[i:i+3],
                "page": i/3+1,
                "pages": xrange(1,numruns/3+1),
                "numpages": numruns/3+1,
            })))

def gridstr(g):
    g = np.array(np.nonzero(g))
    g = g.transpose() + config.bounds[0][:3]
    gstr = json.dumps(g.tolist())
    return gstr

def run_grid(clearout=True):
    global ds
    ds = []
    
    if clearout:
      try:
          shutil.rmtree(out_path)
      except:
          pass
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    
    for name in ('ThreeCanvas.js','Plane.js','Cube.js','video.js',
            'video-js.css','default.jpg','blocksetup.js'):
        shutil.copy('makewww/' + name, out_path)

    with open('makewww/blockviewiframetemplate.html','r') as f:
        tmp = template.Template(f.read())

    datasets = glob.glob('data/sets/study*')
    datasets.sort()
    total = len(datasets)
    for x,name in enumerate(datasets):
        dataset.load_dataset(name)
        name = os.path.split(name)[1]
        
        # Open the ground truth file
        custom = os.path.join('data/sets/', name, 'gt.txt')
        if os.path.exists(custom):
            # Try dataset directory first
            fname = custom
        else:
            # Fall back on generic ground truth file
            match = re.match('.*_z(\d)m_.*', name)
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

        try:
            with open(os.path.join('data/experiments/output/', name,
                    'output.pkl'),'r') as f:
                output = pickle.load(f)
        except IOError:
            continue
        
        with open(os.path.join('data/experiments/output/',name,
                'final_output.txt'),'r') as f:
            final_output = f.read()

        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        print "(%d/%d)" % (x+1,total), name
        out = grid.gt2grid(final_output)
        gt = config.GT

        try:
            c,_ = hashalign.find_best_alignment(out, 0*out, gt, 0*gt)
        except ValueError:
            red = gt | added
            yellow = 0*out
            green = 0*out
            purple = 0*out
            blue = 0*out
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
            assert np.all((red+yellow+purple+blue+green) <= 1)
            assert np.sum(red) + np.sum(blue) + np.sum(green) == np.sum(should_be_there)

            #err = np.sum(yellow | red) 
            #print err, gt.sum(), err/float(gt.sum())
        
        # Output display
        with open(os.path.join(out_path, '%s_block.html' % name),'w') as f:
            f.write(tmp.render(template.Context(dict(
                red=gridstr(red),
                blue=gridstr(blue),
                green=gridstr(green),
                yellow=gridstr(yellow),
                purple=gridstr(purple),
                msg="Output"
            ))))
        
        # Ground truth display
        with open(os.path.join(out_path, '%s_gt.html' % name),'w') as f:
            f.write(tmp.render(template.Context(dict(
                blue=gridstr(added),
                green=gridstr(gt),
                purple=gridstr(removed),
                red='[]',
                yellow='[]',
                msg="Ground Truth"
            ))))
        
        totalblocks = np.sum(green + red + blue)
        incorrect = np.sum(red + yellow)
        
        with open(os.path.join(out_path, '%s_block.txt' % name) ,'w') as f:
            f.write(grid.gt2grid(final_output))
        
        # Only take the metadata we need from the output
        d = dict([(key,output[key]) for key in ('frames', 'time')])
        d['name'] = name
        d['total'] = totalblocks
        d['incorrect'] = incorrect
        d['error'] = float(incorrect) / totalblocks * 100
        ds.append(d)
    return ds


if __name__ == "__main__":
    ds = None
    pkl_file = "makewww_results.pkl"
    if os.path.exists(pkl_file):
        ans = raw_input("Saved results available; use them instead? ([y]/n) ")
        if not ans.startswith("n"):
            with open(pkl_file, 'r') as fp:
                ds = pickle.load(fp)
    
    if ds is None:
        ds = run_grid()
    
    with open(pkl_file, 'w') as fp:
        pickle.dump(ds, fp)
    
    write_grid(ds)

