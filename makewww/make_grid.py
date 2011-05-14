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

out_path = os.path.join('www','grid')


if not 'ds' in globals(): ds = None


def write_grid(ds):
    runs = []
    for d in ds:
        run = dict(name=d['name'],
                   frames=d['frames'],
                   time='%.2f' % d['time'],
                   fps='%.2f' % (d['frames'] / d['time']))
        runs.append(run)

    with open('makewww/index.html','r') as f:
        t = template.Template(f.read())
    
    with open(os.path.join(out_path,'index.html'), 'w') as f:
        f.write(t.render(template.Context({"runs": runs})))


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
    
    shutil.copy('makewww/ThreeCanvas.js', out_path)
    shutil.copy('makewww/Plane.js', out_path)
    shutil.copy('makewww/Cube.js', out_path)
    shutil.copy('makewww/video.js', out_path)
    shutil.copy('makewww/video-js.css', out_path)
    shutil.copy('makewww/default.jpg', out_path)

    with open('makewww/blockviewiframetemplate.html','r') as f:
        tmp = template.Template(f.read())

    datasets = glob.glob('data/sets/study*')
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
        with open(fname) as fp:
            config.GT = grid.gt2grid(fp.read())

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

        def gridstr(g):
            g = np.array(np.nonzero(g))
            g = g.transpose() + config.bounds[0][:3]
            gstr = json.dumps(g.tolist())
            return gstr

        if not config.GT is None:
            gt = config.GT
            _,gt,_,_,_ = grid.xcorr_correction(out, gt)
            red = gt & ~out
            yellow = ~gt & out
            blue = gt & out
        else:
            red = out
            yellow = 0*out
            blue = 0*out

        with open(os.path.join(out_path, '%s_block.html' % name),'w') as f:
            f.write(tmp.render(template.Context(dict(red=gridstr(red),
                                                     blue=gridstr(blue),
                                                     yellow=gridstr(yellow)))))

        with open(os.path.join(out_path, '%s_block.txt' % name) ,'w') as f:
            f.write(grid.gt2grid(final_output))
        
        # Only take the metadata we need from the output
        d = dict([(key,output[key]) for key in ('name', 'frames', 'time')])
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

