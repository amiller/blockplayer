import os
import shutil
import numpy as np
import django.conf
import time
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

    t = template.Template("""
    <title>Blockplayer</title>
    <h3>Blockplayer Output</h3>
    {% for run in runs %}
    <div style='float:left'>
    <h4>{{ run.name }}
      <a href="{{ run.name }}_block.html">[fullscreen]</a>
    </h4>
    <div>
       {{ run.frames }} frames in {{ run.time }} seconds ({{ run.fps }} fps)
    </div>
    <iframe src="{{ run.name }}_block.html" width="256" height="256">
    </iframe>
    </div>
    {% endfor %}
    """)
    with open(os.path.join(out_path,'index.html'),'w') as f:
        f.write(t.render(template.Context(locals())))


def run_grid():
    global ds
    ds = []

    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)
    shutil.copy('makewww/ThreeCanvas.js',out_path)
    shutil.copy('makewww/Plane.js',out_path)
    shutil.copy('makewww/Cube.js',out_path)

    with open('makewww/blockviewiframetemplate.html','r') as f:
        tmp = template.Template(f.read())

    datasets = glob.glob('data/sets/*')
    for name in datasets:
    #for name in ('data/sets/cube',):
        dataset.load_dataset(name)
        name = os.path.split(name)[1]

        import re
        number = int(re.match('.*_z(\d)m_.*', name).groups()[0])
        with open('data/experiments/gt/gt%d.txt' % number) as f:
            config.GT = grid.gt2grid(f.read())

        try:
            with open(os.path.join('data/experiments/output/',name,
                                   'output.pkl'),'r') as f:
                output = pickle.load(f)
        except IOError:
            continue
        with open(os.path.join('data/experiments/output/',name,
                               'final_output.txt'),'r') as f:
            final_output = f.read()

        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        print name
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

        ds.append(output)
    return ds


if __name__ == "__main__":
    ds = run_grid()
    write_grid(ds)
