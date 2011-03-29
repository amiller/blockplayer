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

from blockplayer import config
from blockplayer import dataset
from blockplayer import grid
from blockplayer import lattice
from blockplayer import flatrot
from blockplayer import preprocess
from blockplayer import opencl
from blockplayer import normals

out_path = os.path.join('www','grid')
if not 'ds' in globals(): ds = None

from blockplayer import glxcontext
glxcontext.init()
import sys
sys.stdout.write("GL Version String: ")
glxcontext.printinfo()

modelmat = None


def once():
    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    depth = dataset.depth
    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    opencl.set_rect(rect)
    normals.normals_opencl(from_rect(depth,rect).astype('f'),
                           np.array(from_rect(mask,rect)), rect,
                           6)

    global modelmat

    mat = np.eye(4,dtype='f')
    if modelmat is None:
        mat[:3,:3] = flatrot.flatrot_opencl()
        mat = lattice.lattice2_opencl(mat)
    else:
        mat = modelmat.copy()
        mat[:3,:3] = flatrot.flatrot_opencl(modelmat[:3,:])
        mat = lattice.lattice2_opencl(mat)

    grid.add_votes(lattice.meanx, lattice.meanz, depth, use_opencl=True)
    modelmat = lattice.modelmat


def write_grid():
    runs = []
    global ds
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
    <h4>{{ run.name }}
      <a href="{{ run.name }}_block.html">[fullscreen]</a>
    </h4>
    <div>
       {{ run.frames }} frames in {{ run.time }} seconds ({{ run.fps }} fps)
    </div>
    <iframe src="{{ run.name }}_block.html" width="256" height="256">
    </iframe>
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
        dataset.load_dataset(name)
        name = os.path.split(name)[1]

        d = dict(name=name)
        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        global modelmat
        modelmat = None
        grid.initialize()

        t1 = time.time()
        while 1:
            try:
                dataset.advance()
            except (IOError, ValueError):
                break
            once()
        t2 = time.time()

        d['frames'] = dataset.frame_num
        d['time'] = t2-t1

        g = np.array(np.nonzero(grid.vote_grid>30)).transpose() + \
            grid.bounds[0][:3]
        gstr = json.dumps(g.tolist())

        with open(os.path.join(out_path, '%s_block.html' % name),'w') as f:
            f.write(tmp.render(template.Context(dict(gstr=gstr))))

        ds.append(d)


if __name__ == "__main__":
    if ds is None:
        run_grid()
    write_grid()
