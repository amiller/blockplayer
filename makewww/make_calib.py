#import www
import os
import shutil
import matplotlib
matplotlib.use('agg')
import pylab
import numpy as np
import timeit
import django.conf
try:
    django.conf.settings.configure()
except:
    pass
from django import template

from blockplayer import config
from blockplayer import dataset
from blockplayer import table_calibration as table_calibration

out_path = os.path.join('www','calib')
if not 'ds' in globals(): ds = None

import glxcontext
glxcontext.init()
import sys
sys.stdout.write("GL Version String: ")
glxcontext.printinfo()


def write_calib():
    runs = []
    global ds
    for d in ds:
        run = dict(name=d['name'],dt=d['dt'])
        run['sides'] = []
        for side in 'left', 'right':
            s = dict(side=side, entries=[])
            for k in 'depth bglo bghi'.split():
                s['entries'].append(dict(name=k,src='%s_%s.jpg'%(k, side)))
            run['sides'].append(s)
        runs.append(run)

    t = template.Template("""
    <h3>Table Calibration</h3>
    {% for run in runs %}
    <h4>{{ run.name }}</h4>
    <div>Compute time: <em>{{ run.dt|floatformat:4 }}s</em></div>
    {% for side in run.sides %}
    <div style='width:100%'>
      {% for entry in side.entries %}
      <div style='float:left'>
       <a href='{{run.name}}/{{entry.src}}'>
       <img src='{{run.name}}/{{entry.src}}' width='160px' height='120px' />
       </a>
       <div>{{ entry.name }}_{{ side.side }}</div>
      </div>
      {% endfor %}
      <div style='clear:both'></div>
    </div>
    {% endfor %}
    {% endfor %}
    """)
    with open(os.path.join(out_path,'index.html'),'w') as f:
        f.write(t.render(template.Context(locals())))


def run_calib():
    global ds
    ds = []

    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)

    for i in range(1):
        dataset.load_random_dataset()
        table_calibration.newest_folder = dataset.current_path

        name = 'dataset_%s' % str(i)
        d = dict(name=name)
        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        dt = timeit.timeit(lambda: table_calibration.finish_cube_calib(),
                           number=1)
        d['dt'] = dt

        depthL, depthR = table_calibration.depthL, table_calibration.depthR
        bgL, bgR = config.bgL, config.bgR
        for depth, side, bg in (depthL, 'left', bgL), (depthR, 'right', bgR):
            pylab.figure(0)
            pylab.clf()
            pylab.imshow(depth)
            pylab.savefig(os.path.join(folder,'depth_%s.jpg' % side))

            pylab.clf()
            pylab.imshow(bg['bgHi'])
            pylab.savefig(os.path.join(folder,'bghi_%s.jpg' % side))

            pylab.imshow(bg['bgLo'])
            pylab.savefig(os.path.join(folder,'bglo_%s.jpg' % side))

        ds.append(d)


if __name__ == "__main__":
    if ds is None:
        run_calib()
    write_calib()
