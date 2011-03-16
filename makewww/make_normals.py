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

from blockplayer import dataset
from blockplayer import normals

out_path = os.path.join('www','normal_ims')


def show_normals(n, w):
    pylab.imshow(np.dstack(3*[w]) * n/2+.5)


def write_normals():
    runs = []
    for d in ds:
        run = dict(name=d['name'],entries=[])
        run['entries'].append(dict(name='depth',src='depth.jpg'))
        for k in 'numpy c opencl'.split():
            run['entries'].append(dict(name=k,dt=d[k],src='normals_%s.jpg'%k))
        runs.append(run)
    t = template.Template("""
    <h3>Normals</h3>
    {% for run in runs %}
    <h4>{{ run.name }}</h4>
    <div style='width:100%'>
      {% for entry in run.entries %}
      <div style='float:left'>
       <a href='{{run.name}}/{{entry.src}}'>
       <img src='{{run.name}}/{{entry.src}}' width='160px' height='120px' />
       </a>
       <div>{{ entry.name }}
       {% if entry.dt %} ({{ entry.dt|floatformat:4 }}s){% endif %}
       </div>
      </div>
      {% endfor %}
    </div>
    {% endfor %}
    """)
    with open(os.path.join(out_path,'index.html'),'w') as f:
        f.write(t.render(template.Context(locals())))


def run_normals():
    global ds
    ds = []

    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)

    for i in range(1):
        dataset.load_random_dataset()
        dataset.advance()

        name = 'dataset_%s' % str(i)
        d = dict(name=name)
        folder = os.path.join(out_path, name)
        os.mkdir(folder)

        depthL = dataset.depthL.astype('f')
        pylab.figure(0)
        pylab.clf()
        pylab.imshow(depthL)
        pylab.savefig(os.path.join(folder,'depth.jpg'))

        dt = timeit.timeit(lambda: normals.normals_numpy(depthL),
                           number=1)
        d['numpy'] = dt
        n,w = normals.normals_numpy(depthL)
        pylab.clf()
        show_normals(n, w)
        pylab.savefig(os.path.join(folder,'normals_numpy.jpg'))

        dt = timeit.timeit(lambda: normals.normals_c(depthL),
                           number=1)
        d['c'] = dt
        n,w = normals.normals_c(depthL)
        pylab.clf()
        show_normals(n, w)
        pylab.savefig(os.path.join(folder,'normals_c.jpg'))

        rect = ((0,0),(640,480))
        mask = np.zeros((480,640),'bool')
        mask[1:-1,1:-1] = 1
        normals.opencl.set_rect(rect, ((0,0),(0,0)))
        dt = timeit.timeit(lambda:
                        normals.normals_opencl(depthL, mask, rect).wait(),
                           number=1)
        d['opencl'] = dt
        nw,_ = normals.opencl.get_normals()
        n,w = nw[:,:,:3], nw[:,:,3]
        pylab.clf()
        show_normals(n,w)
        pylab.savefig(os.path.join(folder,'normals_opencl.jpg'))

        ds.append(d)


if __name__ == "__main__":
    if not 'ds' in globals():
        run_normals()
    write_normals()
