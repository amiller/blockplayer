import numpy as np
import pylab
from OpenGL.GL import *
import opennpy

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from wxpy3d.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_lattice', size=(640,480))

from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import grid

def once():
    if not FOR_REAL:
        dataset.advance()
        global depth
        depth = dataset.depth
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global mask, rect

    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    # Compute the surface normals
    normals.normals_opencl(depth, mask, rect)

    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()
    R_aligned = lattice.translation_opencl(R_oriented)

    global modelmat
    if modelmat is None:
        modelmat = R_aligned.copy()
    else:
        modelmat,_ = grid.nearest(modelmat, R_aligned)

    global face, Xo, Yo, Zo
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]

    window.update_xyz(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))

    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.005)


def resume():
    while 1: once()


def start(dset=None, frame_num=0):
    global modelmat
    modelmat = None

    if not FOR_REAL:
        if dset is None:
            dataset.load_random_dataset()
        else:
            dataset.load_dataset(dset)
        while dataset.frame_num < frame_num:
            dataset.advance()
    else:
        config.load('data/newest_calibration')
        dataset.setup_opencl()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()

if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    print """BROKEN DEMO ALERT! I think this demo is out of date. If 
I fix it, I'll probably remember to remove this message."""
    pass
    #go()
