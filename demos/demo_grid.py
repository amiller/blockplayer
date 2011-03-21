import numpy as np
import pylab
from OpenGL.GL import *

import blockplayer.visuals.pointwindow
from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice
from blockplayer import flatrot
from blockplayer import grid

from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_grid', size=(640,480))


def once():
    dataset.advance()
    global depthL,depthR
    depthL,depthR = dataset.depthL,dataset.depthR

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global maskL, rectL
    global maskR, rectR

    (maskL,rectL) = preprocess.threshold_and_mask(depthL,config.bgL)
    (maskR,rectR) = preprocess.threshold_and_mask(depthR,config.bgR)

    opencl.set_rect(rectL,rectR)
    normals.normals_opencl2(from_rect(depthL,rectL).astype('f'),
                            np.array(from_rect(maskL,rectL)), rectL,
                            from_rect(depthR,rectR).astype('f'),
                            np.array(from_rect(maskR,rectR)), rectR,
                            6)

    mat = np.eye(4,dtype='f')
    mat[:3,:3] = flatrot.flatrot_opencl()

    mat = lattice.lattice2_opencl(mat)

    global face, Xo, Yo, Zo
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)

    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]
    #update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))

    global modelmat
    modelmat = mat

    grid.add_votes_opencl(lattice.meanx, lattice.meanz)

    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.005)


def resume():
    while 1: once()


def go():
    dataset.load_random_dataset()
    resume()


def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
    global modelmat
    if not 'modelmat' in globals():
        return

    xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
    mask = Z.flatten()<10
    xyz = xyz[mask,:]

    global axes_rotation
    axes_rotation = np.eye(4)
    if not AXES is None:
        # Rotate the axes
        axes_rotation[:3,:3] = expmap.axis2rot(-AXES)
    window.upvec = axes_rotation[:3,1]

    if not COLOR is None:
        R,G,B,A = COLOR
        color = np.vstack((R.flatten(), G.flatten(), B.flatten(),
                           A.flatten())).transpose()
        color = color[mask,:]

    window.update_points(xyz, color)

    @window.event
    def post_draw():
        pass

if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
