import cv
import pylab
import numpy as np
import timeit

from blockplayer import dataset
#from blockplayer import table_calibration
from blockplayer import normals


def show_normals(n, w, name='normals'):
    im = cv.CreateImage((n.shape[1],n.shape[0]), 32, 3)
    cv.SetData(im, np.dstack(3*[w]) * n[:,:,::-1]/2+.5)
    cv.ShowImage(name, im)


def go():
    while 1: once()


def once():
    dataset.advance()
    global depth
    depth = dataset.depth.astype('f')

    if 0:
        n,w = normals.normals_numpy(depth)
        show_normals(n, w, 'normals_numpy')

    if 1:
        n,w = normals.normals_c(depth)
        show_normals(n, w, 'normals_c')

    if 1:
        rect = ((0,0),(640,480))
        mask = np.zeros((480,640),'bool')
        mask[1:-1,1:-1] = 1
        normals.opencl.set_rect(rect)
        dt = timeit.timeit(lambda:
                           normals.normals_opencl(depth, mask, rect).wait(),
                           number=1)
        #print dt
        nw = normals.opencl.get_normals()
        n,w = nw[:,:,:3], nw[:,:,3]
        show_normals(n, w, 'normals_opencl')

    pylab.waitforbuttonpress(0.005)
