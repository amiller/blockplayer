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
    while 1:
        dataset.advance()
        depthL = dataset.depthL.astype('f')

        #n,w = normals.normals_numpy(depthL)
        #show_normals(n, w, 'normals_numpy')

        #n,w = normals.normals_c(depthL)
        #show_normals(n, w, 'normals_c')

        rect = ((0,0),(640,480))
        mask = np.zeros((480,640),'bool')
        mask[1:-1,1:-1] = 1
        normals.opencl.set_rect(rect, ((0,0),(0,0)))
        dt = timeit.timeit(lambda:
                           normals.normals_opencl(depthL, mask, rect).wait(),
                           number=1)
        #print dt
        nw,_ = normals.opencl.get_normals()
        n,w = nw[:,:,:3], nw[:,:,3]
        show_normals(n, w, 'normals_opencl')

        pylab.waitforbuttonpress(0.01)


if __name__ == "__main__":
    dataset.load_random_dataset()
    go()
