import cv
import pylab
import numpy as np

from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl


def show_normals(n, w, name='normals'):
    im = cv.CreateImage((n.shape[1],n.shape[0]), 32, 3)
    cv.SetData(im, np.dstack(3*[w]) * n[:,:,::-1]/2+.5)
    cv.ShowImage(name, im)


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
                            np.array(from_rect(maskR,rectR)), rectR, 6)

    nwL,nwR = normals.opencl.get_normals()
    nL,wL = nwL[:,:,:3], nwL[:,:,3]
    nR,wR = nwR[:,:,:3], nwR[:,:,3]
    show_normals(nL, wL, 'normalsL')
    show_normals(nR, wR, 'normalsR')

    pylab.waitforbuttonpress(0.01)


def go():
    while 1: once()

if __name__ == '__main__':
    dataset.load_random_dataset()
    #go()
