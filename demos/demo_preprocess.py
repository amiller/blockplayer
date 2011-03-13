from blockplayer import dataset
from blockplayer import normals
from blockplayer import preprocess
from blockplayer import config
import cv
import pylab
import numpy as np


def show_mask(name, m, rect):
    im = cv.CreateImage((m.shape[1],m.shape[0]), 32, 3)
    cv.SetData(im, np.ascontiguousarray(np.dstack(3*[m])))
    (t,l),(b,r) = rect
    cv.Rectangle(im, (t,l), (b,r), (255,255,0))
    cv.ShowImage(name, im)


def once():
    dataset.advance()
    depthL,depthR = dataset.depthL,dataset.depthR
    maskL,rectL = preprocess.threshold_and_mask(depthL,config.bgL)
    maskR,rectR = preprocess.threshold_and_mask(depthR, config.bgR)
    show_mask('maskL', maskL.astype('f'), rectL)
    show_mask('maskR', maskR.astype('f'), rectR)

    pylab.waitforbuttonpress(0.01)


def go():
    while 1: once()


def show_backgrounds():
    pylab.figure(1)
    pylab.imshow(config.bgL['bgHi'])
    pylab.draw()
    pylab.figure(2)
    pylab.clf()
    pylab.imshow(config.bgR['bgHi'])
    pylab.draw()


if __name__ == "__main__":
    dataset.load_random_dataset()
    go()
