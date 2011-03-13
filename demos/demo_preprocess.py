from blockplayer import dataset
from blockplayer import normals
from blockplayer import preprocess
from blockplayer import config
import cv
import pylab


def show_mask(name, m):
    im = cv.CreateImage((m.shape[1],m.shape[0]), 32, 1)
    cv.SetData(im, m)
    cv.ShowImage(name, im)


def once():
    dataset.advance()
    depthL,depthR = dataset.depthL,dataset.depthR
    maskL,rectL = preprocess.threshold_and_mask(depthL,config.bgL)
    maskR,rectR = preprocess.threshold_and_mask(depthR, config.bgR)
    show_mask('maskL', maskL.astype('f'))
    show_mask('maskR', maskR.astype('f'))
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
