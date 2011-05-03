import numpy as np
import pylab
import opennpy
import cv

if not 'FOR_REAL' in globals():
    FOR_REAL = False

from blockplayer import config
from blockplayer import preprocess
from blockplayer import dataset
from blockplayer import colormap
from blockplayer import classify


def show_depth(name, depth):
    im = cv.CreateImage((depth.shape[1],depth.shape[0]), 8, 3)
    cv.SetData(im, colormap.color_map(depth))
    cv.ShowImage(name, im)


def once():
    global depth, rgb
    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depth
        rgb = dataset.rgb
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()
        rgb,_ = opennpy.sync_get_video()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global mask, rect, modelmat

    try:
        (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)
    except IndexError:
        return

    cv.ShowImage('mask',mask.astype('u1')*255)

    global label_image
    label_image = classify.predict(depth, mask)
    cv.ShowImage('label_image', ((label_image[0]+1)*100*mask).astype('u1'))
    pylab.waitforbuttonpress(0.03)


def resume():
    try:
        while 1:
            once()
    except IOError:
        return


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
        opennpy.align_depth_to_rgb()
        dataset.setup_opencl()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()


if __name__ == '__main__':
    pass
    #go()
