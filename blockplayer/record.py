import freenect
import numpy as np
import os
import shutil
import cv
import subprocess
import dataset


def show_depth(name, depth):
    im = cv.CreateImage((depth.shape[1],depth.shape[0]), 32, 1)
    cv.SetData(im, depth.astype('f') / depth.max())
    cv.ShowImage(name, im)


def preview():
    (depthL,_) = freenect.sync_get_depth(1)
    (depthR,_) = freenect.sync_get_depth(0)
    show_depth('depthL', depthL)
    show_depth('depthR', depthR)


def go():
    while 1:
        preview()
        cv.WaitKey(10)


def record(filename=None):
    if filename is None:
        filename = str(np.random.rand())

    foldername = 'data/sets/%s/' % filename
    dataset.folder = foldername

    os.mkdir(foldername)
    shutil.copytree('data/newest_calibration/config', '%s/config' % foldername)
    print "Created new dataset: %s" % foldername

    frame = 0
    try:
        while 1:
            (depthL,_) = freenect.sync_get_depth(0)
            (depthR,_) = freenect.sync_get_depth(1)

            np.save('%s/depthL_%05d.npy' % (foldername,frame), depthL)
            np.save('%s/depthR_%05d.npy' % (foldername,frame), depthR)

            if frame % 30 == 0:
                print 'frame: %d' % frame
            frame = frame + 1
    except KeyboardInterrupt:
        print "Captured %d frames" % frame
        compress()


def compress():
    cmd = "gzip %s/*.npy" % (dataset.folder,)
    print "Running %s" % cmd
    import sys
    sys.stdout.flush()
    subprocess.call(cmd, shell=True)
