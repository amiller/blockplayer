import freenect
import numpy as np
import os
import shutil
import cv
import subprocess
import dataset


def record(filename=None):
    if filename is None:
        filename = str(np.random.rand())

    foldername = 'data/sets/%s/' % filename
    dataset.folder = foldername

    os.mkdir(foldername)
    shutil.copytree('data/newest_calibration', '%s/calib' % foldername)
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
         infe compress()


def compress():
    cmd = "gzip %s/*.npy" % (dataset.folder,)
    print "Running %s" % cmd
    subprocess.call(cmd, shell=True)
