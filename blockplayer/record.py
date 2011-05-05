import opennpy
import numpy as np
import os
import shutil
import cv
import subprocess
import dataset
import colormap
import pylab


def show_depth(name, depth):
    im = cv.CreateImage((depth.shape[1],depth.shape[0]), 8, 3)
    cv.SetData(im, colormap.color_map(depth))
    cv.ShowImage(name, im)


depth_cache = []


def preview():
    opennpy.sync_update()
    (depth,_) = opennpy.sync_get_depth()
    global depth_cache
    depth_cache.append(np.array(depth))
    depth_cache = depth_cache[-5:]
    show_depth('depth', depth)


def go():
    opennpy.align_depth_to_rgb()
    while 1:
        preview()
        pylab.waitforbuttonpress(0.005)


def record(filename=None):
    opennpy.align_depth_to_rgb()
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
            opennpy.sync_update()
            (depth,_) = opennpy.sync_get_depth()
            (rgb,_) = opennpy.sync_get_video()

            np.save('%s/depth_%05d.npy' % (foldername,frame), depth)

            cv.CvtColor(rgb, rgb, cv.CV_RGB2BGR)
            cv.SaveImage('%s/rgb_%05d.png' % (foldername,frame), rgb)

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
