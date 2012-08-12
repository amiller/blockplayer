# Andrew Miller <amiller@cs.ucf.edu> 2011
#
# BlockPlayer - 3D model reconstruction using the Lattice-First algorithm
# See: 
#    "Interactive 3D Model Acquisition and Tracking of Building Block Structures"
#    Andrew Miller, Brandyn White, Emiko Charbonneau, Zach Kanzler, and Joseph J. LaViola Jr.
#    IEEE VR 2012, IEEE TVGC 2012
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

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


def preview(cams):
    opennpy.sync_update()
    global depth_cache
    for cam in cams:
        (depth,_) = opennpy.sync_get_depth(cam)
        depth_cache.append(np.array(depth))
        depth_cache = depth_cache[-6:]
        show_depth('depth_%d'%cam, depth)


def go(cams=(0,)):
    opennpy.align_depth_to_rgb()
    while 1:
        preview(cams)
        pylab.waitforbuttonpress(0.005)


def record(filename=None, cams=(0,), do_rgb=False):
    if len(cams) > 1 and do_rgb:
        print """You're trying to record from 2+ kinects with RGB and depth.
        This probably will not work out for you, but it depends on if you have
        enough USB bandwidth to support all four streams. Call record with 
        do_rgb=False to turn off rgb."""

        
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
            for cam in cams:
                (depth,_) = opennpy.sync_get_depth(cam)
                np.save('%s/depth_%05d_%d.npy' % (foldername,frame,cam), depth)

                if do_rgb:
                    (rgb,_) = opennpy.sync_get_video(cam)
                    cv.CvtColor(rgb, rgb, cv.CV_RGB2BGR)
                    cv.SaveImage('%s/rgb_%05d_%d.png' % (foldername,frame,cam), rgb)
            
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
