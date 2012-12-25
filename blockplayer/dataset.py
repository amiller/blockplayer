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

import numpy as np
import shutil
import glob
import gzip_patch
import gzip
import config
import opencl
import os
import grid
import cv

depths = []
rgbs = []
current_path = None
frame_num = None


def advance(skip=1):
    # Load the image
    global frame_num, depths, rgbs
    frame_num += skip
    depths = []
    rgbs = []
    for cam in range(len(config.cameras)):
        try:
            # Try the old style with no frame field (backward compatible)
            with gzip.open('%s/depth_%05d.npy.gz' % (current_path, frame_num),
                           'rb') as f:
                depths.append(np.load(f))
        except IOError:
            with gzip.open('%s/depth_%05d_%d.npy.gz' % (current_path, frame_num, cam),
                           'rb') as f:
                depths.append(np.load(f))
        try:
            rgb = cv.LoadImage('%s/rgb_%05d_%d.png' % (current_path, frame_num,cam))
            cv.CvtColor(rgb, rgb, cv.CV_RGB2BGR)
            rgbs.append(np.fromstring(rgb.tostring(),'u1').reshape(480,640,3))
        except IOError:
            rgbs = []


def setup_opencl():
    cam = config.cameras[0]
    opencl.setup_kernel((cam.KK,cam.RT))


def load_dataset(pathname):
    global current_path, frame_num

    # Check for consistency, count the number of images
    current_path = pathname

    # Load the config
    config.load(current_path)
    try:
        config.GT = load_gt()
    except IOError:
        config.GT = None

    setup_opencl()

    frame_num = 0


def load_gt():
    name = current_path
    name = os.path.split(name)[1]
    custom = os.path.join('data/sets/', name, 'gt.txt')
    try:
        if os.path.exists(custom):
            # Try dataset directory first
            fname = custom
        else:
            import re
            # Fall back on generic ground truth file
            match = re.match('.*_z(\d)m_(.*)', name)
            number = int(match.groups()[0])
            fname = 'data/experiments/gt/gt%d.txt' % number
            print 'Initializing with groundtruth'

        with open(fname) as f:
            GT = grid.gt2grid(f.read())
        grid.initialize_with_groundtruth(GT)

    except AttributeError: # re.match failed
        print 'Initializing without groundtruth'
        grid.initialize()


def load_random_dataset():
    # Look in the datasets folder, find all the datasets
    # pick one
    sets = glob.glob('data/sets/*/')
    import random
    choice = random.choice(sets)
    load_dataset(choice)


def download():
    # Download from this url
    url = "https://s3.amazonaws.com/blockplayer/datasets.tar.gz"
    print "Use wget: %s" % url
    cmd = "wget %s; cd data/; tar -xzf datasets.tar.gz data/"
    print cmd


if __name__ == "__main__":
    pass
