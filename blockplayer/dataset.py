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

depth = None

current_path = None
frame_num = None


def advance(skip=1):
    # Load the image
    global frame_num, depth, rgb
    frame_num += skip
    with gzip.open('%s/depth_%05d.npy.gz' % (current_path, frame_num),
                   'rb') as f:
        depth = np.load(f)
    try:
        rgb = cv.LoadImage('%s/rgb_%05d.png' % (current_path, frame_num))
        cv.CvtColor(rgb, rgb, cv.CV_RGB2BGR)
        rgb = np.fromstring(rgb.tostring(),'u1').reshape(480,640,3)
    except KeyboardInterrupt:
        rgb = None


def setup_opencl():
    opencl.setup_kernel((config.bg['KK'],config.bg['Ktable']))


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
    with open(os.path.join(current_path, 'config/gt.txt'),'r') as f:
        s = f.read()
    return grid.gt2grid(s)


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
    print "Go wget it yourself. %s" % url
    cmd = "wget %s; cd data/; tar -xzf datasets.tar.gz data/"
    print cmd


if __name__ == "__main__":
    pass
