import numpy as np
import shutil
import glob
import gzip_patch
import gzip
import config
import opencl

depthL = None
depthR = None

current_path = None
frame_num = None


def advance(skip=1):
    # Load the next pair of images
    global frame_num, depthL, depthR
    frame_num += skip
    with gzip.open('%s/depthL_%05d.npy.gz' % (current_path, frame_num),
                   'rb') as f:
        depthL = np.load(f)
    with gzip.open('%s/depthR_%05d.npy.gz' % (current_path, frame_num),
                   'rb') as f:
        depthR = np.load(f)


def load_dataset(pathname):
    global current_path, frame_num

    # Check for consistency, count the number of images
    current_path = pathname

    # Load the config
    config.load(current_path)
    opencl.setup_kernel((config.bgL['KK'],config.bgL['Ktable']),
                        (config.bgR['KK'],config.bgR['Ktable']))

    frame_num = 0


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
