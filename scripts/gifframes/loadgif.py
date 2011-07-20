import Image
import numpy as np
import cv
import StringIO
import glob

def pil2np(im):
    return np.fromstring(im.tostring(), dtype='u1').reshape(im.size[::-1] + (1,))

def load_frames(foldername, size=None, thresh=50):
    """Looks in <foldername>/*.gif for frames, gets them all"""
    frames = []
    for f in sorted(glob.glob('%s/*.gif' % foldername)):
        im = Image.open(f)
        if size is not None: im = im.resize(size)
        im = im.convert('L')
        frames.append((pil2np(im)>=thresh).astype('u1')*255)
    return frames

if __name__ == '__main__':
    print """Play a bunch of firework gifs
    Run convert.sh first if there isn't anything here."""
    files = glob.glob('*.gif')
    def show_them_all(size=None):
        for f in files:
            print 'file: ', f
            frames = load_frames(f + '_frames', size)
            for f in frames:
                cv.ShowImage('test', f)
                cv.WaitKey(100)
    show_them_all()
    show_them_all((256,256))

