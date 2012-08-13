from blockplayer import dataset
from blockplayer import config
from blockplayer import opencl
from rtmodel.rangeimage import RangeImage
from rtmodel.camera import kinect_camera

import numpy as np
import unittest

def load_first():
    dataset.load_dataset('data/sets/study_user3_z1m_add')
    
def from_rect(m,rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]
    
class DatasetTest(unittest.TestCase):
    def test_load_first(self, *args):
        load_first()
        dataset.advance()
        depth, = dataset.depths
        assert depth.dtype == np.uint16
        assert depth.shape == (480,640)
        assert depth.shape == (480,640)
        

class NormalsTest(unittest.TestCase):
    def test_normals(self):

        global rimg
        global nw
        load_first()
        depth, = dataset.depths

        cam, = config.cameras
        rimg = RangeImage(depth, cam)
        rimg.threshold_and_mask(config.bg[0])
        rimg.filter()

        # Numpy Normals
        rimg.compute_normals()

        # Opencl Normals
        opencl.set_rect(rimg.rect)
        opencl.load_filt(rimg.depth_filtered)
        opencl.load_raw(rimg.depth_recip)
        opencl.load_mask(from_rect(rimg.mask, rimg.rect).astype('u1'))
        opencl.compute_normals().wait()
        nw = opencl.get_normals()

        # Compare the OpenCL normals to the gold standard
        global n1, n2
        n1 = rimg.normals*np.dstack(3*[from_rect(rimg.mask,rimg.rect)*rimg.weights])
        n2 = nw[:,:,:3]*np.dstack(3*[from_rect(rimg.mask,rimg.rect)*rimg.weights])

        within_eps = lambda a, b: np.abs(a - b) < 1e-5
        assert np.all(within_eps(n1,n2))


if __name__ == '__main__':
    unittest.main()
        


