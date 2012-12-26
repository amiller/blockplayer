from blockplayer import dataset
from blockplayer import config
from blockplayer import opencl
from blockplayer import lattice
from rtmodel.rangeimage import RangeImage

import numpy as np
import unittest

def load_first():
    dataset.load_dataset('data/sets/study_user3_z1m_add')
    dataset.advance()

def from_rect(m,rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]

def oriented_numpy():
    bg = config.bg[0]
    cam = config.cameras[0]
    rimg = RangeImage(dataset.depths[0], cam)
    rimg.threshold_and_mask(bg)
    rimg.filter()
    rimg.compute_normals()
    rimg.compute_points()
    R_oriented = lattice.orientation_numpy(rimg.normals, rimg.weights)
    assert R_oriented.shape == (4,4)
    return R_oriented

def oriented_opencl():
    bg = config.bg[0]
    cam = config.cameras[0]
    rimg = RangeImage(dataset.depths[0], cam)
    rimg.threshold_and_mask(bg)
    rimg.filter()
    opencl.set_rect(rimg.rect)
    opencl.load_filt(rimg.depth_filtered)
    opencl.load_raw(rimg.depth_recip)
    opencl.load_mask(from_rect(rimg.mask, rimg.rect).astype('u1'))
    opencl.compute_normals().wait()
    R_oriented = lattice.orientation_opencl()
    assert R_oriented.shape == (4,4)
    return R_oriented

class OrientationTest(unittest.TestCase):
    def test_orientation(self):
        load_first()
        R1 = oriented_numpy()
        R2 = oriented_opencl()

        within_eps = lambda a, b: np.abs(a - b) < 1e-5
        assert np.all(within_eps(R1,R2))

if __name__ == '__main__':
    unittest.main()
