import numpy as np
import os

from ctypes import POINTER as PTR, c_byte, c_ushort, c_size_t
speedup_ctypes = np.ctypeslib.load_library('speedup_ctypes.so',
                                           os.path.dirname(__file__))
speedup_ctypes.inrange.argtypes = [PTR(c_ushort), PTR(c_byte), PTR(c_ushort),
                                   PTR(c_ushort), c_size_t]
import speedup_cy


def threshold_and_mask(depth,bg):
    import scipy
    from scipy.ndimage import binary_erosion

    def m_():
        # Optimize this in C?
        return (depth>bg['bgLo']) & (depth<bg['bgHi'])  # background

    def m2_():
        mm = np.empty((480,640),'bool')
        speedup_ctypes.inrange(depth.ctypes.data_as(PTR(c_ushort)),
                               mm.ctypes.data_as(PTR(c_byte)),
                               bg['bgHi'].ctypes.data_as(PTR(c_ushort)),
                               bg['bgLo'].ctypes.data_as(PTR(c_ushort)),
                               480*640)
        return mm

    def m3_():
        mm = np.empty((480,640),'u1')
        speedup_cy.inrange(depth,
                           mm,
                           bg['bgHi'],
                           bg['bgLo'], 480*640)
        return mm

    mask = m3_()
    dec = 3
    dil = binary_erosion(mask[::dec,::dec],iterations=2)
    slices = scipy.ndimage.find_objects(dil)
    a,b = slices[0]
    (l,t),(r,b) = (b.start*dec-10,a.start*dec-10),(b.stop*dec+7,a.stop*dec+7)
    b += -(b-t)%16
    r += -(r-l)%16
    if t<0: t+= 16
    if l<0: l+= 16
    if r>=640: r-= 16
    if b>=480: b-= 16
    return mask, ((l,t),(r,b))
