cimport numpy as np
import numpy as np

cdef stencil_carve_(np.float32_t * depthB,
                    np.uint16_t * depth,
                    np.uint8_t * coords,
                    int GX, int GY, int GZ,
                    np.int32_t * RGBacc,
                    np.uint8_t * rgb,
                    np.uint8_t * RGB,
                    np.float32_t * b_total,
                    np.float32_t * b_occ,
                    np.float32_t * b_vac,
                    int T, int L, int B, int R):
                    
    cdef int GYGZ = GY*GZ
    cdef int ind, dB, d, x, y, z, coord, p1
    for i in range(T, B):
        p1 = i*640
        for j in range(L, R):
            ind = p1+j
            dB = <int> depthB[ind]
            d = <int> depth[ind]
            if dB<1e6:
                x = coords[ind*4+0]
                y = coords[ind*4+1]
                z = coords[ind*4+2]
                coord = GYGZ*x + GZ*y + z
                b_total[coord] += 1
                if d>0:
                    if dB+10 < d:
                        b_vac[coord] += 1
                    if (dB-10 < d) and (d < dB+10):
                        b_occ[coord] += 1
                        RGBacc[coord*3+0] += rgb[ind*3+0]
                        RGBacc[coord*3+1] += rgb[ind*3+1]
                        RGBacc[coord*3+2] += rgb[ind*3+2]

    cdef float recip
    for i in range(GX*GY*GZ):
        recip = 1./(b_occ[i]+1)
        RGB[i*3+0] = <np.uint8_t> (RGBacc[i*3+0]*recip)
        RGB[i*3+1] = <np.uint8_t> (RGBacc[i*3+1]*recip)
        RGB[i*3+2] = <np.uint8_t> (RGBacc[i*3+2]*recip)


def stencil_carve(np.ndarray[np.float32_t, ndim=2, mode='c'] depthB,
                  np.ndarray[np.uint16_t, ndim=2, mode='c'] depth,
                  np.ndarray[np.uint8_t, ndim=3, mode='c'] coords,
                  int GX, int GY, int GZ,
                  np.ndarray[np.int32_t, ndim=4, mode='c'] RGBacc,
                  np.ndarray[np.uint8_t, ndim=3, mode='c'] rgb,                  
                  np.ndarray[np.uint8_t, ndim=4, mode='c'] RGB,
                  np.ndarray[np.float32_t, ndim=3, mode='c'] b_total,
                  np.ndarray[np.float32_t, ndim=3, mode='c'] b_occ,
                  np.ndarray[np.float32_t, ndim=3, mode='c'] b_vac,
                  int T, int L, int B, int R):

    stencil_carve_(<np.float32_t *> depthB.data,
                   <np.uint16_t *> depth.data,
                   <np.uint8_t *> coords.data,
                   GX, GY, GZ,
                   <np.int32_t *> RGBacc.data,
                   <np.uint8_t *> rgb.data,
                   <np.uint8_t *> RGB.data,
                   <np.float32_t *> b_total.data,
                   <np.float32_t *> b_occ.data,
                   <np.float32_t *> b_vac.data,
                   T, L, B, R)                   
