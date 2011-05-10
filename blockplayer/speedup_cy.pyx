cimport numpy as np
import numpy as np
import cython

@cython.cdivision(True)
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
    cdef int i, j
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


@cython.cdivision(True)
def stencil_finish(np.ndarray[np.float32_t, ndim=2, mode='c'] depth_,
                   np.ndarray[np.uint8_t, ndim=3, mode='c'] coords_,
                   np.ndarray[np.float32_t, ndim=2, mode='c'] readpixels_,
                   np.ndarray[np.uint8_t, ndim=3, mode='c'] readpixelsA_,
                   int T, int L, int B, int R):
    cdef int i = 0
    cdef int indd = 640*T + L
    cdef int y, x
    cdef np.float32_t *depth = <np.float32_t *>depth_.data
    cdef np.float32_t *readpixels = <np.float32_t *>readpixels_.data
    cdef np.int32_t *coords = <np.int32_t *>coords_.data
    cdef np.int32_t *readpixelsA = <np.int32_t *>readpixelsA_.data
    for y in range(0, B-T):
       for x in range(0, R-L):
           depth[indd] = 100.0/(1.000001 - readpixels[i])
           coords[indd] = readpixelsA[i]
           indd += 1
           i += 1
       indd += 640-(R-L)


def grid_vertices(grid):
    pass


def fix_colors(np.ndarray[np.uint8_t, ndim=4, mode='c'] hsv_input_,
               np.ndarray[np.int32_t, ndim=1, mode='c'] color_targets_):
    cdef int len = hsv_input_.shape[0] * hsv_input_.shape[1] * hsv_input_.shape[2]
    cdef int clen = color_targets_.shape[0]
    cdef int i, c
    cdef np.uint8_t *hsv_input = <np.uint8_t *>hsv_input_.data
    cdef np.int32_t *color_targets = <np.int32_t *>color_targets_.data
    cdef int besterr, bestcolor, err
    for i in range(len):
        besterr = 10000
        for c in range(clen):
            err = hsv_input[i*3+0] - color_targets[c]
            err = -err if err < 0 else err
            if err < besterr:
                besterr = err
                bestcolor = color_targets[c]
        hsv_input[i*3+0] = bestcolor
        hsv_input[i*3+1] = 255
        hsv_input[i*3+2] = 255
            

@cython.cdivision(True)
cdef depth_inds(np.float32_t *m,
                np.float32_t *KK,
                np.uint16_t *depth,
                np.int32_t *gridmin,
                np.int32_t *gridmax,
                np.uint8_t *vac,
                float LW, float LH, float length):

    cdef int xmin = gridmin[0]
    cdef int ymin = gridmin[1]
    cdef int zmin = gridmin[2]
    cdef int xmax = gridmax[0]
    cdef int ymax = gridmax[1]
    cdef int zmax = gridmax[2]

    cdef float X, Y, Z
    cdef int iX, iY, iZ
    cdef float x, y, z, w
    
    cdef int ix, iy
    cdef float d, dref, drefmet, dmet
    cdef int i = 0
    
    for iX in range(xmin, xmax):
        X = (iX+<float>0.5) * LW
        for iY in range(ymin, ymax):
            Y = (iY+<float>0.5) * LH
            for iZ in range(zmin, zmax):
                Z = (iZ+<float>0.5) * LW

                x = X*m[ 0] + Y*m[ 1] + Z*m[ 2] + m[ 3]
                y = X*m[ 4] + Y*m[ 5] + Z*m[ 6] + m[ 7]
                z = X*m[ 8] + Y*m[ 9] + Z*m[10] + m[11]
                w = X*m[12] + Y*m[13] + Z*m[14] + m[15]

                w = 1/w if not w == 0 else 0
                x = x * w
                y = y * w
                z = z * w

                dref = z
                
                ix = <int> x
                iy = <int> y

                ix = 0 if ix < 0 else ix
                iy = 0 if iy < 0 else iy
                ix = 639 if ix > 639 else ix
                iy = 479 if iy > 479 else iy

                d = depth[iy*640+ix]
                d = 1000. / d if not d == 0 else 0

                z = x*KK[ 8] + y*KK[ 9] + d*KK[10] + KK[11]
                w = x*KK[12] + y*KK[13] + d*KK[14] + KK[15]
                dmet = z/w if not w == 0 else 10000
                
                z = x*KK[ 8] + y*KK[ 9] + dref*KK[10] + KK[11]
                w = x*KK[12] + y*KK[13] + dref*KK[14] + KK[15]
                drefmet = z/w if not w == 0 else 0

                vac[i] = (d>0) and (dmet < (drefmet - length))

                i += 1


def spacecarve(np.ndarray[np.uint16_t, ndim=2, mode='c'] depth,
               np.ndarray[np.uint8_t, ndim=3, mode='c'] vac,
               np.ndarray[np.float32_t, ndim=2, mode='c'] modelmat,
               np.ndarray[np.float32_t, ndim=2, mode='c'] KK,
               np.ndarray[np.int32_t, ndim=1] gridmin,
               np.ndarray[np.int32_t, ndim=1] gridmax,
               float LW, float LH, float length):

    depth_inds(<np.float32_t *> modelmat.data,
               <np.float32_t *> KK.data,
               <np.uint16_t *> depth.data,
               <np.int32_t *> gridmin.data,
               <np.int32_t *> gridmax.data,
               <np.uint8_t *> vac.data, LW, LH, length)

def occvac(np.ndarray[np.int8_t, ndim=3, mode='c'] gridinds_,
           np.ndarray[np.uint8_t, ndim=3, mode='c'] occ_,
           np.ndarray[np.uint8_t, ndim=3, mode='c'] vac_,
           np.ndarray[np.int32_t, ndim=1, mode='c'] gridmin,
           np.ndarray[np.int32_t, ndim=1, mode='c'] gridmax):

    cdef int i
    cdef int length = gridinds_.shape[0]
    cdef int wx = gridmax[0]-gridmin[0]
    cdef int wy = gridmax[1]-gridmin[1]
    cdef int wz = gridmax[2]-gridmin[2]
    cdef int wywz = wy * wz
    cdef int x, y, z

    cdef np.int8_t *gridinds = <np.int8_t *> gridinds_.data
    cdef np.uint8_t *occ = <np.uint8_t *> occ_.data
    cdef np.uint8_t *vac = <np.uint8_t *> vac_.data
    
    for i in range(length):
        if gridinds[8*i+0+3] != 0:
            x = gridinds[8*i+0+0]
            y = gridinds[8*i+0+1]
            z = gridinds[8*i+0+2]
            if occ[x*wywz + y*wz + z] < 35:
                #assert x >= 0 and z >= 0, 'vac >0'
                #assert x < wx and z < wz and y < wy, 'vac < max'
                occ[x*wywz + y*wz + z] += 1

        if gridinds[8*i+4+3] != 0:
            x = gridinds[8*i+4+0]
            y = gridinds[8*i+4+1]
            z = gridinds[8*i+4+2]
            #assert x >= 0 and z >= 0, 'vac >0'
            #assert x < wx and z < wz and y < wy, 'vac < max'
            if vac[x*wywz + y*wz + z] < 35:
                vac[x*wywz + y*wz + z] += 1

    for i in range(wx*wy*wz):
        occ[i] = occ[i] > 30
        vac[i] = vac[i] > 30


cdef diff_coord(int xa, int ya, int za, int ra,
                int x, int y, int z, int r):
    x,y,z = xa-x,ya-y,za-z
    cdef int i
    for i in range(ra):
        x,z = z,-x
    return x,y,z,(r-ra)%4


def match_features(A, B, sq):
    ymax = max([y for _,y,_,_ in A])
    yA = [[_ for _ in A if _[1] == y] for y in range(ymax)]
    yB = [[_ for _ in B if _[1] == y] for y in range(ymax)]
    d = {}
    for y in range(ymax):
        for a in yA[y]:
            for b in yB[y]:
                m = diff_coord(a[0], a[1], a[2], a[3],
                               b[0], b[1], b[2], b[3])
                d.setdefault(m,0)
                d[m] += 1
    return d


def find_features(np.ndarray[np.uint8_t, ndim=3, mode='c'] A):
    cdef int r, i, x, y, z
    cdef int WX = A.shape[0]
    cdef int WY = A.shape[1]
    cdef int WZ = A.shape[2]
    cdef int WYWZ = WY*WZ
    cdef int a0, a1, a2
    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] features
    features = np.empty((WX*WY*WZ, 4), 'i')
    cdef np.int32_t *_features = <np.int32_t *>features.data
    cdef int flen = 0
    cdef np.ndarray[np.uint8_t, ndim=3, mode='c'] Ary
    cdef np.uint8_t *ary
    for r in range(4):
        Ary = np.ascontiguousarray(np.swapaxes(np.rot90(\
            np.swapaxes(A, 1, 2), r), 1, 2))
        ary = <np.uint8_t *> Ary.data
        for x in range(WX-1):
            for y in range(WY):
                i = x*WYWZ + y*WZ
                for z in range(WZ-1):
                    a0 = ary[i+z];
                    a1 = ary[i+z+1];
                    a2 = ary[i+z+WYWZ];
                    if a0 and a1==0 and a2==0:
                        _features[4*flen+0] = x
                        _features[4*flen+1] = y
                        _features[4*flen+2] = z
                        _features[4*flen+3] = r
                        flen += 1
    return features[:flen,:]


def grid_error(np.ndarray[np.uint8_t, ndim=3, mode='c'] occA,
               np.ndarray[np.uint8_t, ndim=3, mode='c'] vacA,
               np.ndarray[np.uint8_t, ndim=3, mode='c'] occB,
               np.ndarray[np.uint8_t, ndim=3, mode='c'] vacB,
               int bx, int bz, int term):

    cdef int offset = -(bz + bx*occA.shape[2]*occA.shape[1])
    cdef int lower = 0 if offset>0 else -offset
    cdef int upper = occA.shape[0]*occA.shape[1]*occA.shape[2]
    upper = upper if offset<0 else upper-offset
    cdef float total = 0
    cdef int term_ = int((term-100)*2 * 1.5)
    cdef int total_ = 0
    cdef int i
    cdef int oA, oB, vA, vB
    cdef np.uint8_t *oA_ = <np.uint8_t *> occA.data
    cdef np.uint8_t *vA_ = <np.uint8_t *> vacA.data
    cdef np.uint8_t *oB_ = <np.uint8_t *> occB.data
    cdef np.uint8_t *vB_ = <np.uint8_t *> vacB.data    
    for i in range(lower, upper):
        oA = oA_[i]
        oB = oB_[i+offset]
        if not oA and not oB:
            continue
        vA = vA_[i]
        vB = vB_[i+offset]
        if vB and oA: total_ += 2
        if oB and vA: total_ += 2
        if oB and oA: total_ -= 1
        if total_ > term_:
            break

    total = total_/float(2.) + 100;
    return total

        #print lower, upper

def inrange(np.ndarray[np.uint16_t, ndim=2, mode='c'] depth_,
            np.ndarray[np.uint8_t, ndim=2, mode='c'] mm_,
            np.ndarray[np.uint16_t, ndim=2, mode='c'] bgHi_,
            np.ndarray[np.uint16_t, ndim=2, mode='c'] bgLo_,
            int length):
    cdef int i
    cdef np.uint16_t *depth = <np.uint16_t *> depth_.data
    cdef np.uint8_t *mm = <np.uint8_t *> mm_.data
    cdef np.uint16_t *bgHi = <np.uint16_t *> bgHi_.data    
    cdef np.uint16_t *bgLo = <np.uint16_t *> bgLo_.data

    for i in range(length):
        mm[i] = depth[i] > bgLo[i] and depth[i] < bgHi[i]

        
grid_q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
         [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
         [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
         [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
         [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
         [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]
grid_q = np.array(grid_q, 'i')
grid_normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2]))
               for qz in grid_q]

    
