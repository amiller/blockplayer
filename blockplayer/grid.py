import numpy as np
import preprocess
import opencl
import lattice
import config
import os
import dataset

GRIDRAD = 18
bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,9,GRIDRAD)


from ctypes import POINTER as PTR, c_byte, c_size_t, c_float
speedup_ctypes = np.ctypeslib.load_library('speedup_ctypes.so',
                                           os.path.dirname(__file__))
speedup_ctypes.histogram.argtypes = [PTR(c_byte), PTR(c_float), PTR(c_float),
                                     c_size_t, c_size_t, c_size_t, c_size_t]


def grid2str():
    global vote_grid
    m = np.choose(vote_grid>30, (' ', '*'))
    layers = [[''.join(_) for _ in m[:,i,:]]
              for i in range(m.shape[1])]
    import pprint
    return pprint.pformat(layers)


def load_gt():
    # This is probably a hack! I don't know where to put the groundtruth
    with open(os.path.join(dataset.current_path, 'config/gt.txt'),'r') as f:
        s = f.read()
    g = np.array(eval(s))
    return g


def grid_vertices(grid,factor=1):
    """
    Given a boolean voxel grid, produce a list of vertices and indices
    for drawing quads or line strips in opengl
    """
    q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
         [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
         [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
         [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
         [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
         [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]

    normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2]))
              for qz in q]

    blocks = np.array(grid.nonzero()).transpose().reshape(-1,1,3)
    q = np.array(q).reshape(1,-1,3)
    vertices = (q + blocks).reshape(-1,3)
    normals = np.tile(normal, (len(blocks),4)).reshape(-1,3)*factor
    line_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
    quad_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,2,3]

    return vertices, normals, line_inds, quad_inds


def initialize():
    b_width = [bounds[1][i]-bounds[0][i] for i in range(3)]
    global vote_grid, carve_grid, keyvote_grid, keycarve_grid
    keyvote_grid = np.zeros(b_width)
    keycarve_grid = np.zeros(b_width)
    vote_grid = np.zeros(b_width)
    carve_grid = np.zeros(b_width)

    global shadow_blocks, solid_blocks, wire_blocks
    shadow_blocks = None
    solid_blocks = None
    wire_blocks = None

if not 'vote_grid' in globals(): initialize()


def refresh():
    global solid_blocks, shadow_blocks, wire_blocks
    solid_blocks = grid_vertices((carve_grid<30)&(vote_grid>30))
    #shadow_blocks = grid_vertices((carve_grid>=30)&(vote_grid>30))
    #shadow_blocks = grid_vertices((carve_grid>=30))
    #wire_blocks = grid_vertices((carve_grid>10))


def drift_correction(new_votes, new_carve):
    """ Using the current values for vote_grid and carve_grid,
    and the new histograms
    generated from the newest frame, find the translation between old and new 
    (in a 3x3 neighborhood, only considering jumps of 1 block) that minimizes 
    an error function between them.
    """
    def error(t):
        """ 
        t: x,y
        The error function is the number of error blocks that fall in a carved
        region.
        """
        nv = np.roll(new_votes, t[0], 0)
        nv = np.roll(nv, t[2], 2)
        nc = np.roll(new_carve, t[0], 0)
        nc = np.roll(nc, t[2], 2)
        return np.sum(np.minimum(nc,vote_grid) + np.minimum(nv,carve_grid) -
                      np.minimum(nv,vote_grid)/4)

    t = [(x,y,z) for x in [0,-1,1,-2,2] for y in [0] for z in [0,-1,1,-2,2]]
    vals = [error(_) for _ in t]
    #print vals
    return t[np.argmin(vals)], np.min(vals)


def depth_inds(modelmat, X, Y, Z):    
    gridmin = bounds[0]
    gridmax = bounds[1]

    bg = config.bg
    mat = np.linalg.inv(np.dot(np.dot(modelmat,
                               bg['Ktable']),
                        bg['KK']))
    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
    w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]

    return x/w, y/w, z/w


def depth_sample(modelmat, depth):
    gridmin = np.zeros((4,),'f')
    gridmax = np.zeros((4,),'f')
    gridmin[:3] = bounds[0]
    gridmax[:3] = bounds[1]

    # Only keep the X,Y,Z points we haven't 'carved out' already
    X,Y,Z = np.mgrid[gridmin[0]:gridmax[0],
                     gridmin[1]:gridmax[1],
                     gridmin[2]:gridmax[2]]+0.5

    X *= config.LW
    Y *= config.LH
    Z *= config.LW

    # Find the reference depth for each voxel, and the sampled depth
    x,y,dref = depth_inds(modelmat, X,Y,Z)
    depth_ = depth.astype('f')
    #depth_[depth==2047] = -np.inf
    import scipy.ndimage
    d = scipy.ndimage.map_coordinates(depth_, (y,x), order=0,
                                      prefilter=False,
                                      cval=-np.inf)

    np.seterr(invalid='ignore')
    # Project to metric depth
    mat = config.bg['KK']
    z = x*mat[2,0] + y*mat[2,1] + d*mat[2,2] + mat[2,3]
    w = x*mat[3,0] + y*mat[3,1] + d*mat[3,2] + mat[3,3]
    dmet = z/w

    z = x*mat[2,0] + y*mat[2,1] + dref*mat[2,2] + mat[2,3]
    w = x*mat[3,0] + y*mat[3,1] + dref*mat[3,2] + mat[3,3]
    drefmet = z/w

    length = np.sqrt((config.LW**2+
                      config.LH**2+
                      config.LW**2))*0.5
    np.seterr(invalid='warn')
    return x,y,d,dref, (d>0)&(dmet<drefmet-length)


def add_votes(xfix, zfix, depth, use_opencl):    
    gridmin = np.zeros((4,),'f')
    gridmax = np.zeros((4,),'f')
    gridmin[:3] = bounds[0]
    gridmax[:3] = bounds[1]

    global gridinds, inds, grid
    if not use_opencl:
        global X,Y,Z, XYZ
        X,Y,Z,face = np.rollaxis(opencl.get_modelxyz(),1)
        XYZ = np.array((X,Y,Z)).transpose()
        fix = np.array((xfix,0,zfix))
        cxyz = np.frombuffer(np.array(face).data,
                             dtype='i1').reshape(-1,4)[:,:3]
        global cx,cy,cz
        cx,cy,cz = np.rollaxis(cxyz,1)
        f1 = cxyz*0.5

        mod = np.array([config.LW, config.LH, config.LW])
        gi = np.floor(-gridmin[:3] + (XYZ-fix)/mod + f1)
        gi = np.array((gi, gi - cxyz))
        gi = np.rollaxis(gi, 1)

        def get_gridinds():
            (L,T),(R,B) = opencl.rect
            length = opencl.length
            return (gridinds[:length,:,:].reshape(T-B,R-L,2,3),
                    gridinds[length:,:,:].reshape(T-B,R-L,2,3))

        gridinds = gi
        grid = get_gridinds()
        inds = gridinds[np.any(cxyz!=0,1),:,:]

    else:
        opencl.compute_gridinds(xfix,zfix,
                                config.LW, config.LH,
                                gridmin, gridmax)
        gridinds = opencl.get_gridinds()
        inds = gridinds[gridinds[:,0,3]!=0,:,:3]    

    if len(inds) == 0: return

    global occH, vacH
    global carve_grid,vote_grid, bins
    bins = [np.arange(0,bounds[1][i]-bounds[0][i]+1) for i in range(3)]
    occH,_ = np.histogramdd(inds[:,0,:], bins)
    vacH,_ = np.histogramdd(inds[:,1,:], bins)

    # Add in the carved pixels
    _,_,_,_,carve = depth_sample(lattice.modelmat,depth)
    
    #vacH *= 0
    vacH += 60*carve
    occH[vacH>30] = 0
    # Correct for drift
    (bx,_,bz),err = drift_correction(occH, vacH)
    if (bx,bz) != (0,0):
        lattice.modelmat[0,3] += bx*config.LW
        lattice.modelmat[2,3] += bz*config.LW
        occH = np.roll(np.roll(occH, bx, 0), bz, 2)
        vacH = np.roll(np.roll(vacH, bx, 0), bz, 2)
        print "drift detected:", bx,bz

    wx,wy,wz = [bounds[1][i]-bounds[0][i] for i in range(3)]

    # occH = np.zeros((wx,wy,wz),'f')
    # vacH = np.zeros((wx,wy,wz),'f')
    # speedup_ctypes.histogram(gridinds.ctypes.data_as(PTR(c_byte)), 
    #                          occH.ctypes.data_as(PTR(c_float)),
    #                          vacH.ctypes.data_as(PTR(c_float)), 
    #                          np.int32(gridinds.shape[0]), 
    #                          np.int32(wx), np.int32(wy), np.int32(wz))

    if 0:
        sums = np.zeros((3,3),'f')
        speedup_ctypes.histogram_error(vote_grid.ctypes.data,
                                       carve_grid.ctypes.data, 
                                       occH.ctypes.data, vacH.ctypes.data, 
                                       sums.ctypes.data_as(PTR(c_float)),
                                       wx, wy, wz);

    # Only update the persistent model if we satisfy a quality condition
    if lattice.dmx >= 0.8 and lattice.dmy >= 0.8 and \
       lattice.countx > 200 and lattice.county >= 200:

        carve_grid[occH>60] = 0
        carve_grid = np.maximum(vacH, carve_grid)

        vote_grid = np.maximum(occH, vote_grid)
        vote_grid[carve_grid>30] = 0

    # Recenter the voxel grid
    if np.any(vote_grid>30):
        mean = -(gridmin[:3] + \
                np.floor(np.mean(np.nonzero(vote_grid>30),1))).astype('i')
        lattice.modelmat[:3,3] += [mean[0]*config.LW, 0, mean[2]*config.LW]
        carve_grid = np.roll(np.roll(carve_grid, mean[0], 0), mean[2], 2)
        vote_grid = np.roll(np.roll(vote_grid, mean[0], 0), mean[2], 2)

    refresh()
