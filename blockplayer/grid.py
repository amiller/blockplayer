import numpy as np
import preprocess
import opencl
import lattice
import carve
import config
import os

GRIDRAD = 8
bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,8,GRIDRAD)


from ctypes import POINTER as PTR, c_byte, c_size_t, c_float
speedup_ctypes = np.ctypeslib.load_library('speedup_ctypes.so',
                                           os.path.dirname(__file__))
speedup_ctypes.histogram.argtypes = [PTR(c_byte), PTR(c_float), PTR(c_float),
                                     c_size_t, c_size_t, c_size_t, c_size_t]


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
    solid_blocks = carve.grid_vertices((vote_grid>30))
    shadow_blocks = carve.grid_vertices((carve_grid>10)&(vote_grid>30))
    wire_blocks = carve.grid_vertices((carve_grid>10))


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
        return np.sum(np.minimum(nc,vote_grid) + np.minimum(nv,carve_grid))

    t = [(x,y,z) for x in [0,-1,1] for y in [0] for z in [0,-1,1]]
    vals = [error(_) for _ in t]
    #print vals
    return t[np.argmin(vals)]

  
def add_votes_opencl(xfix,zfix):
    gridmin = np.zeros((4,),'f')
    gridmax = np.zeros((4,),'f')
    gridmin[:3] = bounds[0]
    gridmax[:3] = bounds[1]

    global occH, vacH
    global carve_grid,vote_grid

    opencl.compute_gridinds(xfix,zfix, config.LW, config.LH, gridmin, gridmax)
    global gridinds
    gridinds = opencl.get_gridinds()

    inds = gridinds[gridinds[:,0,3]!=0,:,:3]    
    if len(inds) == 0: return
  
    bins = [np.arange(0,bounds[1][i]-bounds[0][i]+1)-0.5 for i in range(3)]
    occH,_ = np.histogramdd(inds[:,0,:], bins)
    vacH,_ = np.histogramdd(inds[:,1,:], bins)
    bx,_,bz = drift_correction(occH, vacH)
    if 0 and (bx,bz) != (0,0):
        lattice.modelmat[0,3] += bx*config.LW
        lattice.modelmat[2,3] += bz*config.LW
        print "drift detected:", bx,bz
        return lattice.modelmat[:3,:4]

    wx,wy,wz = [bounds[1][i]-bounds[0][i] for i in range(3)]

    # occH = np.zeros((wx,wy,wz),'f')
    # vacH = np.zeros((wx,wy,wz),'f')
    # speedup_ctypes.histogram(gridinds.ctypes.data_as(PTR(c_byte)), 
    #                          occH.ctypes.data_as(PTR(c_float)),
    #                          vacH.ctypes.data_as(PTR(c_float)), 
    #                          np.int32(gridinds.shape[0]), 
    #                          np.int32(wx), np.int32(wy), np.int32(wz))
  
    sums = np.zeros((3,3),'f')
  
    speedup_ctypes.histogram_error(vote_grid.ctypes.data,
                                   carve_grid.ctypes.data, 
                                   occH.ctypes.data, vacH.ctypes.data, 
                                   sums.ctypes.data_as(PTR(c_float)),
                                   wx, wy, wz);
                                 
    t = np.argmin(sums)
    ts = [(z,x) for x in (-1,0,1) for z in (-1,0,1)]
    bx, bz = ts[t]
    print sums
  
    if 0 and sums.flatten()[t] < sums.flatten()[4] - 2:
        #vote_grid  = np.roll(np.roll( vote_grid, -bx, 0), -bz, 2)
        #carve_grid = np.roll(np.roll(carve_grid, -bx, 0), -bz, 2)
        #occH = np.roll(np.roll(occH, bx, 0), bz, 2)
        #vacH = np.roll(np.roll(vacH, bx, 0), bz, 2)
        #vg = np.roll(np.roll( vg, bx, 0), bz, 2)
        lattice.modelmat[0,3] += bx*config.LW
        lattice.modelmat[2,3] += bz*config.LW
        print "drift detected:", bx,bz
        return lattice.modelmat[:3,:4]

    if 0:
        carve_grid = np.maximum(vacH,carve_grid)
        vote_grid = np.maximum(occH,vote_grid)
        vote_grid *= (carve_grid<30)

    refresh()
