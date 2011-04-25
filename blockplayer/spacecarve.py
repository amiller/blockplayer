import numpy as np
import config


def depth_inds(modelmat, X, Y, Z):
    gridmin, gridmax = config.bounds

    bg = config.bg
    mat = np.linalg.inv(np.dot(np.dot(modelmat,
                               bg['Ktable']),
                               bg['KK']))
    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
    w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]

    return x/w, y/w, z/w


def carve(depth, modelmat):
    """
    Sample the depth image at the center point of each voxel. Mark as 'vacant
    all the voxels that we can see right through.

    Output:
        vac: vacancy grid [gridmin:gridmax]
        occ: unused
    """
    global x, y, d

    gridmin, gridmax = config.bounds

    # Consider the center points of each candidate voxel
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
    d = 1000./scipy.ndimage.map_coordinates(depth_, (y,x), order=0,
                                            prefilter=False,
                                            cval=-np.inf)

    # Project to metric depth
    np.seterr(invalid='ignore')
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

    global vac
    vac = (d>0)&(dmet<drefmet-length)
    return vac
