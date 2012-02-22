import numpy as np
import config

# Build a look up table for quick conversion between 
# OpenNI values (millimeters) and projective values
# (1/meters)
lut = np.arange(5000).astype('f')
lut[1:] = 1000./lut[1:]
lut[0] = -1e8


def recip_depth_openni(depth):
    import scipy.weave
    assert depth.dtype == np.uint16
    output = np.empty(depth.shape,'f')
    N = np.prod(depth.shape)
    code = """
    int i;
    for (i = 0; i < (int)N; i++) {
      output[i] = lut[depth[i]];
    }
    """
    scipy.weave.inline(code, ['output','depth','lut','N'])
    return output


def projection():
    """
    Camera matrix for the aligned 
    """
    fx = 528.0
    fy = 528.0
    cx = 320.0
    cy = 267.0

    mat = np.array([[fx, 0, -cx, 0],
                    [0, -fy, -cy, 0],
                    [0, 0, 0, 1],
                    [0, 0, -1., 0]]).astype('f')
    return np.ascontiguousarray(mat)


full_vu = np.mgrid[:480,:640].astype('f')


def convertOpenNI2Real_weave(depth, u=None, v=None,
                       mat=np.ascontiguousarray(
                                np.linalg.inv(projection()))):
    assert mat.dtype == np.float32
    assert mat.dtype == np.float32
    assert mat.shape == (4,4)
    assert mat.flags['C_CONTIGUOUS']
    assert depth.dtype == np.uint16

    if u is None or v is None: v,u = full_vu
    assert depth.shape == u.shape == v.shape

    X,Y = u,v
    x,y,z = [np.empty(depth.shape, 'f') for i in range(3)]

    N = np.prod(depth.shape)
    code = """
    int i;
    for (i = 0; i < (int)N; i++) {
      float Z = lut[depth[i]];
      float x_ = X[i]*mat[0] + Y[i]*mat[1] + Z*mat[2] + mat[3];
      float y_ = X[i]*mat[4] + Y[i]*mat[5] + Z*mat[6] + mat[7];
      float z_ = X[i]*mat[8] + Y[i]*mat[9] + Z*mat[10] + mat[11];
      float w = X[i]*mat[12] + Y[i]*mat[13] + Z*mat[14] + mat[15];
      w = 1/w;
      x[i] = x_*w;
      y[i] = y_*w;
      z[i] = z_*w;
    }
    """
    import scipy.weave
    scipy.weave.inline(code, ['X','Y','depth','x','y','z','N','mat','lut'])
    return x,y,z


def convertOpenNI2Real_numpy(depth, u=None, v=None,
                       mat=np.linalg.inv(projection())):

    if u is None or v is None: v,u = full_vu

    X,Y,Z = u,v, recip_depth_openni(depth)
    x,y,z = [np.empty(depth.shape, 'f') for i in range(3)]

    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
    w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
    w = 1/w
    return x*w, y*w, z*w


def convertReal2OpenNI(X, Y, Z, mat=projection()):

    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
    w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
    _z = 1000.*w/z
    w = 1/w
    return x*w, y*w, _z


from calibkinect_cy import convertOpenNI2Real
