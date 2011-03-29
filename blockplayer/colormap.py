import numpy as np
import scipy.weave

v = np.arange(0,1,1.0/2048)
v = np.power(v, 3) * 6
t_gamma = (v * 6 * 256).astype('i4')


def color_map(depth):
    assert depth.shape == (480,640)
    assert depth.dtype == np.uint16
    depth_mid = np.empty((depth.shape[0],depth.shape[1],3),'u1')
    code = """
// This comes from the openkinect project,
// https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c

for (int i = 0; i < 640*480; i++) {
    int pval = t_gamma[depth[i]];
    int lb = pval & 0xff;
    switch (pval>>8) {
    case 0:
        depth_mid[3*i+0] = 255;
        depth_mid[3*i+1] = 255-lb;
        depth_mid[3*i+2] = 255-lb;
        break;
    case 1:
        depth_mid[3*i+0] = 255;
        depth_mid[3*i+1] = lb;
        depth_mid[3*i+2] = 0;
        break;
    case 2:
        depth_mid[3*i+0] = 255-lb;
        depth_mid[3*i+1] = 255;
        depth_mid[3*i+2] = 0;
        break;
    case 3:
        depth_mid[3*i+0] = 0;
        depth_mid[3*i+1] = 255;
        depth_mid[3*i+2] = lb;
        break;
    case 4:
        depth_mid[3*i+0] = 0;
        depth_mid[3*i+1] = 255-lb;
        depth_mid[3*i+2] = 255;
        break;
    case 5:
        depth_mid[3*i+0] = 0;
        depth_mid[3*i+1] = 0;
        depth_mid[3*i+2] = 255-lb;
        break;
    default:
        depth_mid[3*i+0] = 0;
        depth_mid[3*i+1] = 0;
        depth_mid[3*i+2] = 0;
        break;
     }
}     """
    scipy.weave.inline(code, ['depth_mid', 't_gamma', 'depth'])
    return depth_mid
