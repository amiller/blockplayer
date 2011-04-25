import numpy as np
import pylab
import opennpy


from blockplayer import calibkinect
from blockplayer import config
from blockplayer import preprocess

if config.ALIGNED:
    opennpy.align_depth_to_rgb()

from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='demo_calib', size=(640,480))

mat = np.linalg.inv(calibkinect.aligned_projection())
mat = np.ascontiguousarray(mat)

PREPROCESS = True


def once():
    global depth
    global rgb
    opennpy.sync_update()
    depth,_ = opennpy.sync_get_depth()
    rgb,_ = opennpy.sync_get_video()

    global x,y,z,xyz

    if PREPROCESS:
        global mask, rect
        mask, rect = preprocess.threshold_and_mask(depth, config.bg)

        (l,t),(r,b) = rect
        depth_ = np.ascontiguousarray(depth[t:b,l:r])
        v,u = np.mgrid[t:b,l:r].astype('f')
        x,y,z = calibkinect.convertOpenNI2Real(depth_,u,v)
        x=x[mask[t:b,l:r]]
        y=y[mask[t:b,l:r]]
        z=z[mask[t:b,l:r]]

        rgb_ = rgb[t:b,l:r,:][mask[t:b,l:r],:]
    else:
        x,y,z = calibkinect.convertOpenNI2Real(depth)
        #global X,Y,Z
        #X,Y,Z = calibkinect.convertReal2OpenNI(x,y,z)

        x,y,z = map(lambda _:_.flatten(), (x,y,z))
        rgb_ = rgb.reshape(-1,3)

    #x = x[depth>0]
    #y = y[depth>0]
    #z = z[depth>0]
    xyz_ = np.ascontiguousarray(np.dstack((x,y,z)).reshape(-1,3))
    #xyz_ = xyz - xyz.mean(0)
    #xyz_ /= np.std(xyz_)

    rgba = np.empty((xyz_.shape[0],4),'f')
    rgba[:,3] = 1
    rgba[:,:3] = rgb_.astype('f')/256.0

    window.lookat = np.array([0,0,0])
    window.update_points(xyz_,rgba)
    window.clearcolor = [0,0,0,0]
    window.Refresh()
    pylab.waitforbuttonpress(0.005)


def go():
    config.load('data/newest_calibration')
    try:
        while 1:
            once()
    except KeyboardInterrupt:
        pass


if 'window' in globals():
    window.Refresh()


if __name__ == '__main__':
    pass
    #go()
