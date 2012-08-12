# Andrew Miller <amiller@cs.ucf.edu> 2011
#
# BlockPlayer - 3D model reconstruction using the Lattice-First algorithm
# See: 
#    "Interactive 3D Model Acquisition and Tracking of Building Block Structures"
#    Andrew Miller, Brandyn White, Emiko Charbonneau, Zach Kanzler, and Joseph J. LaViola Jr.
#    IEEE VR 2012, IEEE TVGC 2012
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import calibkinect
import config
import opennpy
from pylab import *
from rtmodel.camera import kinect_camera
from rtmodel.rangeimage import RangeImage


newest_folder = "data/newest_calibration"


def from_rect(m, rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]


def finish_table_calib(cam=0):
    # We've already picked the bound points for each image
    global depth
    boundpts = np.load('%s/config/boundpts_%d.npy' % (newest_folder, cam))
    depth = np.load('%s/config/depth_%d.npy' % (newest_folder, cam))

    while len(config.cameras) <= cam: config.cameras.append(None)
    config.cameras[cam] = find_plane(depth, boundpts)
    config.save(newest_folder)


def run_calib(cam=0):
    close('all')
    """Run the table plane calibration
    """
    global points, depth
    print("Getting an image from the camera")
    opennpy.align_depth_to_rgb()
    for i in range(10):
        opennpy.sync_update()
        depth, _ = opennpy.sync_get_depth(cam)
        rgb, _ = opennpy.sync_get_video(cam)

    fig = figure(1)
    clf()
    points = []

    def pick(event):
        global points
        points.append((event.xdata, event.ydata))
        print('Picked point %d of 4' % (len(points)))

    #imshow(depth)
    figure(2); imshow(1./depth)
    figure(1); imshow(rgb*np.dstack(3*[1./depth]))
    draw()
    fig.canvas.mpl_disconnect('button_press_event')
    fig.canvas.mpl_connect('button_press_event', pick)

    print("Click four points")
    while len(points) < 4:
        waitforbuttonpress(0.001)

    print 'OK'
    np.save('%s/config/boundpts_%d' % (newest_folder, cam), points)
    np.save('%s/config/depth_%d' % (newest_folder, cam), depth)

    finish_table_calib(cam)


def make_mask(boundpts, size=(640,480)):
    u,v = uv = np.mgrid[:size[1],:size[0]][::-1]
    mask = np.ones(size[::-1],bool)
    for (x,y),(dx,dy) in zip(boundpts, boundpts - np.roll(boundpts,1,0)):
        mask &= ((uv[0]-x)*dy - (uv[1]-y)*dx)<0    
    return mask


def make_boundptsM(boundpts, KtableKK, tb0, tableplane):
    pass

def find_plane(depth, boundpts):
    from wxpy3d.camerawindow import CameraWindow
    global window
    if not 'window' in globals():
        window = CameraWindow()

    # Build a mask of the image inside the convex points clicked
    mask = make_mask(boundpts)

    # Borrow the initialization from calibkinect
    KK = calibkinect.projection()
    KK = np.ascontiguousarray(KK)

    # Find the average plane going through here
    global n,w
    cam = kinect_camera()
    rimg = RangeImage(depth, cam)
    rimg.filter(win=6)
    rimg.compute_normals()
    n,w = rimg.normals, rimg.weights
    maskw = mask & (w>0)
    abc = n[maskw].mean(0)
    abc /= np.sqrt(np.dot(abc,abc))
    a,b,c = abc
    x,y,z = [_[maskw].mean() for _ in calibkinect.convertOpenNI2Real_numpy(depth)]
    d = -(a*x+b*y+c*z)
    tableplane = np.array([a,b,c,d])
    #tablemean = np.array([x,y,z])

    # Backproject the table plane into the image using inverse transpose

    # Build a matrix projecting sensor points to an system with
    # the origin on the table, and Y pointing up from the table
    # NOTE: the default orientation is offset by 45 degrees from the camera's
    # natural direction.
    v1 = np.array([a,b,c]); v1 /= np.sqrt(np.dot(v1,v1))
    v0 = np.cross(v1, [-1,0,1]); v0 /= np.sqrt(np.dot(v0,v0))
    v2 = np.cross(v0,v1);
    Ktable = np.eye(4)
    Ktable[:3,:3] = np.vstack((v0,v1,v2)).T
    Ktable[:3,3] = [x,y,z]
    Ktable = np.linalg.inv(Ktable).astype('f')

    KtableKK = np.dot(Ktable, KK).astype('f')

    #tableplane2 = np.linalg.inv(KtableKK)[1,:]
    #tableplane2

    #within_eps = lambda a, b: np.abs(a-b) < 1e-5
    #assert within_eps(tableplane2, tableplane)

    global boundptsM
    tb0 = np.dot(KK.T, tableplane)
    tb0[2] = tb0[2]

    boundptsM = []
    for (up,vp) in boundpts:
        # First project the image points (u,v) onto to the (imaged) tableplane
        dp = -(tb0[0]*up + tb0[1]*vp + tb0[3])/tb0[2]

        # Then project them into metric space
        xp, yp, zp, wp = np.dot(KtableKK, [up,vp,dp,1])
        xp /= wp ; yp /= wp ; zp /= wp;
        boundptsM += [[xp,yp,zp]]

    # Use OpenGL and framebuffers to draw the table and the walls
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    rb,rbc = glGenRenderbuffers(2)
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rb);
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,640,0,480,0,-10)
    glMultMatrixf(np.linalg.inv(KtableKK).T)

    def draw():
        glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        for x,y,z in boundptsM:
            glVertex(x,y,z,1)
        for (x,y,z),(x_,y_,z_) in zip(boundptsM,np.roll(boundptsM,1,0)):
            glVertex(x ,y ,z ,1)
            glVertex(x_,y_,z_,1)
            glVertex(0,1,0,0)
            glVertex(0,1,0,0)
        glEnd()
        glDisable(GL_CULL_FACE)
        glFinish()

    # Rendering the outside faces gives us the near walls
    gf = glGetIntegerv(GL_FRONT_FACE)
    glFrontFace(GL_CCW)
    draw()
    openglbgHi = glReadPixels(0, 0, 640, 480,
                              GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640)

    # Rendering the interior faces gives us the table plane and the far walls
    glFrontFace(GL_CW)
    draw()
    openglbgLo = glReadPixels(0, 0, 640, 480,
                              GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640)
    glFrontFace(gf)

    # We need to invert the depth measurements to obtain kinect-style range
    # images.
    #    initially, openglbgHi/Lo are in units of 1/m.
    #    afterwards, they are in units of mm.
    global hi,lo
    openglbgHi = 1000./(openglbgHi*10)
    openglbgLo = 1000./(openglbgLo*10)
    lo = np.array(openglbgLo)
    hi = np.array(openglbgHi)

    openglbgLo[openglbgHi==openglbgLo] = 0

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(2, [rb,rbc]);
    glDeleteFramebuffers(1, [fbo]);

    # Some of the observed image points are nearer than the fitted plane.
    # We want fewer false positives in this case, so take the maximum
    # of either the fitted plane or the observed depth
    background = np.array(depth)
    background[~mask] = 0
    background = np.maximum(background,openglbgHi)

    openglbgLo = openglbgLo.astype(np.uint16)
    background = background.astype(np.uint16)
    background[background>=5] -= 5   # Reduce false positives even more.

    return dict(
        bgLo=openglbgLo,
        bgHi=background,
        boundpts=boundpts,
        boundptsM=boundptsM,
        KK=KK,
        Ktable=Ktable)
