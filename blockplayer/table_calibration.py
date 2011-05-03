from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import normals
import calibkinect
import config
import opennpy
from pylab import *


newest_folder = "data/newest_calibration"


def from_rect(m, rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]


def finish_table_calib():
    # We've already picked the bound points for each image
    global depth
    boundpts = np.load('%s/config/boundpts.npy' % newest_folder)
    depth = np.load('%s/config/depth.npy' % newest_folder)

    config.bg = find_plane(depth, boundpts)
    config.save(newest_folder)


def run_calib():
    close('all')
    """Run the table plane calibration
    """
    global points
    print("Getting an image from the camera")
    opennpy.align_depth_to_rgb()
    [opennpy.sync_update() for i in range(10)]
    depth, _ = opennpy.sync_get_depth()

    fig = figure(1)
    clf()
    points = []

    def pick(event):
        global points
        points.append((event.xdata, event.ydata))
        print('Picked point %d of 4' % (len(points)))

    imshow(1./depth)
    draw()
    fig.canvas.mpl_disconnect('button_press_event')
    fig.canvas.mpl_connect('button_press_event', pick)

    print("Click four points")
    while len(points) < 4:
        waitforbuttonpress(0.001)

    print 'OK'
    np.save('%s/config/boundpts' % (newest_folder), points)
    np.save('%s/config/depth' % (newest_folder), depth)

    finish_table_calib()


def find_plane(depth, boundpts):
    from visuals.camerawindow import CameraWindow
    global window
    if not 'window' in globals():
        window = CameraWindow()

    # Build a mask of the image inside the convex points clicked
    u,v = uv = np.mgrid[:480,:640][::-1]
    mask = np.ones((480,640),bool)
    for (x,y),(dx,dy) in zip(boundpts, boundpts - np.roll(boundpts,1,0)):
        mask &= ((uv[0]-x)*dy - (uv[1]-y)*dx)<0

    # Borrow the initialization from calibkinect
    KK = np.linalg.inv(calibkinect.projection()).astype('f')
    KK = np.ascontiguousarray(KK)

    # Find the average plane going through here
    global n,w
    n,w = normals.normals_c(depth)
    maskw = mask & (w>0)
    abc = n[maskw].mean(0)
    abc /= np.sqrt(np.dot(abc,abc))
    a,b,c = abc
    x,y,z = [_[maskw].mean() for _ in
             calibkinect.convertOpenNI2Real_numpy(depth)]
    d = -(a*x+b*y+c*z)
    tableplane = np.array([a,b,c,d])
    #tablemean = np.array([x,y,z])

    # Backproject the table plane into the image using inverse transpose
    global tb0
    tb0 = np.dot(KK.transpose(), tableplane)
    tb0[2] = tb0[2]

    # Build a matrix projecting sensor points to an system with
    # the origin on the table, and Y pointing up from the table
    # NOTE: the default orientation is offset by 45 degrees from the camera's
    # natural direction.
    v1 = np.array([a,b,c]); v1 /= np.sqrt(np.dot(v1,v1))
    v0 = np.cross(v1, [-1,0,1]); v0 /= np.sqrt(np.dot(v0,v0))
    v2 = np.cross(v0,v1);
    Ktable = np.eye(4)
    Ktable[:3,:3] = np.vstack((v0,v1,v2)).transpose()
    Ktable[:3,3] = [x,y,z]
    Ktable = np.linalg.inv(Ktable).astype('f')

    KtableKK = np.dot(Ktable, KK).astype('f')

    global boundptsM
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
    glMultMatrixf(np.linalg.inv(KtableKK).transpose())

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

    gf = glGetIntegerv(GL_FRONT_FACE)
    glFrontFace(GL_CCW)
    draw()
    openglbgHi = glReadPixels(0, 0, 640, 480,
                              GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
    glFrontFace(GL_CW)
    draw()
    openglbgLo = glReadPixels(0, 0, 640, 480,
                              GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
    glFrontFace(gf)

    global hi,lo
    openglbgHi = 1000./(openglbgHi*10)
    openglbgLo = 1000./(openglbgLo*10)
    lo = np.array(openglbgLo)
    hi = np.array(openglbgHi)

    #openglbgLo[openglbgLo>=2047] = 0
    #openglbgHi[np.isnan(openglbgHi)] = 0
    openglbgLo[openglbgHi==openglbgLo] = 0

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(2, [rb,rbc]);
    glDeleteFramebuffers(1, [fbo]);

    background = np.array(depth)
    background[~mask] = 0
    background = np.maximum(background,openglbgHi)
    #backgroundM = normals.project(background)

    openglbgLo = openglbgLo.astype(np.uint16)
    background = background.astype(np.uint16)
    background[background>=5] -= 5
    #openglbgLo += 5

    return dict(
        bgLo=openglbgLo,
        bgHi=background,
        boundpts=boundpts,
        boundptsM=boundptsM,
        KK=KK,
        Ktable=Ktable)
