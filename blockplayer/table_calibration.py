from pylab import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import freenect
import opencl
import expmap
import normals
import flatrot
import lattice
import preprocess
import calibkinect
import config
from config import LW

#from visuals.camerawindow import CameraWindow
#if not 'window' in globals():
#    window = CameraWindow()


newest_folder = "data/newest_calibration/config"


def from_rect(m, rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]


def finish_table_calib():
    # We've already picked the bound points for each image
    global depthL, depthR
    boundptsL = np.load('%s/boundptsL.npy' % newest_folder)
    boundptsR = np.load('%s/boundptsR.npy' % newest_folder)
    depthL = np.load('%s/depthL.npy' % newest_folder)
    depthR = np.load('%s/depthR.npy' % newest_folder)

    config.bgL = find_plane(depthL, boundptsL)
    config.bgR = find_plane(depthR, boundptsR)
    config.save(newest_folder)


def finish_cube_calib(side='L'):
    """This finds a matrix that puts the Left and Right views in a common
    reference frame, using a calibration cube that's 18 units long.
    """
    config.load(newest_folder)
    opencl.setup_kernel((config.bgL['KK'],config.bgL['Ktable']),
                        (config.bgR['KK'],config.bgR['Ktable']))

    depthL = np.load('%s/cubeL.npy' % newest_folder)
    depthR = np.load('%s/cubeR.npy' % newest_folder)

    global maskL,rectL,maskR,rectR
    (maskL,rectL) = preprocess.threshold_and_mask(depthL, config.bgL)
    (maskR,rectR) = preprocess.threshold_and_mask(depthR, config.bgR)

    opencl.set_rect(rectL,rectR)
    normals.normals_opencl2(from_rect(depthL,rectL).astype('f'),
                            np.array(from_rect(maskL,rectL)), rectL,
                            from_rect(depthR,rectR).astype('f'),
                            np.array(from_rect(maskR,rectR)), rectR,
                            6)

    global nL, wL, nR, wR, mL, mR
    nwL,nwR = opencl.get_normals()
    nL,wL = nwL[:,:,:3],nwL[:,:,3]
    nR,wR = nwR[:,:,:3],nwR[:,:,3]
    #nL,wL = normals.normals_numpy(depthL.astype('f'),rectL)
    #nR,wR = normals.normals_numpy(depthR.astype('f'),rectR)
    axesL = flatrot.flatrot_numpy(nL,wL)
    axesR = flatrot.flatrot_numpy(nR,wR)

    mL,mR = np.eye(4,dtype='f'), np.eye(4,dtype='f')
    mL[:3,:3] = expmap.axis2rot(axesL)
    mR[:3,:3] = expmap.axis2rot(axesR)

    # Tinker with the rotation here
    mR[:3,:3] = np.dot(expmap.axis2rot(np.array([0,np.pi/2,0])),mR[:3,:3])

    # Find the translation component using the lattice
    mL = lattice.lattice2(nL,wL,depthL,mL,
                          np.dot(config.bgL['Ktable'],config.bgL['KK']),
                          rectL,init_t=True)

    X,_,Z = lattice.XYZ
    cx,_,cz = lattice.cXYZ
    minX = np.round(np.median(X[cx>0])/LW)*LW
    minZ = np.round(np.median(Z[cz>0])/LW)*LW
    mL[0,3] -= minX
    mL[2,3] -= minZ

    mR = lattice.lattice2(nR,wR,depthR,mR,
                          np.dot(config.bgR['Ktable'],config.bgR['KK']),
                          rectR,init_t=True)
    X,_,Z = lattice.XYZ
    cx,_,cz = lattice.cXYZ
    minX = np.round(np.median(X[cx>0])/LW)*LW
    minZ = np.round(np.median(Z[cz>0])/LW)*LW
    mR[0,3] -= minX
    mR[2,3] -= minZ
    mR[2,3] -= 16*LW

    if side=='L':
        lattice.lattice2(nL,wL,depthL,mL,
                         np.dot(config.bgL['Ktable'],config.bgL['KK']),
                         rectL)
    else:
        lattice.lattice2(nR,wR,depthR,mR,
                         np.dot(config.bgR['Ktable'],config.bgR['KK']),
                         rectR)

    calib_fix = np.dot(np.linalg.inv(mL),mR)
    config.bgR['Ktable'] = np.dot(calib_fix, config.bgR['Ktable'])
    config.save(newest_folder)


def run_calib(*args):
    close('all')
    _run_calib(*args)


def run_cubeframes():
    global depthL, depthR
    freenect.sync_stop()
    depthL, _ = freenect.sync_get_depth(0)
    np.save('%s/cubeL' % newest_folder, depthL)
    freenect.sync_stop()
    depthR, _ = freenect.sync_get_depth(1)
    np.save('%s/cubeR' % newest_folder, depthR)
    print('saved cube images')


def _run_calib(side='L'):
    """Run the table plane calibration for each camera.
    """
    global points
    print("Making sure only one camera is turned on")
    _ = freenect.sync_get_depth(0)
    _ = freenect.sync_get_depth(1)
    freenect.sync_stop()
    name = 'left' if side=='L' else 'right'
    number = 0 if side=='L' else 1

    print("Getting an image from the %s camera" % name)
    depth, _ = freenect.sync_get_depth(number)

    fig = figure(number)
    clf()
    points = []

    def pick(event):
        global points
        if len(points) >= 4:
            return
        points.append((event.xdata, event.ydata))
        print('Picked point %d of 4' % (len(points)))
        if len(points) == 4:
            print 'OK'
            np.save('%s/boundpts%s' % (newest_folder, side), points)
            np.save('%s/depth%s' % (newest_folder, side), depth)
            print('saved ', name)
            if side=='L':
                points = []
                _run_calib('R')
            if side=='R':
                finish_table_calib()

    imshow(depth)
    draw()
    fig.canvas.mpl_disconnect('button_press_event')
    fig.canvas.mpl_connect('button_press_event', pick)

    print("Make sure that this is the %s camera" % name)
    print("Click four points")


def find_plane(depth, boundpts):
    # Build a mask of the image inside the convex points clicked
    u,v = uv = np.mgrid[:480,:640][::-1]
    mask = np.ones((480,640),bool)
    for (x,y),(dx,dy) in zip(boundpts, boundpts - np.roll(boundpts,1,0)):
        mask &= ((uv[0]-x)*dy - (uv[1]-y)*dx)<0

    # Borrow the initialization from calibkinect
    # FIXME (this is assumed to be the case by normals.project, etc)
    KK = calibkinect.xyz_matrix().astype('f')

    # Find the average plane going through here
    n,w = normals.normals_c(depth.astype(np.float32))
    maskw = mask & (w>0)
    abc = n[maskw].mean(0)
    abc /= np.sqrt(np.dot(abc,abc))
    a,b,c = abc
    x,y,z = [_[maskw].mean() for _ in normals.project(depth)]
    d = -(a*x+b*y+c*z)
    tableplane = np.array([a,b,c,d])
    tablemean = np.array([x,y,z])

    # Backproject the table plane into the image using inverse transpose
    tb0 = np.dot(KK.transpose(), tableplane)

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

    boundptsM = []
    for (up,vp) in boundpts:
        # First project the image points (u,v) onto to the (imaged) tableplane
        dp  = -(tb0[0]*up  + tb0[1]*vp  + tb0[3])/tb0[2]

        # Then project them into metric space
        xp, yp, zp, wp  = np.dot(KtableKK, [up,vp,dp,1])
        xp  /= wp ; yp  /= wp ; zp  /= wp;
        boundptsM += [[xp,yp,zp]]

    # Use OpenGL and framebuffers to draw the table and the walls
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    rb,rbc = glGenRenderbuffers(2)
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
    glEnable(GL_DEPTH_TEST)
    glDrawBuffer(0)
    glReadBuffer(0)
    glClear(GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,640,0,480,0,-3000)
    glMultMatrixf(np.linalg.inv(KtableKK).transpose())
    def draw():
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
    openglbgHi = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
    glFrontFace(GL_CW)
    draw()
    openglbgLo = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
    glFrontFace(gf)
  
    openglbgHi *= 3000
    openglbgLo *= 3000
    #openglbgLo[openglbgLo>=2047] = 0
    openglbgHi[openglbgHi>=2047] = 0
    openglbgLo[openglbgHi==openglbgLo] = 0
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(2, [rb,rbc]);
    glDeleteFramebuffers(1, [fbo]);
    glReadBuffer(GL_BACK)
    glDrawBuffer(GL_BACK)
  
    background = np.array(depth)
    background[~mask] = 2047
    background = np.minimum(background,openglbgHi)
    #backgroundM = normals.project(background)
  
    openglbgLo = openglbgLo.astype(np.uint16)
    background = background.astype(np.uint16)
    background[background>=3] -= 3
    openglbgLo += 3
    
    if 1:
        figure(0);
        imshow(openglbgLo)
    
        figure(1);
        imshow(background)
    
    return dict(
        bgLo=openglbgLo,
        bgHi=background,
        boundpts=boundpts,
        boundptsM=boundptsM,
        KK=KK,
        Ktable=Ktable)
