from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import config
import blockdraw
import cv
import speedup_cy

if not 'initialized' in globals():
    initialized = False


# Color targets, defined as hues from 0 to 180
# red, yellow, green, blue, red
color_targets = np.array([0, 20, 50, 120, 180],'i')
color_names = ['red', 'yellow', 'green', 'blue', 'red']

def print_colors():
    
    for c, t in zip(color_names, color_targets):
        F = np.array([[[t, 255, 255]]], dtype='u1')
        cv.CvtColor(F, F, cv.CV_HSV2RGB)
        print c + '\t', F[0,0,:]


def setup():
    global initialized
    if initialized:
        return

    global fbo
    global rb, rbc, rbs

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    rb,rbc = glGenRenderbuffers(2)
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rb);

    glBindRenderbuffer(GL_RENDERBUFFER, rbc)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbc)
    #print('Framebuffer ready:',
    #      glCheckFramebufferStatus(GL_FRAMEBUFFER)==
    #      GL_FRAMEBUFFER_COMPLETE_EXT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    initialized = True


def render_blocks(occ_grid, modelmat, rect=((0,0),(640,480))):
    """
    Returns the result of rendering occ_grid from the point of view of the
    camera.
        returns:
            depthB: numpy array shape=(480,640) dtype=np.float32,
                    distance in mm, just like the kinect (openni)
    """
    setup()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    (L,T),(R,B) = rect

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    #glViewport(L, 480-B, R-L, B-T)
    glViewport(0, 0, 640, 480)
    glClearColor(0,0,0,0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #glOrtho(L, R, T, B, 0, -3000)
    glOrtho(0, 640, 0, 480, -10, 0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    KtableKK = np.dot(config.bg['Ktable'], config.bg['KK'])
    glMultMatrixf(np.linalg.inv(KtableKK).transpose())
    glMultMatrixf(np.linalg.inv(modelmat).transpose())

    glScale(config.LW, config.LH, config.LW)
    glTranslate(*config.bounds[0])

    blocks = blockdraw.grid_vertices(occ_grid, None)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointeri(blocks['vertices'])
    glColor(1,0,0)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointerub(blocks['coords'])
    glDrawElementsui(GL_QUADS, blocks['quad_inds'])
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glFinish()

    readpixels = glReadPixels(L, T, R-L, B-T, GL_DEPTH_COMPONENT, GL_FLOAT)
    readpixelsA = glReadPixels(L, T, R-L, B-T, GL_RGBA, GL_UNSIGNED_BYTE,
                              outputType='array')
    depth = np.empty((480,640),dtype='f')
    coords = np.empty((480,640,4),dtype='u1')
    speedup_cy.stencil_finish(depth, coords, readpixels, readpixelsA, T, L, B, R)
    glBindFramebuffer(GL_FRAMEBUFFER,0)
    return coords, depth


def stencil_carve(depth, modelmat, occ_grid, rgb=None, rect=((0,0),(640,480))):
    """

    """
    global coords, b_total, b_occ, b_vac, depthB

    (L,T),(R,B) = rect
    L,T,R,B = map(int, (L,T,R,B))
    coords, depthB = render_blocks(occ_grid,
                                   modelmat,
                                   rect=rect)
    #print depth.mean(), occ_grid.mean(), rect, depthB.mean()
    assert coords.dtype == np.uint8
    assert depthB.dtype == np.float32
    assert depth.dtype == np.uint16
    assert coords.shape[2] == 4
    assert coords.shape[:2] == depthB.shape == depth.shape == (480,640)

    b_total = np.zeros(occ_grid.shape, 'f')
    b_occ = np.zeros(occ_grid.shape, 'f')
    b_vac = np.zeros(occ_grid.shape, 'f')

    gridmin = np.array(config.bounds[0])
    gridmax = np.array(config.bounds[1])
    gridlen = gridmax-gridmin

    if rgb is None:
        rgb = np.empty((480,640,3),'u1')

    global RGBacc, RGB, HSV
    RGBacc = np.zeros((b_total.shape[0],
                       b_total.shape[1],
                       b_total.shape[2], 3),'i')
    RGB = np.zeros((b_total.shape[0],
                    b_total.shape[1],
                    b_total.shape[2], 3),'u1')
    HSV = np.zeros((b_total.shape[0],
                    b_total.shape[1],
                    b_total.shape[2], 3),'u1')
    assert rgb.dtype == np.uint8

    if 1:
        speedup_cy.stencil_carve(depthB, depth, coords,
                                 gridlen[0], gridlen[1], gridlen[2],
                                 RGBacc, rgb, RGB,
                                 b_total, b_occ, b_vac,
                                 T, L, B, R)
    else:
        # This may be out of date - prefer the weave version
        bins = [np.arange(0,gridmax[i]-gridmin[i]+1)-0.5
                for i in range(3)]
        c = coords[:,:,:3].reshape(-1,3)
        w = ((depth>0)&(depthB<inf)).flatten()
        b_total,_ = np.histogramdd(c, bins, weights=w)
        b_occ,_ = np.histogramdd(c, bins, weights=w&
                                 (np.abs(depthB-depth)<10).flatten())
        b_vac,_ = np.histogramdd(c, bins, weights=w&
                                 (depthB+10<depth).flatten())


    if 1:
        cv.CvtColor(RGB.reshape(1,-1,3), HSV.reshape(1,-1,3), cv.CV_RGB2HSV);
        #HSV[:,:,:,1:] = 255
        #Hdiff = np.abs(HSV[:,:,:,:1] - color_targets.reshape(1,1,1,-1))
        #HSV[:,:,:,0] = color_targets[np.argmin(Hdiff,axis=3)]
        speedup_cy.fix_colors(HSV, color_targets)
        cv.CvtColor(HSV.reshape(1,-1,3), RGB.reshape(1,-1,3), cv.CV_HSV2RGB);
    else:
        RGB = (RGB.astype('i')*4).clip(0,255)
    return b_occ, b_vac, b_total
