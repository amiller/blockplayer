from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import config
import grid
import lattice

if not 'initialized' in globals():
    initialized = False


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
    glOrtho(0, 640, 0, 480, 0, -3000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    KtableKK = np.dot(config.bg['Ktable'], config.bg['KK'])
    glMultMatrixf(np.linalg.inv(KtableKK).transpose())

    glMultMatrixf(np.linalg.inv(modelmat).transpose())

    glScale(config.LW, config.LH, config.LW)
    glTranslate(*grid.bounds[0])

    blocks = grid.grid_vertices(occ_grid)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointeri(blocks['vertices'])
    glColor(1,0,0)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointerub(blocks['coords'])
    glDrawElementsui(GL_QUADS, blocks['quad_inds'])
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glFinish()

    depth = np.zeros((480,640),dtype='f')+1.

    depth[T:B,L:R] = glReadPixels(L, T, R-L, B-T, GL_DEPTH_COMPONENT,
                                  GL_FLOAT).reshape(B-T, R-L)

    coords = np.empty((480,640,4),dtype='u1')
    coords[T:B,L:R,:] = glReadPixels(L, T, R-L, B-T, GL_RGBA,
                                     GL_UNSIGNED_BYTE,
                        outputType='array').reshape(B-T, R-L, -1)
    glBindFramebuffer(GL_FRAMEBUFFER,0)

    return coords, depth*3000
