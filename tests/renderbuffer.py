from visuals.camerawindow import CameraWindow
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *


if not 'window' in globals():
    window = CameraWindow()


def draw_offscreen_color():
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    rbc = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbc)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbc)
    glClear(GL_COLOR_BUFFER_BIT)
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 0, 480, -10, 10)

    def draw():
        glBegin(GL_TRIANGLES)
        glVertex(100, 100)
        glVertex(200, 400)
        glVertex(300, 100)
        glEnd()
        glFinish()

    draw()
    output = glReadPixels(0, 0, 640, 480, GL_BGR,
                          GL_FLOAT).reshape(480, 640, 3)
    return output


def draw_offscreen_depth():
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    rb = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rb)
    glDrawBuffer(0)
    glReadBuffer(0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 0, 480, -10, 10)

    def draw():
        glBegin(GL_TRIANGLES)
        glVertex(100, 100)
        glVertex(200, 400)
        glVertex(300, 100)
        glEnd()
        glFinish()

    draw()
    output = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT,
                          GL_FLOAT).reshape(480, 640)
    return output
