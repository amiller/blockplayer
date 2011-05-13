from pointwindow import PointWindow
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from blockplayer import config
from blockplayer import blockdraw


# Window for drawing point cloudsc
class BlockWindow(PointWindow):

    def __init__(self, *args, **kwargs):
        self.modelmat = np.eye(4)
        self.flag_drawgrid = False
        super(BlockWindow,self).__init__(*args, **kwargs)

    def draw_board(self):
        LW,LH = config.LW, config.LH
        glPolygonOffset(1.0,0.2)
        glEnable(GL_POLYGON_OFFSET_FILL)

        # Draw the gray table
        if 'bg' in config.__dict__:
            glBegin(GL_QUADS)
            glColor(0.2,0.2,0.2,1)
            for x,y,z in config.bg['boundptsM']:
                glVertex(x,y,z)
            glEnd()

        glPushMatrix()
        glMultMatrixf(np.linalg.inv(self.modelmat).transpose())
        glScale(LW,LH,LW)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glPushMatrix()
        glTranslate(*config.bounds[0])
        blockdraw.draw()
        glPopMatrix()

        # Draw the shadow blocks (occlusions)
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Draw the axes for the model coordinate space
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
        glEnd()

        # Draw a grid for the model coordinate space
        if self.flag_drawgrid:
            glLineWidth(1)
            glBegin(GL_LINES)
            GR = config.GRIDRAD/2
            glColor3f(1.0,1.0,1.0)
            for j in range(0,1):
                for i in range(-GR,GR+1):
                    glVertex(i,j,-GR); glVertex(i,j,GR)
                    glVertex(-GR,j,i); glVertex(GR,j,i)
            glEnd()
            glPopMatrix()
            pass
        glLineWidth(1)

    def on_draw(self):
        super(BlockWindow,self).set_camera()

        glClearColor(*self.clearcolor)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        self._wrap('pre_draw')

        if not self.modelmat is None:
            self.draw_points()
            self.draw_board()

        self._wrap('post_draw')
