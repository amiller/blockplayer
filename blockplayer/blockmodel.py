from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL import arrays
import numpy as np
from rtmodel.rangeimage import RangeImage
from rtmodel.camera import Camera
from wxpy3d.opengl_state import opengl_state
import glxcontext
from contextlib import contextmanager
from OpenGL.GL import shaders

#if not 'initialized' in globals():
initialized = False
 
def initialize():
    global initialized
    glxcontext.makecurrent()

    if initialized:
        return

    global fbo
    global rb, rbc, rbs

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Create three render buffers:
    #    1) xyz/depth  2) normals 3)  4) internal depth
    rbXYZD,rbN,rbD = glGenRenderbuffers(3)

    # 1) XYZ/D buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbXYZD)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbXYZD)

    # 2) Normals buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbN)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT1,
                              GL_RENDERBUFFER, rbN)

    # 4) depth buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbD)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbD)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    global VERTEX_SHADER, FRAGMENT_SHADER, SYNTHSHADER

    # This vertex shader simulates a range image camera. It takes in 
    # camera (kinect) parameters as well as model parameters. It project
    # the model points into the camera space in order to get the right
    # correct geometry. We additionally precompute the world-coordinates
    # XYZ for each pixel, as well as the world-coordinate normal vectors.
    # (This eliminates some of the steps that would normally be computed
    # using image processing, i.e. filtering, subtraction, normals.)
    VERTEX_SHADER = shaders.compileShader("""#version 120
        varying vec4 v_xyz;
        varying vec3 n_xyz;
        varying vec4 position;
        uniform mat4 model;
        uniform mat3 model_norm;
        uniform mat4 view_inv;

        mat3 mat4tomat3(mat4 m) { return mat3(m[0].xyz, m[1].xyz, m[2].xyz); }

        void main() {
            v_xyz = view_inv * gl_ModelViewMatrix * gl_Vertex;
            n_xyz = gl_Normal;
            position = gl_ModelViewProjectionMatrix * gl_Vertex;
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        }""", GL_VERTEX_SHADER)

    
    FRAGMENT_SHADER = shaders.compileShader("""#version 120
        varying vec4 v_xyz;
        varying vec3 n_xyz;
        varying vec4 position;
        varying out vec4 xyzd;
        varying out vec4 nxyz;
        uniform mat4 view_inv;

        void main() {
            xyzd.w = 200 / (1.0 - position.z/position.w);
            xyzd.xyz = v_xyz.xyz;
            nxyz.xyz = n_xyz;
            //xyzd = abs((view_inv, gl_ModelViewMatrix)[2]);
            //nxyz = abs((gl_ModelViewMatrix)[3]);//abs(modelviewprojection[3]);//vec4(1,2,3,4);//n_xyz;
        }""", GL_FRAGMENT_SHADER)

    SYNTHSHADER = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
    initialized = True

@contextmanager
def render_context(RT, cam, rect=((0,0),(640,480))):
    # Configure the framebuffer object
    initialize()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    #glDrawBuffers(GL_NONE)
    glDrawBuffers([GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1])
    (L,T),(R,B) = rect
    glViewport(0, 0, 640, 480)
    glClearColor(0,0,0,0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Trick OpenGL into giving us a projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 0, 480, -10, 0)
    orthomat = glGetFloatv(GL_PROJECTION_MATRIX).T

    # Configure the shader
    shaders.glUseProgram(SYNTHSHADER)

    # Upload uniform parameters (model and camera matrices)
    assert RT.shape == (4,4) and RT.dtype == np.float32
    assert type(cam) is Camera

    def uniform(f, name, mat, n=1): 
        assert mat.dtype == np.float32
        mat = np.ascontiguousarray(mat)
        f(glGetUniformLocation(SYNTHSHADER, name), n, True, mat)

    view_inv = np.dot(np.dot(np.linalg.inv(RT), cam.RT), cam.KK)
    uniform(glUniformMatrix4fv, 'view_inv', view_inv)

    def read(debug=False):
        global readpixels, readpixelsA, readpixelsB, depth

        glReadBuffer(GL_COLOR_ATTACHMENT0)
        readpixelsA = glReadPixels(L, T, R-L, B-T, GL_RGBA, GL_FLOAT,
                                   outputType='array').reshape((480,640,4))

        glReadBuffer(GL_COLOR_ATTACHMENT1)
        readpixelsB = glReadPixels(L, T, R-L, B-T, GL_RGBA, GL_FLOAT,
                                   outputType='array').reshape((480,640,4))

        # Returns the distance in milliunits
        depth = readpixelsA[:,:,3]

        # Sanity check
        if debug:
            if depth.max() == 0: print 'Degenerate (zero) depth image'
            print 'Checking two equivalent depth calculations'
            old = np.seterr(divide='ignore')
            within_eps = lambda a,b: np.all(np.abs(a - b) < 2)
            readpixels = glReadPixels(L, T, R-L, B-T, GL_DEPTH_COMPONENT, GL_FLOAT).reshape((480,640))
            depth2 = (100.0 / np.nan_to_num(1.0 - readpixels)).astype('u2')
            assert within_eps(depth2, depth)
            np.seterr(**old)

        color = readpixelsA[:,:,:3]
        normals = readpixelsB[:,:,:3]
        return depth, color, normals

    try:
        yield read

    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        shaders.glUseProgram(0)


class BlockModel(object):
    def __init__(self, occ, RT=np.eye(4,dtype='f'), dims=None, bounds=None):
        if dims is None:
            from blockplayer import config
            dims = config.LW, config.LH, config.LW
        if bounds is None:
            #(x0,y0,z0),(x1,y1,z1)
            bounds = config.bounds
        self.dims = dims
        self.bounds = bounds
        self.occ = occ
        self.blocks = grid_vertices(self.occ, None)
        self.RT = RT

    def draw(self, debug=False):
        blocks = self.blocks
        with opengl_state():
            glScale(*self.dims)
            glTranslate(*self.bounds[0])
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointeri(blocks['vertices'])

            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointerf(blocks['normals'])

            glColor(1,0,0)
            #glEnableClientState(GL_COLOR_ARRAY)
            #glColorPointerub(blocks['coords'])

            glDrawElementsui(GL_QUADS, blocks['quad_inds'])
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glFinish()

            if debug:
                print 'modelviewprojection'
                print(glGetFloatv(GL_MODELVIEW_MATRIX).T)


    def render(self, camera, rect=((0,0),(640,480)), do_read=True):
        with render_context(self.RT, camera, rect) as read:
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_DEPTH_TEST)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glMultMatrixf(np.linalg.inv(camera.KK).T)
            glMultMatrixf(np.linalg.inv(camera.RT).T)
            glMultMatrixf(self.RT.T)

            self.draw()

            if do_read:
                depth,color,normals = read(debug=False)
                rimg = RangeImage(depth.astype('u2'), camera)
                assert normals.shape[2] == 3 and normals.dtype == np.float32
                #rimg.normals = normals
                #rimg.weights = depth > 0
                return rimg, color, normals



def gt2grid(gtstr, chars='*rR'):
    g = np.array(map(lambda _: map(lambda __: tuple(__), _), eval(gtstr)))
    g = np.rollaxis(g,1)
    res = (g==chars[0])
    for c in chars[1:]:
        res = np.logical_or(res, g==c)
    return np.ascontiguousarray(res)


def grid2gt(occ):
    m = np.choose(occ, (' ', '*'))
    layers = [[''.join(_) for _ in m[:,i,:]]
              for i in range(m.shape[1])]
    import pprint
    return pprint.pformat(layers)


def grid_vertices(grid, color=None):
    """
    Given a boolean voxel grid, produce a list of vertices and indices
    for drawing quads or line strips in opengl
    """
    q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
             [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
             [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
             [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
             [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
             [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]

    normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2]))
              for qz in q]

    blocks = np.array(grid.nonzero()).transpose().reshape(-1,1,3)
    q = np.array(q).reshape(1,-1,3)

    vertices = (q + blocks).reshape(-1,3)
    coords = (q*0 + blocks).astype('u1').reshape(-1,3)

    if not color is None:
        assert color.shape[3] == 3
        color = color[grid,:].reshape(-1,1,3)
        cc = (q.astype('u1')*0+color).reshape(-1,3)
        assert cc.dtype == np.uint8
    else:
        cc = coords

    normals = np.tile(normal, (len(blocks),4)).reshape(-1,3)
    line_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
    quad_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,2,3]

    return dict(blocks=blocks, vertices=vertices, coords=coords,
                normals=normals, line_inds=line_inds, quad_inds=quad_inds,
                color=cc)


def draw():
    global blocks
    for block in blocks.values():
        draw_block(block)
