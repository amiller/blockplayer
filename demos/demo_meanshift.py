import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from blockplayer import expmap
from blockplayer import dataset
from blockplayer import config
from wxpy3d import PointWindow
from wxpy3d.opengl_state import opengl_state

from rtmodel.camera import kinect_camera
from rtmodel.rangeimage import RangeImage

global window
if not 'window' in globals():
    window = PointWindow(title='demo_meanshift', size=(640,480))
    glutInit()
    window.Move((0,0))

dataset.load_dataset('data/sets/6dof_cube_0')

def normalize(X):
    return X / np.sqrt(np.sum(X*X))

def random_basis():
    X = normalize(np.random.rand(3)-0.5)
    Y = normalize(np.random.rand(3)-0.5)
    Z = normalize(np.cross(X, Y))
    X = normalize(np.cross(Y, Z))
    return np.array([X,Y,Z]).astype('f')

def stack(arrs):
    dtype = arrs[0].dtype
    stack = []
    for a in arrs:
        assert a.dtype == dtype
        assert len(a.shape) >= 2
        stack.append(a.reshape((-1,) + a.shape[2:]))
    return np.concatenate(stack)

def unstack(stack, rects):
    c = 0
    imgs = []
    for rect in rects:
        (l,t),(r,b) = rect
        h,w = b-t, r-l
        imgs.append(stack[c:c+w*h].reshape((h,w,-1)))
        c += w*h
    return imgs

def dataset_points(cams=(0,1)):
    global rimgs
    rimgs = []
    for cam in cams:
        camera = config.cameras[cam]['Ktable']
        rimg = RangeImage(dataset.depths[cam], camera)
        rimg.threshold_and_mask(config.cameras[cam])
        rimg.filter(win=6)
        rimg.compute_normals()
        rimgs.append(rimg)

    n = stack([rimg.normals for rimg in rimgs])
    w = stack([rimg.weights for rimg in rimgs])

    global basis, nxyz
    basis = random_basis()
    nxyz = n[w>0]#n
    window.update_points(XYZ=nxyz)

def random_points(N=1000, mu=0.2):
    basis = random_basis()
    ind = np.random.randint(3, size=(N,))
    points = (np.sign(np.random.rand(N,1)-0.5) * basis[ind, :] +
              np.random.normal(scale=mu, size=(N,1)) * basis[(ind+1)%3, :] +
              np.random.normal(scale=mu, size=(N,1)) * basis[(ind+2)%3, :])
    points = points / np.sqrt(np.sum(points*points, 1)).reshape(-1,1)
    return basis, points.astype('f')

def color_axis(X,Y,Z,w,d=0.3):
    X2,Y2,Z2 = X*X,Y*Y,Z*Z
    d = 1/d
    cc = Y2+Z2, Z2+X2, X2+Y2
    # I changed this, so it's possible the threshold should be adjusted
    # cx = [w*np.maximum(1.0-(c*d)**2),0*c) for c in cc]
    cx = [w*np.maximum(1.0-(c*(d**2)),0*c) for c in cc]
    return [c for c in cx]

def mean_shift(estimate, nxyz, d=0.3):
    global X, Y, Z
    X, Y, Z = np.dot(estimate, nxyz.transpose())
    X2, Y2, Z2 = X*X, Y*Y, Z*Z
    cc = Y2+Z2, Z2+X2, X2+Y2

    # Either uniform weighting (within a region)
    cc = [(c<d**2).astype('f') for c in cc]
    # Or inverse square weighting
    #cc = [1*np.maximum(1.0-(c*((1/d)**2)),0*c).astype('f')
    #      for c in cc]

    # Note
    # cx, cy, and cz are all disjoint.
    # Therefore (cx+cz).sum() == cx.sum() + cz.sum()

    global cx, cy, cz
    cx, cy, cz = cc
    nx = ((cy-cz)*Y*Z).sum() / (cy+cz).sum() / 1 \
        if (cy+cz).sum() > 0 else 0
    ny = ((cz-cx)*Z*X).sum() / (cz+cx).sum() / 1 \
        if (cz+cx).sum() > 0 else 0
    nz = ((cx-cy)*X*Y).sum() / (cx+cy).sum() / 1 \
        if (cx+cy).sum() > 0 else 0

    rgba = np.array(cc + [np.ones_like(cc[0])]).transpose()
    window.update_points(XYZ=nxyz, RGBA=rgba/2+.5)

    return expmap.euler2rot((nx, ny, nz))
    

@window.event
def post_draw():
    with opengl_state():
      if 0: 
        B = np.eye(4); B[:3,:3] = basis
        glMultMatrixf(np.linalg.inv(B).transpose())
        glScale(1.3, 1.3, 1.3)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(3, 0xAAAA)
        glBegin(GL_LINES)
        glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
        glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
        glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
        glEnd()

    with opengl_state():
        #quad = gluNewQuadric()
        #gluSphere(quad, 0.99, 10, 10)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(3, 0xAAAA)
        glColor(0.1,0.1,0.1)
        glutWireSphere(1, 10, 10)
        glTranslate(-1,-1,-1)
        glScale(0.3,0.3,0.3)
        glBegin(GL_LINES)
        glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
        glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
        glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
        glEnd()


    with opengl_state():
      if 1:
        B = np.eye(4); B[:3,:3] = estimate
        glMultMatrixf(np.linalg.inv(B).transpose())
        glLineWidth(2)
        glScale(1.3, 1.3, 1.3)
        glBegin(GL_LINES)
        glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
        glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
        glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
        #glColor(1,0,0); glVertex(-1,0,0); glVertex(1,0,0)
        #glColor(0,1,0); glVertex(0,-1,0); glVertex(0,1,0)
        #glColor(0,0,1); glVertex(0,0,-1); glVertex(0,0,1)
        glEnd()


@window.eventx
def EVT_IDLE(*kwargs):
    return
    window.rotangles[0] += 0.03
    window.rotangles[1] += 0.03
    window.Refresh()


D = 0.4

def meanshift():
    global estimate
    for i in xrange(4):
        estimate = np.dot(np.linalg.inv(mean_shift(estimate, nxyz, D)), estimate)
        mean_shift(estimate, nxyz, d=D)
        window.Refresh()
        pylab.waitforbuttonpress(0.2)

    
basis, nxyz = random_points(mu=0.1)
estimate = random_basis()
#estimate = array([[ 0.86042833,  0.48787376,  0.14711326],
#                  [-0.40038016,  0.46868348,  0.78742081],
#                  [ 0.31521237, -0.73642039,  0.59860349]], 
#                 dtype=float32)

mean_shift(estimate, nxyz, D)
#window.update_points(XYZ=nxyz)
window.Refresh()


def once():
    dataset.advance(skip=6)
    dataset_points()
    figure(1)
    clf();
    imshow(dataset.depths[0])
    figure(2)
    clf();
    imshow(dataset.depths[1])
    meanshift()


def go():
    while 1:
        once()

@window.eventx
def EVT_KEY_DOWN(evt):
    global basis, nxyz
    if evt.GetKeyCode() == ord('R'):
        basis, nxyz = random_points(mu=0.1)
        mean_shift(estimate, nxyz, D)
    if evt.GetKeyCode() == ord('M'):
        meanshift()
    if evt.GetKeyCode() == ord('A'):
        dataset.advance(skip=20)
        dataset_points()
        figure(1)
        clf();
        imshow(dataset.depths[0])
        figure(2)
        clf();
        imshow(dataset.depths[1])
    window.Refresh()
