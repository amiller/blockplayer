import numpy as np
from OpenGL.GL import *
from blockplayer import expmap
from wxpy3d import PointWindow
from wxpy3d.opengl_state import opengl_state
global window
if not 'window' in globals():
    window = PointWindow(title='demo_meanshift', size=(640,480))
    window.Move((0,0))

def normalize(X):
    return X / np.sqrt(np.sum(X*X))

def random_basis():
    X = normalize(np.random.rand(3)-0.5)
    Y = normalize(np.random.rand(3)-0.5)
    Z = normalize(np.cross(X, Y))
    X = normalize(np.cross(Y, Z))
    return np.array([X,Y,Z]).astype('f')

def random_points(N=1000, mu=0.1):
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
    nx = ((cy-cz)*Y*Z).sum() / (cy+cz).sum()
    ny = ((cz-cx)*Z*X).sum() / (cz+cx).sum()
    nz = ((cx-cy)*X*Y).sum() / (cx+cy).sum()

    rgba = np.array(cc + [np.ones_like(cc[0])]).transpose()
    window.update_points(XYZ=nxyz, RGBA=rgba/2+.5)

    return expmap.euler2rot((nx, ny, nz))
    

@window.event
def post_draw():
    with opengl_state():
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
        B = np.eye(4); B[:3,:3] = estimate
        glMultMatrixf(np.linalg.inv(B).transpose())
        glLineWidth(2)
        glScale(1.3, 1.3, 1.3)
        glBegin(GL_LINES)
        glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
        glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
        glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
        glEnd()


@window.eventx
def EVT_IDLE(*kwargs):
    return
    window.rotangles[0] += 0.03
    window.rotangles[1] += 0.03
    window.Refresh()


def meanshift():
    global estimate
    for i in xrange(80):
        estimate = np.dot(estimate, np.linalg.inv(mean_shift(estimate, nxyz)))
        mean_shift(estimate, nxyz)
        window.Refresh()
        pylab.waitforbuttonpress(0.1)

    
#basis, nxyz = random_points()
estimate = random_basis()
estimate = array([[ 0.86042833,  0.48787376,  0.14711326],
                  [-0.40038016,  0.46868348,  0.78742081],
                  [ 0.31521237, -0.73642039,  0.59860349]], 
                 dtype=float32)

mean_shift(estimate, nxyz)
#window.update_points(XYZ=nxyz)
window.Refresh()
