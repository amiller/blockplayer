import numpy as np
import pylab
from OpenGL.GL import *
import cv

from blockplayer import expmap
from blockplayer import dataset
from blockplayer import config
from blockplayer import preprocess
from blockplayer import normals
from blockplayer import opencl
from blockplayer import lattice


from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='flatrot_opencl', size=(640,480))


def show_flatrot():
    pass


def show_normals_sphere(n, w):
    R,G,B = color_axis(n)
    window.update_xyz(n[:,:,0], n[:,:,1], n[:,:,2], (R,G,B,w*(R+G+B)))
    window.upvec = np.array([0,1,0])
    window.Refresh()


def show_normals(n, w, name='normals'):
    im = cv.CreateImage((n.shape[1],n.shape[0]), 32, 3)
    cv.SetData(im, np.dstack(3*[w]) * n[:,:,::-1]/2+.5)
    cv.ShowImage(name, im)


def show_normals_polar(n, w):
    X,Y,Z = n[:,:,0], n[:,:,1], n[:,:,2]
    dx,dy,dz = lattice.project(X,Y,Z, R_oriented)
    cx,cy,cz = lattice.color_axis(dx,dy,dz,w)
    by = (w>0)&(np.abs(dy)<0.3)  # &blocks
    theta = np.arctan2(dz[by], dx[by])*4
    hist,edges = np.histogram(theta, 64, range=(-np.pi,np.pi),normed=True)
    pylab.figure(1)
    #pylab.clf()
    pylab.polar(edges[:-1], hist)
    figure(2)
    pylab.imshow(by)
    return theta,hist,edges,by


def once():
    global depth
    if not FOR_REAL:
        dataset.advance()
        depth = dataset.depth
    else:
        opennpy.sync_update()
        depth,_ = opennpy.sync_get_depth()

    def from_rect(m,rect):
        (l,t),(r,b) = rect
        return m[t:b,l:r]

    global mask, rect
    try:
        (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)
    except IndexError:
        return

    normals.opencl.set_rect(rect)
    normals.normals_opencl(depth, mask, rect).wait()

    # Find the lattice orientation and then translation
    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()

    if 'R_correct' in globals():
        # Correct the lattice ambiguity for 90 degree rotations just by
        # using the previous estimate. This is good enough for illustrations
        # but global alignment is preferred (see hashalign)
        R_oriented,_ = grid.nearest(R_correct, R_oriented)

    R_aligned = lattice.translation_opencl(R_oriented)
    R_correct = R_aligned

    # Find the color based on the labeling from lattice
    global face, Xo, Yo, Zo
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)

    global cx,cy,cz
    cx,cy,cz,_ = np.rollaxis(np.frombuffer(np.array(face).data,
                                           dtype='i1').reshape(-1,4),1)-1
    global R,G,B
    R,G,B = [np.abs(_).astype('f') for _ in cx,cy,cz]
    if 0:
        G *= 0
        R *= 0
        B *= 0
    else:
        pass

    # Draw the points collected on a sphere
    nw = normals.opencl.get_normals()
    global n,w
    n,w = nw[:,:,:3], nw[:,:,3]

    if 1:  # Point cloud position mode
        X,Y,Z,_ = np.rollaxis(opencl.get_xyz(),1)
    else:
        X,Y,Z = n[:,:,0], n[:,:,1], n[:,:,2]
        X,Y,Z = map(lambda _: _.copy(), (X,Y,Z))

    # Render the points in 'table space' but colored with the axes from flatrot
    window.update_xyz(X,Y,Z, COLOR=(R,G*0,B,R*0+w.flatten()))
    window.clearcolor = [1,1,1,0]
    window.Refresh()
    #pylab.imshow(1./depth)
    pylab.waitforbuttonpress(0.01)


def resume():
    while 1: once()


def go(dset=None, frame_num=0, forreal=False):
    global FOR_REAL
    FOR_REAL = forreal
    start(dset, frame_num)
    resume()


def start(dset=None, frame_num=0):
    if not FOR_REAL:
        if dset is None:
            dataset.load_random_dataset()
        else:
            dataset.load_dataset(dset)
        while dataset.frame_num < frame_num:
            dataset.advance()
    else:
        config.load('data/newest_calibration')
        dataset.setup_opencl()
        global R_correct
        if 'R_correct' in globals():
            del R_correct


@window.event
def post_draw():
    # Draw some axes
    if 0:
        glLineWidth(3)
        glPushMatrix()
        mat = np.eye(4)
        mat[:3,:3] = R_correct[:3,:3]
        if 'R_oriented' in globals():
            glMultMatrixf(np.linalg.inv(mat).transpose())

        glScalef(1.5,1.5,1.5)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(-1,0,0); glVertex3f(1,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
        glColor3f(0,0,1); glVertex3f(0,0,-1); glVertex3f(0,0,1)
        glEnd()
        glPopMatrix()


if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    print """BROKEN DEMO WARNING: This demo has fallen out of sync. If I fix this demo, I'll probably notice to remove this message."""
    #go()
