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
from blockplayer import flatrot
from blockplayer import lattice
from blockplayer import classify


from blockplayer.visuals.pointwindow import PointWindow
global window
if not 'window' in globals():
    window = PointWindow(title='flatrot_opencl', size=(640,480))


def show_flatrot():
    pass


def show_normals_sphere(n, w):
    R,G,B = color_axis(n)
    window.update_xyz(n[:,:,0], n[:,:,1], n[:,:,2], (R+.5,G+.5,B+.5,w*(R+G+B)))
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

    global mask, rect, modelmat
    (mask,rect) = preprocess.threshold_and_mask(depth,config.bg)

    normals.opencl.set_rect(rect)
    normals.normals_opencl(depth, mask, rect).wait()

    global R_oriented, R_aligned, R_correct
    R_oriented = lattice.orientation_opencl()

    nw = normals.opencl.get_normals()
    global n,w
    n,w = nw[:,:,:3], nw[:,:,3]
    #show_normals(n, w, 'normals_opencl')
    #show_normals_sphere(n, w)

    # Perform the 'labeling' by rotating the normals using the output from
    # flatrot, then threshold with color_axis
    global dx,dy,dz
    global cx,cy,cz
    global labelmask
    X,Y,Z = n[:,:,0], n[:,:,1], n[:,:,2]
    X,Y,Z = map(lambda _: _.copy(), (X,Y,Z))

    if 1:
        global label_image
        label_image = classify.predict(depth)
        blocks = from_rect((label_image[0]==0)|(label_image[1]<0.7) & mask, rect)
        dx,dy,dz = lattice.project(X,Y,Z, R_oriented)
        cx,cy,cz = lattice.color_axis(dx,dy,dz,w)
        by = (np.abs(dy)<0.3) & (blocks)  # &blocks
        labelmask = by

        inlier = ((cx>0)|(cz>0))
        outlier = ~inlier
        correct = inlier&blocks | (outlier&~blocks)
        incorrect = inlier&~blocks | (outlier&blocks)
        X[~by] = Y[~by] = Z[~by] = 0

        #R,G,B = (cx>0)&blocks, 0*blocks, (cz>0)&blocks
        R,G,B = incorrect,by*0,correct
        R,G,B = map(lambda _: _.astype('f'), (R,G,B))
    else:
        dx,dy,dz = lattice.project(X,Y,Z, R_oriented)
        cx,cy,cz = lattice.color_axis(dx,dy,dz,w)
        R,G,B = np.abs(cx).astype('f'),cy.astype('f')*0,np.abs(cz).astype('f')

    # Render the points in 'table space' but colored with the axes from flatrot
    window.update_xyz(X,Y,Z, COLOR=(R,G,B,R+G+B+.5))
    window.clearcolor = [1,1,1,0]
    window.Refresh()
    pylab.imshow(1./depth)
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


@window.event
def post_draw():
    # Draw some axes
    glLineWidth(3)
    glPushMatrix()
    if 'R_oriented' in globals():
        glMultMatrixf(np.linalg.inv(R_oriented).transpose())

    glScalef(1.5,1.5,1.5)
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
    glEnd()
    glPopMatrix()


if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    pass
    #go()
