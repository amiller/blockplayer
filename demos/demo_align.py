# Andrew Miller <amiller@cs.ucf.edu> 2011
#
# BlockPlayer - 3D model reconstruction using the Lattice-First algorithm
# See: 
#    "Interactive 3D Model Acquisition and Tracking of Building Block Structures"
#    Andrew Miller, Brandyn White, Emiko Charbonneau, Zach Kanzler, and Joseph J. LaViola Jr.
#    IEEE VR 2012, IEEE TVGC 2012
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import pylab
from blockplayer import grid
from blockplayer import hashalign
from blockplayer import blockdraw

from blockplayer.visuals.blockwindow import BlockWindow
global window
if not 'window' in globals():
    window = BlockWindow(title='demo_align', size=(640,480))
    window.Move((0,0))


def load_gt():
    global GT
    with open('data/experiments/gt/gt1.txt') as f:
        GT = grid.gt2grid(f.read())


if not 'GT' in globals():
    GT = None
    load_gt()


def show_align(A, B):
    blockdraw.clear()
    blockdraw.show_grid('0', A & ~B, color=np.array([1,0,0,1]))
    blockdraw.show_grid('1', B & ~A, color=np.array([0,0,1,1]))
    blockdraw.show_grid('2', A & B, color=np.array([1,0,1,1]))
    window.clearcolor=[1,1,1,0]
    window.Refresh()


def once():
    global problem
    problem = GT.copy()
    problem = grid.apply_correction(problem,
                                    np.random.randint(-18,18),
                                    np.random.randint(-18,18),
                                    np.random.randint(0,4))

    featureA = hashalign.find_features(GT)
    featureB = hashalign.find_features(problem)
    matches = hashalign.match_features(featureA, featureB, GT.shape[0])

    bestmatch, besterr = hashalign.find_best_alignment(GT, GT*0,
                                                       problem, problem*0)

    if 0:
        for match in matches:
            show_align(GT, hashalign.apply_correction(problem, *match))
            window.Refresh()
            pylab.waitforbuttonpress(0.01)

        print besterr, bestmatch
    show_align(GT, hashalign.apply_correction(problem, *bestmatch))


if 'window' in globals():
    window.Refresh()

if __name__ == '__main__':
    pass
