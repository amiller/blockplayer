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

# This module will store all the configuration variables
# that are video sequence-specific and that do not
# change during the course of running a video sequence.
import numpy as np
import cPickle as pickle


def load(dir_path):
    with open('%s/config/config.pkl' % dir_path, 'r') as f:
        globals().update(pickle.load(f))
        globals()['LH'] = 0.0192


def save(dir_path):
    with open('%s/config/config.pkl' % dir_path,'w') as f:
        pickle.dump(dict(bg=bg,
                         LH=LH,
                         LW=LW,
                         ), f)


# Duplo block sizes
LH = 0.0198
LW = 0.016


# Jenga Block sizes FIXME:(needs to be remeasured)
#LH = 0.0150
#LW = 0.0200


GRIDRAD = 18
bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,9,GRIDRAD)

