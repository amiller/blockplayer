# Andrew Miller <amiller@cs.ucf.edu> 2012
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
from rtmodel import camera

# Useful Quantities
# Duplo block sizes
duplo_LH = 0.0198
duplo_LW = 0.016

# Jenga Block sizes FIXME:(needs to be remeasured)
jenga_LH = 0.0150
jenga_LW = 0.0200


# Runtime Parameters (not saved or written by config.load)
GRIDRAD = 18
bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,9,GRIDRAD)

# Default Parameters (saved and restored by config.load)
blocktype = 'duplo'
LW,LH = duplo_LW, duplo_LH
cameras = []
version = '2012Aug6'

def load(dir_path):
    global LH, LW, cameras, bg, blocktype, center

    with open('%s/config/config.pkl' % dir_path, 'r') as f:
        conf = pickle.load(f)
        assert type(conf) is dict

        if not 'version' in conf:
            # Assume version from study_user_data for vr2012 paper
            LH = conf['LH']
            LW = conf['LW']
            bg = [conf['bg'],]
            cameras = [camera.Camera(KK=c['KK'], RT=c['Ktable']) for c in bg]
            blocktype = 'duplo' # FIXME: What about jenga? Some 
                                # data in this format has jenga.

        elif conf['version'] == '2012Jul24':
            assert conf['blocktype'] == 'duplo'
            blocktype='duplo'
            LH = duplo_LH
            LW = duplo_LW
            bg = conf['cameras']
            for c in bg: c['Ktable'][2,3] -= 0.45
            cameras = [camera.Camera(KK=c['KK'], RT=c['Ktable']) for c in bg]

        else:
            raise ValueError('unrecognized version %s' % version)

def save(dir_path):
    with open('%s/config/config.pkl' % dir_path,'w') as f:
        pickle.dump(dict(cameras=cameras,
                         blocktype=blocktype,
                         center=center,
                         version=version,
                         ), f)
