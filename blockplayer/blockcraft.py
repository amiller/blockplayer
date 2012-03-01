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
import config
import hashalign


def centered_rotated(R_correct, occ):
    """
    Args:
        R_correct: model matrix 
        occ: 3D voxel grid
    Returns:
        3D voxel grid
    """
    # Apply a 90 degree rotation
    angle = np.arctan2(R_correct[0,2], R_correct[0,0])
    angle_90 = np.round(angle/(np.pi/2))

    return hashalign.apply_correction(occ, 0, 0, 0, -angle_90)


def translated_rotated(R_correct, occ):
    # Apply a 90 degree rotation
    angle = np.arctan2(R_correct[0,2], R_correct[0,0])
    angle_90 = np.round(angle/(np.pi/2))

    # Apply a translation
    bx,_,bz = np.round(np.linalg.inv(R_correct)[:3,3]/config.LW).astype('i')

    return hashalign.apply_correction(occ, bx, 0, bz, -angle_90)

