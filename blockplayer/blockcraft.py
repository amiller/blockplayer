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
    print bx, bz

    return hashalign.apply_correction(occ, bx, 0, bz, -angle_90)


def asdfasdfasdfasdf():
    pass
