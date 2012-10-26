"""
blockplayer/server.py - a socket server that provides pose/model output as well as
access to the camera

Protocol:
    A duplex listening tcp socket is created and bound to port 8134.
    Both streams consist of JSON formatted messages.

    output stream:
        ("depth", 480*[640*[int]])   
        ("pose", 4*[4*[float]])         a pose message is sent every successful frame 
                                        (i.e., at most 30/s)
        ("voxels", X*[Y*[Z*[0 or 1]]])  a voxel grid is send every successful frame
                                        (i.e., at most 30/s)

    input stream:
        "depth":      (a depth message is inserted into the output stream)
"""

import zmq
import json
import opennpy
import numpy as np
import time

def setup():
    global socket
    if not 'socket' in globals():
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.bind('tcp://*:8134')

def send_pose(RT_correct):
    setup()
    assert RT_correct.dtype == np.float32
    assert RT_correct.shape == (4,4)
    socket.send_json(('pose',RT_correct.tolist()))

def send_voxels(grid_occ):
    setup()
    assert grid_occ.dtype == bool
    assert len(grid_occ.shape) == 3
    socket.send_json(('voxels',grid_occ.astype(np.uint8).tolist()))

def poll():
    setup()
    while True:
        try:
            cmd = socket.recv_json(zmq.NOBLOCK)
            if cmd == 'depth':
                opennpy.sync_update()
                depth,_ = opennpy.sync_get_depth()
                socket.send_json(('depth',depth.tolist()))
            if cmd == 'rgb':
                opennpy.sync_update()
                rgb,_ = opennpy.sync_get_video()
                socket.send_json(('rgb',rgb.tolist()))
        except zmq.ZMQError, e:
            if e.errno == zmq.EAGAIN: break
            else: raise

if __name__ == '__main__':
    print 'Running 0mq server on tcp://*:8134'
    while True:
        poll()
        send_pose(np.eye(4).astype('f'))
        time.sleep(0.033)
