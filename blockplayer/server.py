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

    input stream:
        "depth":      (a depth message is inserted into the output stream)
"""

import zmq
import json
import opennpy
import numpy as np
import time

if not 'socket' in globals():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind('tcp://*:8134')

def send_pose(RT_correct):
    assert RT_correct.dtype == np.float32
    assert RT_correct.shape == (4,4)
    socket.send_json(('pose',RT_correct.tolist()))

def poll():
    while True:
        try:
            cmd = socket.recv_json(zmq.NOBLOCK)
            if cmd == 'depth':
                opennpy.sync_update()
                depth,_ = opennpy.sync_get_depth()
                socket.send_json(('depth',depth.tolist()))
        except zmq.ZMQError, e:
            if e.errno == zmq.EAGAIN: break
            else: raise

if __name__ == '__main__':
    print 'Running 0mq server on tcp://*:8134'
    while True:
        poll()
        send_pose(np.eye(4).astype('f'))
        time.sleep(0.033)