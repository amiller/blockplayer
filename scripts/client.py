"""
scripts/client.py - a simple client used to test the output-server
"""

import zmq
import numpy as np
import time
import cv

if not 'socket' in globals():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect('tcp://*:8134')

def poll():
    while True:
        try:
            kind, data = socket.recv_json(zmq.NOBLOCK)
            if kind == 'depth':
                depth = np.array(data, 'u2')
                cv.ShowImage('depth', cv.fromarray(depth.astype('f')/1024.))
                print 'Received depth', depth.shape, depth.dtype
            if kind == 'rgb':
                rgb = np.array(data, 'u1')
                cv.ShowImage('rgb', cv.fromarray(rgb))
                print 'Received rgb', rgb.shape, rgb.dtype
            if kind == 'pose':
                pose = np.array(data, 'f')
                assert pose.shape == (4,4)
                print 'Received pose', pose.shape, pose.dtype
            if kind == 'voxels':
                voxels = np.array(data, bool)
                assert len(np.array(data).shape) == 3
                print 'Received voxels', voxels.shape, time.time(), voxels.sum()
            if kind == 'voxels_colors':
                shape = data[0]
                voxels = data[1]
                assert len(shape) == 3
                assert shape[0]*shape[1]*shape[2] == len(voxels)
                print 'Received color voxels', shape, time.time(), voxels
        except zmq.ZMQError, e:
            if e.errno == zmq.EAGAIN: break
            else: raise

def get_depth():
    socket.send_json("depth")
    time.sleep(0.2)
    poll()

def get_video():
    socket.send_json("rgb")
    time.sleep(0.2)
    poll()

if __name__ == '__main__':
    print 'Connecting 0mq client to tcp://*:8134'
    print 'Commands to try: get_depth(), poll(), go()'
