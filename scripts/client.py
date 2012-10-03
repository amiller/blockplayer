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
            if kind == 'pose':
                pass
                #print 'Received pose %d', pose.shape, pose.dtype
        except zmq.ZMQError, e:
            if e.errno == zmq.EAGAIN: break
            else: raise

def get_depth():
    socket.send_json("depth")
    time.sleep(0.2)
    poll()

if __name__ == '__main__':
    print 'Connecting 0mq client to tcp://*:8134'
    print 'Commands to try: get_depth(), poll(), go()'

