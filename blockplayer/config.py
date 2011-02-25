# This module will store all the configuration variables
# that are video sequence-specific and that do not
# change during the course of running a video sequence.

import numpy as np
import cPickle as pickle


def load(dir_path):
    with open('%s/config.pkl' % dir_path, 'r') as f:
        globals().update(pickle.load(f))


def save(dir_path):
  with open('%s/config.pkl' % dir_path,'w') as f:
    pickle.dump(dict(bgL=bgL,
                     bgR=bgR,
                     LH=LH,
                     LW=LW,
                     ), f)


RIGHT2LEFT = np.eye(4)


# Duplo block sizes
LH = 0.0180
LW = 0.0160

# Jenga Block sizes FIXME:(needs to be remeasured)
#LH = 0.0150
#LW = 0.0200
