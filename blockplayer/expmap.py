import numpy as np
import numpy
import math
## Functions supporting exponential map representation
# i.e., rodrigues.m in TOOLBOX_calib

def axis2rot(axis):
	# Return the rotation matrix for this axis-angle/expmap
	h = np.sqrt(np.sum(axis**2))
	if h == 0: return np.eye(3)
	axis = axis / h
	s,c = np.sin(h), np.cos(h)
	ux,uy,uz = axis
	R = np.array([
		[ux*ux+(1-ux*ux)*c,  ux*uy*(1-c)-uz*s,  ux*uz*(1-c)+uy*s],
		[ux*uy*(1-c)+uz*s,   uy*uy+(1-uy*uy)*c, uy*uz*(1-c)-ux*s],
		[ux*uz*(1-c)-uy*s,   uz*uy*(1-c)+ux*s,  uz*uz+(1-uz*uz)*c]])
	return R

def euler2rot(euler):
  h1, h2, h3 = euler
  c,s = np.cos, np.sin
  Rx = np.array([[ 1,     0,     0],
                 [ 0, c(h1), -s(h1)],
                 [ 0, s(h1),  c(h1)]])
  Ry = np.array([[ c(h2), 0, s(h2)],
                 [0,      1, 0    ],
                 [-s(h2), 0, c(h2)]])
  Rz = np.array([[c(h3), -s(h3), 0],
                 [s(h3),  c(h3), 0],
                 [    0,      0, 1]])
  return np.dot(np.dot(Rx, Ry), Rz)

### Check
# rot * rot2axis(rot) == rot2axis(rot)
# axis2rot(rot2axis(rot)) == rot


"""
# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2011, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""

def rot2axis(matrix):
  """Return rotation angle and axis from rotation matrix.

  >>> angle = (random.random() - 0.5) * (2*math.pi)
  >>> direc = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> angle, direc, point = rotation_from_matrix(R0)
  >>> R1 = rotation_matrix(angle, direc, point)
  >>> is_same_transform(R0, R1)
  True

  """
  R = numpy.array(matrix, dtype=numpy.float32)
  R33 = R[:3, :3]
  # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
  l, W = numpy.linalg.eig(R33.T)
  i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-6)[0]
  if not len(i):
      raise
      return np.array([0,0,0],np.float32)
      #raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
  direction = numpy.real(W[:, i[-1]]).squeeze()
  # point: unit eigenvector of R33 corresponding to eigenvalue of 1
  l, Q = numpy.linalg.eig(R)
  i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-6)[0]
  if not len(i):
      raise
      #return np.array([0,0,0],np.float32)
  point = numpy.real(Q[:, i[-1]]).squeeze()
  #point /= point[3]
  # rotation angle depending on direction
  cosa = (numpy.trace(R33) - 1.0) / 2.0
  if abs(direction[2]) > 1e-6:
      sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
  elif abs(direction[1]) > 1e-6:
      sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
  else:
      sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
  angle = math.atan2(sina, cosa)
  return angle * direction
