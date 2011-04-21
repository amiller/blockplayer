"""
These are some functions to help work with kinect camera calibration and projective
geometry. 

Tasks:
- Convert the kinect depth image to a metric 3D point cloud
- Convert the 3D point cloud to texture coordinates in the RGB image

Notes about the coordinate systems:
 There are three coordinate systems to worry about. 
 1. Kinect depth image:
    u,v,depth
    u and v are image coordinates, (0,0) is the top left corner of the image
                               (640,480) is the bottom right corner of the image
    depth is the raw 11-bit image from the kinect, where 0 is infinitely far away
      and larger numbers are closer to the camera
      (2047 indicates an error pixel)
      
 2. Kinect rgb image:
    u,v
    u and v are image coordinates (0,0) is the top left corner
                              (640,480) is the bottom right corner
                              
 3. XYZ world coordinates:
    x,y,z
    The 3D world coordinates, in meters, relative to the depth camera. 
    (0,0,0) is the camera center. 
    Negative Z values are in front of the camera, and the positive Z direction points
       towards the camera. 
    The X axis points to the right, and the Y axis points up. This is the standard 
       right-handed coordinate system used by OpenGL.
    

"""
import numpy as np


def depth2xyzuv(depth, u=None, v=None):
  """
  Return a point cloud, an Nx3 array, made by projecting the kinect depth map 
    through intrinsic / extrinsic calibration matrices
  Parameters:
    depth - comes directly from the kinect 
    u,v - are image coordinates, same size as depth (default is the original image)
  Returns:
    xyz - 3D world coordinates in meters (Nx3)
    uv - image coordinates for the RGB image (Nx3)
  
  You can provide only a portion of the depth image, or a downsampled version of
    the depth image if you want; just make sure to provide the correct coordinates
    in the u,v arguments. 
    
  Example:
    # This downsamples the depth image by 2 and then projects to metric point cloud
    u,v = mgrid[:480:2,:640:2]
    xyz,uv = depth2xyzuv(freenect.sync_get_depth()[::2,::2], u, v)
    
    # This projects only a small region of interest in the upper corner of the depth image
    u,v = mgrid[10:120,50:80]
    xyz,uv = depth2xyzuv(freenect.sync_get_depth()[v,u], u, v)
  """
  if u is None or v is None:
    u,v = np.mgrid[:480,:640]
  
  # Build a 3xN matrix of the d,u,v data
  C = np.vstack((u.flatten(), v.flatten(), depth.flatten(), 0*u.flatten()+1))

  # Project the duv matrix into xyz using xyz_matrix()
  X,Y,Z,W = np.dot(xyz_matrix(),C)
  X,Y,Z = X/W, Y/W, Z/W
  xyz = np.vstack((X,Y,Z)).transpose()
  xyz = xyz[Z<0,:]

  # Project the duv matrix into U,V rgb coordinates using rgb_matrix() and xyz_matrix()
  U,V,_,W = np.dot(np.dot(uv_matrix(), xyz_matrix()),C)
  U,V = U/W, V/W
  uv = np.vstack((U,V)).transpose()    
  uv = uv[Z<0,:]       

  # Return both the XYZ coordinates and the UV coordinates
  return xyz, uv


def uv_matrix():
  """
  Returns a matrix you can use to project XYZ coordinates (in meters) into
      U,V coordinates in the kinect RGB image
  """

  rot = np.array([9.9998802393075870e-01, -6.2977998218913971e-04,
       4.8533877065907388e-03, 7.2759751545855894e-04,
       9.9979611553846304e-01, -2.0179146564111142e-02,
       -4.8396897536878121e-03, 2.0182436210091532e-02,
       9.9978460013730641e-01]).reshape(3,3)

  trans = np.array([[2.3531805894121169e-02, -1.3769320426104585e-03,
       1.8042422163477460e-02]])

  I = np.eye(3); I[1,1]=-1; I[2,2]=-1;
  rot = np.dot(I, np.dot(rot.transpose(), I))
  trans = np.dot(I, -trans.transpose())
  
  m = np.hstack((rot, trans))
  m = np.vstack((m, np.array([[0,0,0,1]])))
  KK = np.array([[-521, 0, 330, 0],
                 [0, 521, 272, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
  m = np.dot(KK, (m))
  return m


def xyz_matrix():
  fx = 583.0
  fy = 583.0
  cx = 321
  cy = 249
  a = -0.0028300396
  b = 3.1006268
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0,   0, 0,    -1],
                  [0,   0, a,     b]])
  return mat
