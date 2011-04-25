import scipy
import numpy as np
import calibkinect 
import grid

u,v = np.mgrid[:480,:640]

def project(depth, u, v):
  X,Y,Z = u,v,depth
  mat = np.dot(calibkinect.uv_matrix(), calibkinect.xyz_matrix())
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w

colormap = [
  [0,1,0],
  [0,0,1],
  [1,1,0],
  [1,0,0],
]
  
def project_colors(depth, rgb, rect):
  (l,t),(r,b) = rect
  v,u = np.mgrid[t:b,l:r].astype('f')
  global uv
  # Project the duv matrix into U,V rgb coordinates using
  # rgb_matrix() and xyz_matrix()
  uv = project(depth[t:b,l:r], u, v)[::-1]
  
  return [scipy.ndimage.map_coordinates(rgb[:,:,i].astype('f'), uv,
                                        order=0)
          for i in range(3)]


def initialize():
  global cgrid
  cgrid = None


def update_colors(depth, rgb, rect):
  coords = grid.coords
  depthB = grid.depthB
  
  (L,T),(R,B) = rect
  L,T,R,B = map(int, (L,T,R,B))

  #print depth.mean(), occ_grid.mean(), rect, depthB.mean()
  assert coords.dtype == np.uint8
  assert depthB.dtype == np.float32
  assert depth.dtype == np.uint16
  assert rgb.dtype == np.uint8
  assert coords.shape[2] == 4
  assert coords.shape[:2] == depthB.shape == depth.shape == (480,640)

  gridlen = grid.gridmax - grid.gridmin

  global bR,bG,bB
  bR = np.zeros_like(grid.vote_grid).astype('f')
  bG = np.zeros_like(grid.vote_grid).astype('f')
  bB = np.zeros_like(grid.vote_grid).astype('f')

  zRGB = project_colors(depth, rgb, rect)
  iR,iG,iB = [np.empty(depthB.shape,dtype='f') for _ in range(3)]
  iR[T:B,L:R], iG[T:B,L:R], iB[T:B,L:R] = zRGB

  import scipy.weave
  code = """
    for (int i = T; i < B; i++) {
      for (int j = L; j < R; j++) {
        int ind = i*640+j;
        int dB = depthB[ind];
        int d = depth[ind];

        if (dB<2047 && (dB-10 < d && d < dB+10)) {
           int x = coords[ind*4+0];
           int y = coords[ind*4+1];
           int z = coords[ind*4+2];
           int coord = gridlen[1]*gridlen[2]*x + gridlen[2]*y + z;
           bR[coord] += iR[ind];
           bG[coord] += iG[ind];
           bB[coord] += iB[ind];
        }           
      }
    }
    """
  scipy.weave.inline(code, ['bR','bG','bB',
                            'iR','iG','iB',
                            'coords',
                            'depthB', 'depth',
                            'gridlen',
                            'T','B','L','R'])

  global bRGB
  bRGB = (np.array((bR,bG,bB))/grid.b_total.reshape(1,*bR.shape)).astype('u1')
  bRGB = np.rollaxis(bRGB,0,4)
  return bRGB


def choose_colors(R,G,B):
  c1 = np.argmax((R,G,B),0)-1
  c2 = (B*10>G)+2
  c = [c1<0]*c2 + [c1>=0]*c1
  return c.squeeze()
