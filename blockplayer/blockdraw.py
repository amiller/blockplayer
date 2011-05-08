import numpy as np
from OpenGL.GL import *

if not 'blocks' in globals():
    blocks = {}


def clear():
    global blocks
    blocks = {}


def show_grid(name, grid,
              color=np.array([1,0,0,1]), opacity=1.0,
              line_color=np.array([1,1,1,1])):
    assert color is None or color.shape == (4,) or color.shape[3]==3
    d = {}
    if color.shape == (4,):
        d.update(grid_vertices(grid, None))
    else:
        d.update(grid_vertices(grid, color))
    d['solid_color'] = color if color.shape == (4,) else None
    d['line_color'] = line_color
    global blocks
    blocks[name] = d


def draw_block(blocks):

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointeri(blocks['vertices'])
    # glColor(0.3,0.3,0.3)

    if not blocks['solid_color'] is None:
        glColor(*blocks['solid_color'])
    else:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerub(blocks['color'])

    glDrawElementsui(GL_QUADS, blocks['quad_inds'])
    glDisableClientState(GL_COLOR_ARRAY)

    glColor(*blocks['line_color'])
    glDrawElementsui(GL_LINES, blocks['line_inds'])
    glDisableClientState(GL_VERTEX_ARRAY)


def grid_vertices(grid, color=None):
    return grid_vertices_numpy(grid, color)


def grid_vertices_numpy(grid, color=None):
    """
    Given a boolean voxel grid, produce a list of vertices and indices
    for drawing quads or line strips in opengl
    """
    q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
         [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
         [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
         [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
         [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
         [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]

    normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2]))
              for qz in q]

    blocks = np.array(grid.nonzero()).transpose().reshape(-1,1,3)
    q = np.array(q).reshape(1,-1,3)

    vertices = (q + blocks).reshape(-1,3)
    coords = (q*0 + blocks).astype('u1').reshape(-1,3)

    if not color is None:
        assert color.shape[3] == 3
        color = color[grid,:].reshape(-1,1,3)
        cc = (q.astype('u1')*0+color).reshape(-1,3)
        assert cc.dtype == np.uint8
    else:
        cc = coords

    normals = np.tile(normal, (len(blocks),4)).reshape(-1,3)
    line_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
    quad_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,2,3]

    return dict(blocks=blocks, vertices=vertices, coords=coords,
                normals=normals, line_inds=line_inds, quad_inds=quad_inds,
                color=cc)


def draw():
    global blocks
    for block in blocks.values():
        draw_block(block)
