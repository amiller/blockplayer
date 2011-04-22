cdef extern from *:
     void glx_init()
     void glx_printinfo()
     void glx_makecurrent()

def init():
    glx_init()

def printinfo():
    glx_printinfo()

def makecurrent():
    glx_makecurrent()