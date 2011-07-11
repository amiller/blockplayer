cdef extern from *:
    void glx_init()
    void glx_printinfo()
    void glx_makecurrent()
    void glx_destroy()

def init():
    glx_init()

def printinfo():
    glx_printinfo()

def makecurrent():
    glx_makecurrent()

def destroy():
    glx_destroy()
