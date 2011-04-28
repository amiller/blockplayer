from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("blockplayer.glxcontext",
                       ["blockplayer/glxcontext/_glxcontext.c",
                       "blockplayer/glxcontext/glxcontext.pyx"],
                       libraries=['X11','GL']),
             Extension("blockplayer.calibkinect_cy",
                       ["blockplayer/calibkinect_cy.pyx"]),
             Extension("blockplayer.speedup_ctypes",
                       ["blockplayer/speedup_ctypes.c"])]


setup(name='BlockPlayer',
      version='0.01',
      packages=['blockplayer','blockplayer.visuals'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
