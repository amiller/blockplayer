from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(name='BlockPlayer',
      version='0.01',
      packages=['blockplayer'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("glxcontext", ["glxcontext/glxcontext.pyx",
                                            "glxcontext/_glxcontext.c"],
                             libraries=['X11','GL'])])
