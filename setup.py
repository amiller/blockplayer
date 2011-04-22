from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("blockplayer.glxcontext",
                       ["blockplayer/glxcontext/_glxcontext.c",
                       "blockplayer/glxcontext/glxcontext.pyx"],
                       libraries=['X11','GL']),
             Extension("ntk.ntk",
                       ["ntk/ntk.pyx"],
                       language='c++',
                       include_dirs=['ntk/nestk/deps/openni/Include',
                                     'ntk/nestk/deps/openni/Nite/Include'],
                       runtime_library_dirs=['ntk/','ntk/lib'],
                       library_dirs=['ntk/'],
                       #extra_objects=['ntk/lib_ntk.a'],
                       libraries=['XnDevicesSensorV2','_ntk']),
             Extension("blockplayer.speedup_ctypes",
                       ["blockplayer/speedup_ctypes.c"])]


setup(name='BlockPlayer',
      version='0.01',
      packages=['blockplayer','blockplayer.visuals'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
