from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("blockplayer.calibkinect_cy",
                       ["blockplayer/calibkinect_cy.pyx"]),
             Extension("blockplayer.speedup_cy",
                       ["blockplayer/speedup_cy.pyx"]),
             Extension("blockplayer.speedup_ctypes",
                       ["blockplayer/speedup_ctypes.c"])]


setup(name='BlockPlayer',
      version='0.1',
      author='Andrew Miller',
      email='amiller@cs.ucf.edu',
      packages=['blockplayer'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      install_requires=['distribute', 'cython', 'pyopencl', 'PyOpenGL', 'numpy', 'scipy'],
      dependency_links = [
        "https://github.com/amiller/wxpy3d/tarball/master#egg=wxpy3d-1.0",
        "https://github.com/amiller/glxcontext/tarball/master#egg=glxcontext-1.0",
        ])
