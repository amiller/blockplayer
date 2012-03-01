Building
========
The blockplayer project is intended to be run from a command line in the current working directory. The following command builds the necessary cython files in place.

    python setup.py build_ext --inplace

It is also possible to build the normal way.


Dependencies
============
Blockplayer has several library dependencies that may be difficult to satisfy on your system. The script <code>vmdist/install_vm.sh</code> is the best reference for how to set it up. In some cases the script refers to the most recent version of a project, therefore the script may not be 'future-proofed'. The virtual machine image available at http://isue-server.eecs.ucf.edu/amillervr2012/ is preinstalled with a working combination of all dependencies.


Running BlockPlayer on a headless machine
=========================================

To run the experiment and display the results on a headless machine, you should start an X virtual server with
    Xvfb

If a graphics card isn't available, then you should specify the mesa (rather than nvidia) OpenGL drivers.
    LD_PRELOAD=/usr/lib/mesa/libGL.so xvfb-run bash


Reproducibility Kit
=================================

In addition to a source code distribution, BlockPlayer comes with a VirtualBox machine image (*.vdi) which has all necessary dependencies pre-installed.


Building the Reproducibility Kit
--------------------------------
Instructions for building the reproducibility kit are given below.


Requirements:
- VirtualBox OSE
- An Ubuntu install media (ubuntu-11.10-x86.iso)


1. vmdist/make_vm.sh
2. vmdist/restore_vm.sh
1. vmdist/install_vm.sh installs all dependencies. It is meant to be run from a stock Ubuntu 11.10, and has been tested on a VirtualBox VM.


TODO
- Instructions for downloading a dataset
- Run tests
- Run the experiment (experiments/make_output.py)
- Run the output comparison (experiments/exp_gterr.py)
- Generate the report (makewww/make_grid)
