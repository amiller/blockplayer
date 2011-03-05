set -e

# Useful linux things
sudo apt-get install -y emacs git build-essential cmake subversion \
python2.6 python-setuptools python-dev \
gfortran libatlas-base-dev \
libxft2-dev libgtk2.0-dev python-gtk2-dev python-wxgtk2.8

# Numpy
set +e; git clone git://github.com/numpy/numpy.git numpy; set -e
pushd numpy; 
  python setup.py build && sudo python setup.py install
popd

# Scipy
svn co http://svn.scipy.org/svn/scipy/trunk scipy
pushd scipy
  python setup.py build && sudo python setup.py install
popd

# Matplotlib
svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib matplotlib
pushd matplotlib
  python setup.py build && sudo python setup.py install
popd

# Other python libraries
sudo easy_install ipython nose

# BlockPlayer
set +e; git clone git://github.com/amiller/blockplayer.git; set -e

# OpenCL
# Pointers here: http://www.luxrender.net/wiki/Building_on_Ubuntu_10.10
wget http://orwell.fiit.stuba.sk/~nou/ati-opencl-runtime_2.3_i386.deb
wget http://orwell.fiit.stuba.sk/~nou/ati-opencl-dev_2.3.deb
sudo dpkg -i ati-opencl-runtime_2.3_i386.deb
sudo dpkg -i ati-opencl-dev_2.3.deb

# Opencv
svn co https://code.ros.org/svn/opencv/trunk/opencv
pushd opencv
  cmake .
  make
  sudo make install
popd
