# This script is intended to be idempotent - you should be able to run it multiple times and it will
# still work. However it's not necessarily optimized for that, it might duplicate work each time.

set -e

# Useful linux things
sudo apt-get install -y emacs git build-essential cmake subversion \
python2.6 python-setuptools python-dev \
gfortran libatlas-base-dev \
libxft2-dev libgtk2.0-dev python-gtk2-dev python-wxgtk2.8 \
libusb-1.0 libglut3-dev libxmu-dev

# Numpy
set +e; git clone git://github.com/numpy/numpy.git numpy; set -e
set +e; ln -s /usr/local/lib/python2.6/dist-packages/numpy/core/include/numpy /usr/local/include/; set -e
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
echo "backend: WX" >> ~/.matplotlib/matplotlibrc

# Other python libraries
sudo easy_install ipython nose PyOpenGL cython django

# Libfreenect
set +e; git clone https://github.com/OpenKinect/libfreenect.git; set -e
set +e; sudo echo "/usr/local/lib/" > /etc/ld.so.conf.d/local.conf; set -e
pushd libfreenect
cmake . -DBUILD_PYTHON=On
make
sudo make install
popd

# BlockPlayer
set +e; git clone git://github.com/amiller/blockplayer.git; set -e
echo "export PYTHONPATH=\$PYTHONPATH:/home/user/blockplayer" >> ~/.bashrc

# OpenCL
# Pointers here: http://www.luxrender.net/wiki/Building_on_Ubuntu_10.10
wget -N http://orwell.fiit.stuba.sk/~nou/ati-opencl-runtime_2.3_i386.deb
wget -N http://orwell.fiit.stuba.sk/~nou/ati-opencl-dev_2.3.deb
sudo dpkg -i ati-opencl-runtime_2.3_i386.deb
sudo dpkg -i ati-opencl-dev_2.3.deb

# Opencv
svn co https://code.ros.org/svn/opencv/trunk/opencv
pushd opencv
  cmake . -DBUILD_REFMAN=Off
  make
  sudo make install
popd
