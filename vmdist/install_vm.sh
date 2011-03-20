# This script is intended to be idempotent - you should be able to run it multiple times and it will
# still work. However it's not necessarily optimized for that, it might duplicate work each time.

set -e

# Useful linux things
sudo apt-get update
sudo apt-get install -y emacs23-nox git build-essential cmake subversion \
python2.6 python-setuptools python-dev \
gfortran libatlas-base-dev \
libxft2-dev libgtk2.0-dev python-gtk2-dev python-wxgtk2.8 \
libusb-1.0 libglut3-dev libxmu-dev xvfb libgl1-mesa-dev libglu1-mesa-dev \
libx11-dev lighttpd

# Numpy
set +e; git clone git://github.com/numpy/numpy.git numpy; set -e
sudo ln -fs /usr/local/lib/python2.6/dist-packages/numpy/core/include/numpy /usr/local/include/
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
set +e; mkdir ~/.matplotlib; set -e
echo "backend: WX" >> ~/.matplotlib/matplotlibrc


# OpenCL
# Pointers here: http://www.luxrender.net/wiki/Building_on_Ubuntu_10.10
wget -N http://orwell.fiit.stuba.sk/~nou/ati-opencl-runtime_2.3_i386.deb
wget -N http://orwell.fiit.stuba.sk/~nou/ati-opencl-dev_2.3.deb
sudo dpkg -i ati-opencl-runtime_2.3_i386.deb
sudo dpkg -i ati-opencl-dev_2.3.deb


# Other python libraries
sudo easy_install ipython nose PyOpenGL cython django pyopencl

# Libfreenect
set +e; git clone https://github.com/OpenKinect/libfreenect.git; set -e
set +e; sudo bash -c 'echo "/usr/local/lib/" > /etc/ld.so.conf.d/local.conf'; set -e
sudo ldconfig
pushd libfreenect
cmake . -DBUILD_PYTHON=On
make
sudo make install
popd


# Opencv
svn co https://code.ros.org/svn/opencv/trunk/opencv
sudo ln -fs /usr/local/lib/python2.6/dist-packages /usr/local/lib/python2.6/site-packages
pushd opencv
  cmake . -DBUILD_REFMAN=Off
  make
  sudo make install
popd


# BlockPlayer
set +e; git clone git@github.com:amiller/blockplayer.git; set -e
pushd blockplayer
./download.sh
python setup.py build
python setup.py build_ext --inplace
sudo python setup.py install
xvfb-run python makewww/make_normals.py
xvfb-run python makewww/make_calib.py
echo "Running lighttpd -Df lighttpd.conf"
echo "View the results at localhost:8090"
echo "(this won't return, ctrl+c if you like)"
sudo lighttpd -Df lighttpd.conf
popd
