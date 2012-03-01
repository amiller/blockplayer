# This script is intended to be idempotent - you should be able to run it multiple times and it will
# still work. However it's not necessarily optimized for that, it might duplicate work each time.

set -e

# Useful linux things
sudo apt-get update
sudo apt-get install -y emacs23-nox git build-essential cmake subversion \
python python-distribute python-dev python-numpy python-scipy python-matplotlib \
libxft2-dev libgtk2.0-dev python-gtk2-dev python-wxgtk2.8 \
libusb-1.0 freeglut3-dev libxmu-dev xvfb libgl1-mesa-dev libglu1-mesa-dev \
libx11-dev lighttpd openjdk-7-jdk doxygen curl

set +e; mkdir ~/.matplotlib; set -e
echo "backend: WX" >> ~/.matplotlib/matplotlibrc


# OpenCL
# Pointers here: http://www.luxrender.net/wiki/Building_on_Ubuntu_10.10
wget -N https://s3.amazonaws.com/blockplayer/ati-opencl-runtime_2.3_i386.deb
wget -N https://s3.amazonaws.com/blockplayer/ati-opencl-dev_2.3.deb
sudo dpkg -i ati-opencl-runtime_2.3_i386.deb
sudo dpkg -i ati-opencl-dev_2.3.deb


# Other python libraries
sudo easy_install -U distribute
sudo easy_install ipython nose PyOpenGL cython django pyopencl


# OpenNI / Sensor2
set +e; git clone https://github.com/OpenNI/OpenNI.git; set -e
pushd OpenNI/Platform/Linux/CreateRedist
git checkout unstable
bash ./RedistMaker
cd ../Redist/OpenNI*
sudo bash ./install.sh
popd

set +e; git clone https://github.com/avin2/SensorKinect.git; set -e
pushd SensorKinect/Platform/Linux/CreateRedist
git checkout unstable
bash ./RedistMaker
cd ../Redist/Sensor*
sudo bash ./install.sh
popd

# OpenNPy
set +e; git clone https://github.com/bwhite/opennpy.git; set -e
pushd opennpy
python setup.py build && sudo python setup.py install
popd

# GlxContext
set +e; git clone https://github.com/amiller/glxcontext; set -e
pushd glxcontext
python setup.py build && sudo python setup.py install
popd

# Wxpy3d
set +e; git clone https://github.com/amiller/wxpy3d; set -e
pushd wxpy3d
python setup.py build && sudo python setup.py install
popd


# Opencv
svn co https://code.ros.org/svn/opencv/trunk/opencv
PYTHON=python2.7
sudo mv /usr/local/lib/$PYTHON/site-packages /usr/local/lib/$PYTHON/backup.site-packages
sudo ln -fsT /usr/local/lib/$PYTHON/dist-packages /usr/local/lib/$PYTHON/site-packages
pushd opencv
  cmake . -DBUILD_REFMAN=Off
  make
  sudo make install
popd


# BlockPlayer
set +e; git clone git://github.com/amiller/blockplayer.git; set -e
pushd blockplayer
python setup.py build
python setup.py build_ext --inplace
sudo python setup.py install
./download.sh
xvfb-run python experiments/make_output.py
xvfb-run python makewww/make_grid
xvfb-run python makewww/make_calib.py
echo "Running lighttpd -Df lighttpd.conf"
echo "View the results at localhost:8090"
echo "(this won't return, ctrl+c if you like)"
sudo lighttpd -Df lighttpd.conf
popd
