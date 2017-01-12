#!/bin/bash
source ./common.sh

NPROC=1
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  export CXX="g++-4.8" CC="gcc-4.8" FC="gfortran-4.8"
  export PYTHONPATH=/opt/python/2.7.12/lib/python2.7/site-packages
  export PYTHONHOME=/opt/python/2.7.12
  export PATH=/opt/python/2.7.12/bin:${PATH}
  cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DON_TRAVIS=ON -DON_COVERALLS=ON -DCOVERALLS_UPLOAD=ON ${EXTRA_CMAKE_OPTS}
  NRPOC=`nproc`
  make -j $NPROC
  make coveralls
  sudo make install
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  export PYTHONPATH=/usr/local/lib/python2.7/site-packages
  cmake .. -DON_TRAVIS=ON -DON_COVERALLS=ON -DCOVERALLS_UPLOAD=ON ${EXTRA_CMAKE_OPTS}
  NPROC=`sysctl -n hw.ncpu`
  make -j $NPROC
fi
