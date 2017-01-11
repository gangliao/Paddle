# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)
INCLUDE(python_module)

FIND_PACKAGE(PythonInterp 2.7)
FIND_PACKAGE(PythonLibs 2.7)

SET(py_env "")

IF(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
    find_python_module(pip REQUIRED)
    find_python_module(numpy REQUIRED)
    find_python_module(wheel REQUIRED)
    find_python_module(google.protobuf REQUIRED)
    FIND_PACKAGE(NumPy REQUIRED)
ELSE(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
    ##################################### ncurses #######################################
    SET(NCURSES_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncurses)
    SET(NCURSES_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/install/ncurses)
    ExternalProject_Add(ncurses
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${NCURSES_SOURCES_DIR}
        URL                 http://ftp.gnu.org/gnu/ncurses/ncurses-6.0.tar.gz
        CONFIGURE_COMMAND   <SOURCE_DIR>/configure --prefix=${NCURSES_INSTALL_DIR}
        UPDATE_COMMAND      ""
        BUILD_IN_SOURCE     1
    )
    ####################################################################################

    #################################### readline #######################################
    SET(READLINE_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/readline)
    SET(READLINE_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/install/readline)
    ExternalProject_Add(readline
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${READLINE_SOURCES_DIR}
        URL                 http://ftp.gnu.org/gnu/readline/readline-7.0.tar.gz
        CONFIGURE_COMMAND   <SOURCE_DIR>/configure --prefix=${READLINE_INSTALL_DIR}
        UPDATE_COMMAND      ""
        BUILD_IN_SOURCE     1
        DEPENDS ncurses
    )
    ####################################################################################

    ##################################### PYTHON ########################################
    SET(PYTHON_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/python)
    SET(PYTHON_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/install/python)
    SET(_python_DIR ${PYTHON_INSTALL_DIR})

    IF(UNIX)
        SET(PYTHON_FOUND ON)
        SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include/python2.7" CACHE PATH "Python include dir" FORCE)
        SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/lib/libpython2.7.so" CACHE FILEPATH "Python library" FORCE)
        SET(PYTHON_EXECUTABLE ${PYTHON_INSTALL_DIR}/bin/python CACHE FILEPATH "Python executable" FORCE)
        SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/lib/python2.7/site-packages" CACHE PATH "Python site-packages path" FORCE)
    ELSEIF(WIN32)
        SET(PYTHON_FOUND ON)
        SET(PYTHON_INCLUDE_DIR "${PYTHON_INSTALL_DIR}/include" CACHE PATH "Python include dir" FORCE)
        SET(PYTHON_LIBRARIES "${PYTHON_INSTALL_DIR}/libs/python27.dll" CACHE FILEPATH "Python library" FORCE)
        SET(PYTHON_EXECUTABLE "${PYTHON_INSTALL_DIR}/bin/python.exe" CACHE FILEPATH "Python executable" FORCE)
        SET(PY_SITE_PACKAGES_PATH "${PYTHON_INSTALL_DIR}/Lib/site-packages" CACHE PATH "Python site-packages path" FORCE)
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown system !")
    ENDIF()

    ExternalProject_Add(python
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${PYTHON_SOURCES_DIR}
        URL                 https://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
        CONFIGURE_COMMAND   env PATH=${NCURSES_INSTALL_DIR}/bin/:${READLINE_INSTALL_DIR}/bin/:$ENV{PATH}
                            LD_LIBRARY_PATH=${NCURSES_INSTALL_DIR}/lib/:${NCURSES_INSTALL_DIR}/lib64/:${READLINE_INSTALL_DIR}/lib/:${READLINE_INSTALL_DIR}/lib64/:$ENV{LD_LIBRARY_PATH}
                            DYLD_LIBRARY_PATH=${NCURSES_INSTALL_DIR}/lib/:${NCURSES_INSTALL_DIR}/lib64/:${READLINE_INSTALL_DIR}/lib/:${READLINE_INSTALL_DIR}/lib64/:$ENV{DYLD_LIBRARY_PATH}
                            <SOURCE_DIR>/configure --enable-shared --prefix=${PYTHON_INSTALL_DIR}
        UPDATE_COMMAND      ""
        BUILD_IN_SOURCE     1
        DEPENDS             ncurses readline zlib
    )

    SET(py_env
        PATH=${PYTHON_INSTALL_DIR}/bin:$ENV{PATH}
        CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
        LD_LIBRARY_PATH=${PYTHON_INSTALL_DIR}/lib/:$ENV{LD_LIBRARY_PATH}
        DYLD_LIBRARY_PATH=${PYTHON_INSTALL_DIR}/lib/:$ENV{DYLD_LIBRARY_PATH}
        PYTHONHOME=${PYTHON_INSTALL_DIR}
        PYTHONPATH=${PYTHON_INSTALL_DIR}/lib/python2.7:${PYTHON_INSTALL_DIR}/lib/python2.7/lib-dynload:${PY_SITE_PACKAGES_PATH})
    ####################################################################################

    ##################################### SETUPTOOLS ###################################
    SET(SETUPTOOLS_SOURCES_DIR ${PYTHON_SOURCES_DIR}/setuptools)
    ExternalProject_Add(setuptools
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${SETUPTOOLS_SOURCES_DIR}
        URL                 "https://pypi.python.org/packages/source/s/setuptools/setuptools-18.3.2.tar.gz"
        BUILD_IN_SOURCE     1
        PATCH_COMMAND       ""
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
        INSTALL_COMMAND     ""
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS             python zlib
    )
    #####################################################################################

    ##################################### SIX ###########################################
    SET(SIX_SOURCES_DIR ${PYTHON_SOURCES_DIR}/six)
    ExternalProject_Add(six
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX              ${SIX_SOURCES_DIR}
        URL                 https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz
        BUILD_IN_SOURCE     1
        PATCH_COMMAND       ""
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
        INSTALL_COMMAND     ""
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS             python setuptools
    )
    #####################################################################################

    ##################################### CYTHON ########################################
    SET(CYTHON_SOURCES_DIR ${PYTHON_SOURCES_DIR}/cython)
    ExternalProject_Add(cython
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX                ${CYTHON_SOURCES_DIR}
        URL                   https://github.com/cython/cython/archive/0.25.2.tar.gz
        GIT_TAG               0.25.2
        BUILD_IN_SOURCE       1
        CONFIGURE_COMMAND     ""
        PATCH_COMMAND         ""
        UPDATE_COMMAND        ""
        INSTALL_COMMAND       ""
        BUILD_COMMAND         env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS               python
    )
    ####################################################################################

    ##################################### NUMPY ########################################
    SET(NUMPY_SOURCES_DIR ${PYTHON_SOURCES_DIR}/numpy)
    SET(NUMPY_TAG_VERSION "v1.11.3")
    SET(NUMPY_VERSION "1.11.3")

    SET(EGG_NAME "")
    SET(PYTHON_NUMPY_INCLUDE_DIR "")
    IF(WIN32)
        SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-${HOST_SYSTEM}.egg")
    ELSE(WIN32)
        IF(APPLE)
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-${HOST_SYSTEM}-${MACOS_VERSION}")
        ELSE(APPLE)
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-linux")
            SET(EGG_NAME "numpy-${NUMPY_VERSION}-py2.7-linux")
        ENDIF(APPLE)

        FOREACH(suffix x86_64 intel fat64 fat32 universal)
            LIST(APPEND PYTHON_NUMPY_INCLUDE_DIR ${PY_SITE_PACKAGES_PATH}/${EGG_NAME}-${suffix}.egg/numpy/core/include)
        ENDFOREACH()
    ENDIF(WIN32)

    ExternalProject_Add(numpy
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/numpy/numpy.git
        GIT_TAG             ${NUMPY_TAG_VERSION}
        CONFIGURE_COMMAND   ""
        UPDATE_COMMAND      ""
        PREFIX              ${NUMPY_SOURCES_DIR}
        BUILD_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py build
        INSTALL_COMMAND     env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        BUILD_IN_SOURCE     1
        DEPENDS             python setuptools cython
    )
    ####################################################################################

    ##################################### WHEEL ########################################
    SET(WHEEL_SOURCES_DIR ${PYTHON_SOURCES_DIR}/wheel)
    ExternalProject_Add(wheel
        ${EXTERNAL_PROJECT_LOG_ARGS}
        URL                 https://pypi.python.org/packages/source/w/wheel/wheel-0.29.0.tar.gz
        PREFIX              ${WHEEL_SOURCES_DIR}
        CONFIGURE_COMMAND   ""
        UPDATE_COMMAND      ""
        BUILD_COMMAND       ""
        INSTALL_COMMAND     env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        BUILD_IN_SOURCE     1
        DEPENDS             python setuptools
    )
    ####################################################################################

    ################################### PROTOBUF #######################################
    SET(PY_PROTOBUF_SOURCES_DIR ${PYTHON_SOURCES_DIR}/protobuf)
    ExternalProject_Add(python-protobuf
        ${EXTERNAL_PROJECT_LOG_ARGS}
        URL                   https://pypi.python.org/packages/e0/b0/0a1b364fe8a7d177b4b7d4dca5b798500dc57a7273b93cca73931b305a6a/protobuf-3.1.0.post1.tar.gz
        URL_MD5               38b5fb160c768d2f8444d0c6d637ff91
        PREFIX                ${PY_PROTOBUF_SOURCES_DIR}
        BUILD_IN_SOURCE       1
        PATCH_COMMAND         ""
        CONFIGURE_COMMAND     ""
        UPDATE_COMMAND        ""
        BUILD_COMMAND         env ${py_env} ${PYTHON_EXECUTABLE} setup.py build
        INSTALL_COMMAND       env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
        DEPENDS               python setuptools six
    )
    ####################################################################################

    LIST(APPEND external_project_dependencies python setuptools six cython wheel python-protobuf numpy)

ENDIF(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_NUMPY_INCLUDE_DIR})
