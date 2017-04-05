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

SET(CLAPACK_SOURCES_DIR ${THIRD_PARTY_PATH}/clapack)
SET(CLAPACK_INSTALL_DIR ${THIRD_PARTY_PATH}/install/clapack)
SET(CLAPACK_INCLUDE_DIR "${CLAPACK_INSTALL_DIR}/include" CACHE PATH "clapack include directory." FORCE)

SET(CLAPACK_LIBS_DIR "${CLAPACK_INSTALL_DIR}/lib")
SET(CLAPACK_LIBRARIES
    ${CLAPACK_LIBS_DIR}/liblapack.a
    ${CLAPACK_LIBS_DIR}/libf2c.a
    ${CLAPACK_LIBS_DIR}/libcblaswr.a
    CACHE FILEPATH "CLAPACK_LIBRARIES" FORCE)

INCLUDE_DIRECTORIES(${CLAPACK_INCLUDE_DIR})

ExternalProject_Add(
    clapack
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/Xreki/clapack.git"
    PREFIX          ${CLAPACK_SOURCES_DIR}
    UPDATE_COMMAND  ""
    INSTALL_COMMAND ""
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${CLAPACK_INSTALL_DIR}
    CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    CMAKE_ARGS      -DBUILD_SINGLE=OFF
    CMAKE_ARGS      -DBUILD_COMPLEX=OFF
    CMAKE_ARGS      -DBUILD_COMPLEX16=OFF
    CMAKE_ARGS      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    CMAKE_ARGS      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

ExternalProject_Add_Step(
    clapack clapack_install
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack/INCLUDE/blaswrap.h" "${CLAPACK_INCLUDE_DIR}/blaswrap.h"
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack/INCLUDE/clapack.h" "${CLAPACK_INCLUDE_DIR}/clapack.h"
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack/INCLUDE/f2c.h" "${CLAPACK_INCLUDE_DIR}/f2c.h"                
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack-build/BLAS/WRAP/libcblaswr.a" "${CLAPACK_LIBS_DIR}/libcblaswr.a"
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack-build/F2CLIBS/libf2c/libf2c.a" "${CLAPACK_LIBS_DIR}/libf2c.a"
    COMMAND ${CMAKE_COMMAND} -E copy "${CLAPACK_SOURCES_DIR}/src/clapack-build/SRC/liblapack.a" "${CLAPACK_LIBS_DIR}/liblapack.a"
    DEPENDEES install
)

LIST(APPEND external_project_dependencies clapack)
