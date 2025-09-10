# Copyright 2023 Ryan Curtin (http://www.ratml.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------


## NOTE: If you prefer to link your programs _directly_ with OpenBLAS, BLAS or LAPACK,
## NOTE: do not use this installer.
## NOTE: To use Bandicoot without installation, compile your programs along these lines:
## NOTE: g++ prog.cpp -o prog -O2 -I /home/user/bandicoot-1.0.0/include -DCOOT_DONT_USE_WRAPPER -DCOOT_USE_OPENCL -larmadillo -lOpenCL -lclBLAS
## NOTE: The above command line assumes that you have unpacked the Bandicoot code archive into /home/user/
## NOTE: You will need to adjust this for newer versions of Bandicoot (ie. "1.0.0" needs to be changed),
## NOTE: and/or if you have unpacked the Bandicoot archive into a different directory.
## NOTE: On macOS, replace -lOpenCL with -framework OpenCL
## NOTE: More details: https://coot.sourceforge.net/docs.html#direct_linking


cmake_minimum_required(VERSION 3.6...3.10 FATAL_ERROR)

if(POLICY CMP0025)
  # enable differentiation between vanilla Clang and AppleClang
  cmake_policy(SET CMP0025 NEW)
  message(STATUS "*** set cmake policy CMP0025 to NEW")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "CMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}")

project(bandicoot CXX C)
include(CheckIncludeFileCXX)
include(CheckLibraryExists)

set(COOT_USE_WRAPPER true)

# the settings below will be automatically configured by the rest of this script

set(COOT_USE_OPENCL false)
set(COOT_USE_CUDA   false)

## extract version from sources

set(COOT_VERSION_FILE_NAME "${PROJECT_SOURCE_DIR}/include/bandicoot_bits/coot_version.hpp")

if(NOT EXISTS ${COOT_VERSION_FILE_NAME})
  message(FATAL_ERROR "Can't read ${COOT_VERSION_FILE_NAME}")
endif()

file(READ ${COOT_VERSION_FILE_NAME} COOT_VERSION_FILE_CONTENTS)
string(REGEX REPLACE ".*#define COOT_VERSION_MAJOR ([0-9]+).*" "\\1" COOT_VERSION_MAJOR "${COOT_VERSION_FILE_CONTENTS}")
string(REGEX REPLACE ".*#define COOT_VERSION_MINOR ([0-9]+).*" "\\1" COOT_VERSION_MINOR "${COOT_VERSION_FILE_CONTENTS}")
string(REGEX REPLACE ".*#define COOT_VERSION_PATCH ([0-9]+).*" "\\1" COOT_VERSION_PATCH "${COOT_VERSION_FILE_CONTENTS}")

message(STATUS "Configuring Bandicoot ${COOT_VERSION_MAJOR}.${COOT_VERSION_MINOR}.${COOT_VERSION_PATCH}...")

string(COMPARE EQUAL "${CMAKE_CXX_FLAGS}" "" CXX_FLAGS_EMPTY)
if(NOT CXX_FLAGS_EMPTY)
  message(STATUS "")
  message(STATUS "*** WARNING: variable 'CMAKE_CXX_FLAGS' is not empty; this may cause problems!")
  message(STATUS "")
endif()

# As Red Hat Enterprise Linux (and related systems such as Fedora)
# does not search /usr/local/lib by default, we need to place the
# library in either /usr/lib or /usr/lib64

if(NOT APPLE)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "*** CMAKE_INSTALL_PREFIX was initalised by cmake to the default value of ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "*** CMAKE_INSTALL_PREFIX changed to /usr")
    set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Standard install prefix" FORCE)
  endif()
endif()


# Adhere to GNU filesystem layout conventions
include(GNUInstallDirs)

if(MSVC)
  option(BUILD_SHARED_LIBS "build shared library" OFF)
else()
  option(BUILD_SHARED_LIBS "build shared library" ON)
endif()

option(OPENBLAS_PROVIDES_LAPACK "Assume that OpenBLAS provides LAPACK functions" OFF)
## Example use on the command line:
## cmake -D OPENBLAS_PROVIDES_LAPACK=true .

option(ALLOW_FLEXIBLAS_LINUX "Allow detection of FlexiBLAS on Linux" ON)
# set(ALLOW_FLEXIBLAS_LINUX false)
## uncomment the above line to disable the detection of FlexiBLAS;
## you can also disable FlexiBLAS detection directly on the command line:
## cmake -D ALLOW_FLEXIBLAS_LINUX=false .

option(ALLOW_OPENBLAS_MACOS "Allow detection of OpenBLAS on macOS" OFF)
## Example use on the command line:
## cmake -D ALLOW_OPENBLAS_MACOS=true .

option(ALLOW_BLAS_LAPACK_MACOS "Allow detection of BLAS and LAPACK on macOS" OFF)
## Example use on the command line:
## cmake -D ALLOW_BLAS_LAPACK_MACOS=true .

option(FIND_OPENCL "Search for an OpenCL implementation" ON)
## Example use on the command line:
## cmake -D FIND_OPENCL=false .

option(FIND_CUDA "Search for a CUDA implementation" ON)
## Example use on the command line:
## cmake -D FIND_CUDA=false .

set(DEFAULT_BACKEND "" CACHE STRING "Default backend to use; must be \"CUDA_BACKEND\" or \"CL_BACKEND\"")
## Example use on the command line:
## cmake -D DEFAULT_BACKEND="CL_BACKEND" .

option(BUILD_TESTS "Build tests" OFF)
## you can also enable building the tests directly on the command line:
## cmake -D BUILD_TESTS=true .
##
## to run the tests on the command line:
## make
## ctest


if(WIN32)
  message(STATUS "")
  message(STATUS "*** WARNING: automatic installation is experimental for this platform.")
  message(STATUS "*** WARNING: if anything breaks, you get to keep all the pieces.")
  message(STATUS "*** WARNING: manual installation is described in the README file.")
  message(STATUS "")
endif()

if(MSVC AND BUILD_SHARED_LIBS)
  message(STATUS "")
  message(STATUS "*** WARNING: building shared library with MSVC is not supported." )
  message(STATUS "*** WARNING: if anything breaks, you get to keep all the pieces.")
  message(STATUS "")
endif()


message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )

message(STATUS "")
message(STATUS "*** Options:"                                           )
message(STATUS "BUILD_SHARED_LIBS         = ${BUILD_SHARED_LIBS}"       )
message(STATUS "OPENBLAS_PROVIDES_LAPACK  = ${OPENBLAS_PROVIDES_LAPACK}")
message(STATUS "ALLOW_FLEXIBLAS_LINUX     = ${ALLOW_FLEXIBLAS_LINUX}"   )
message(STATUS "ALLOW_OPENBLAS_MACOS      = ${ALLOW_OPENBLAS_MACOS}"    )
message(STATUS "ALLOW_BLAS_LAPACK_MACOS   = ${ALLOW_BLAS_LAPACK_MACOS}" )
message(STATUS "BUILD_TESTS               = ${BUILD_TESTS}"             )

message(STATUS "")
message(STATUS "*** Looking for external libraries")

##
## Find LAPACK and BLAS libraries, or their optimised versions
##

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_aux/Modules/")
set(COOT_HAS_BLAS false)
set(COOT_HAS_LAPACK false)

if(APPLE)
  message(STATUS "Detected macOS")

  set(COOT_USE_ACCELERATE true)

  if(ALLOW_OPENBLAS_MACOS)
    include(COOT_FindOpenBLAS)
    message(STATUS "OpenBLAS_FOUND = ${OpenBLAS_FOUND}")
    message(STATUS "")
    message(STATUS "*** If use of OpenBLAS is causing problems,")
    message(STATUS "*** rerun cmake with detection of OpenBLAS disabled:")
    message(STATUS "*** cmake -D ALLOW_OPENBLAS_MACOS=false .")
    message(STATUS "")

    if(OpenBLAS_FOUND)
      set(COOT_LIBS ${COOT_LIBS} ${OpenBLAS_LIBRARIES})
      set(COOT_USE_ACCELERATE false)
      set(COOT_HAS_BLAS true)
      if(OPENBLAS_PROVIDES_LAPACK)
        set(COOT_HAS_LAPACK true)
      endif()
    endif()
  endif()

  if(ALLOW_BLAS_LAPACK_MACOS)
    include(COOT_FindBLAS)
    include(COOT_FindLAPACK)
    message(STATUS "  BLAS_FOUND = ${BLAS_FOUND}"  )
    message(STATUS "LAPACK_FOUND = ${LAPACK_FOUND}")
    message(STATUS "")
    message(STATUS "*** If use of BLAS and LAPACK is causing problems,")
    message(STATUS "*** rerun cmake with detection of BLAS and LAPACK disabled:")
    message(STATUS "*** cmake -D ALLOW_BLAS_LAPACK_MACOS=false .")
    message(STATUS "")

    if(BLAS_FOUND AND LAPACK_FOUND)
      set(COOT_LIBS ${COOT_LIBS} ${BLAS_LIBRARIES})
      set(COOT_LIBS ${COOT_LIBS} ${LAPACK_LIBRARIES})
      set(COOT_USE_ACCELERATE false)
    endif()

    if(BLAS_FOUND)
      set(COOT_HAS_BLAS true)
    endif()

    if(LAPACK_FOUND)
      set(COOT_HAS_LAPACK true)
    endif()
  endif()

  if(COOT_USE_ACCELERATE)
    # Use the Accelerate-provided BLAS and LAPACK.
    set(COOT_HAS_BLAS true)
    set(COOT_HAS_LAPACK true)

    set(COOT_LIBS ${COOT_LIBS} "-framework Accelerate")
    message(STATUS "Added '-framework Accelerate' to compiler flags")
  endif()

  if(DEFINED CMAKE_CXX_COMPILER_ID)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
      message(STATUS "Detected Clang compiler on macOS. Added '-stdlib=libc++' to compiler flags")
    endif()

    if(NOT (${CMAKE_MAJOR_VERSION} LESS 3))
      if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        message(STATUS "Detected AppleClang compiler on macOS. Added '-stdlib=libc++' to compiler flags")
      endif()
    endif()
  endif()

  set(CMAKE_MACOSX_RPATH 1)

else()

  if(ALLOW_FLEXIBLAS_LINUX AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    include(COOT_FindFlexiBLAS)
  else()
    set(FlexiBLAS_FOUND false)
  endif()

  include(COOT_FindMKL)
  include(COOT_FindOpenBLAS)
  include(COOT_FindBLAS)
  include(COOT_FindLAPACK)

  message(STATUS "FlexiBLAS_FOUND = ${FlexiBLAS_FOUND}" )
  message(STATUS "      MKL_FOUND = ${MKL_FOUND}"       )
  message(STATUS " OpenBLAS_FOUND = ${OpenBLAS_FOUND}"  )
  message(STATUS "     BLAS_FOUND = ${BLAS_FOUND}"      )
  message(STATUS "   LAPACK_FOUND = ${LAPACK_FOUND}"    )

  if(FlexiBLAS_FOUND)

    set(COOT_HAS_LAPACK true)
    set(COOT_HAS_BLAS   true)

    set(COOT_LIBS ${COOT_LIBS} ${FlexiBLAS_LIBRARIES})

    message(STATUS "")
    message(STATUS "*** Using FlexiBLAS to access BLAS and LAPACK functions.")
    message(STATUS "*** https://www.mpi-magdeburg.mpg.de/projects/flexiblas")
    message(STATUS "")
    message(STATUS "*** If using FlexiBLAS causes problems, ")
    message(STATUS "*** rerun cmake with FlexiBLAS detection disabled:")
    message(STATUS "*** cmake -D ALLOW_FLEXIBLAS_LINUX=false .")
    message(STATUS "")

  elseif(MKL_FOUND)

    set(COOT_HAS_LAPACK true)
    set(COOT_HAS_BLAS   true)
    set(COOT_LIBS ${COOT_LIBS} ${MKL_LIBRARIES})

    message(STATUS "")
    message(STATUS "*** If the MKL libraries are installed in non-standard locations such as")
    message(STATUS "*** /opt/intel/mkl, /opt/intel/composerxe/, /usr/local/intel/mkl")
    message(STATUS "*** make sure the run-time linker can find them.")
    message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
    message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
    message(STATUS "")
    message(STATUS "*** On systems with SELinux enabled (eg. Fedora, RHEL),")
    message(STATUS "*** you may need to change the SELinux type of all MKL libraries")
    message(STATUS "*** to fix permission problems that may occur during run-time.")
    message(STATUS "")

  else()

    if(OpenBLAS_FOUND AND ATLAS_FOUND)
      message(STATUS "")
      message(STATUS "*** NOTE: found both OpenBLAS and ATLAS; ATLAS will not be used")
    endif()

    if(OpenBLAS_FOUND AND BLAS_FOUND)
      message(STATUS "")
      message(STATUS "*** NOTE: found both OpenBLAS and BLAS; BLAS will not be used")
    endif()

    if(OpenBLAS_FOUND)

      set(COOT_HAS_BLAS true)
      set(COOT_LIBS ${COOT_LIBS} ${OpenBLAS_LIBRARIES})

      if(OPENBLAS_PROVIDES_LAPACK)
        set(COOT_HAS_LAPACK true)
      else()
        message(STATUS "")
        message(STATUS "*** NOTE: if OpenBLAS is known to provide LAPACK functions, recommend to")
        message(STATUS "*** NOTE: rerun cmake with the OPENBLAS_PROVIDES_LAPACK option enabled:")
        message(STATUS "*** NOTE: cmake -D OPENBLAS_PROVIDES_LAPACK=true .")
      endif()

      message(STATUS "")
      message(STATUS "*** If the OpenBLAS library is installed in")
      message(STATUS "*** /usr/local/lib or /usr/local/lib64")
      message(STATUS "*** make sure the run-time linker can find it.")
      message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
      message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
      message(STATUS "")

    endif()

    if(LAPACK_FOUND)
      if(OpenBLAS_FOUND AND OPENBLAS_PROVIDES_LAPACK)
        message(STATUS "*** NOTE: found both OpenBLAS and LAPACK;")
        message(STATUS "*** NOTE: option OPENBLAS_PROVIDES_LAPACK is enabled,")
        message(STATUS "*** NOTE: so will not link with plain LAPACK.")
        message(STATUS "")
      else()
        set(COOT_HAS_LAPACK true)
        set(COOT_LIBS ${COOT_LIBS} ${LAPACK_LIBRARIES})
      endif()
    endif()
  endif()
endif()

if(NOT COOT_HAS_LAPACK)
  message(STATUS "")
  message(STATUS "*** ERROR: No LAPACK compatible library has been found.")

  if(OpenBLAS_FOUND)
    message(STATUS "")
    message(STATUS "*** NOTE: OpenBLAS found but LAPACK not found.")
    message(STATUS "*** NOTE: OpenBLAS may have been built without LAPACK functions,")
    message(STATUS "*** NOTE: so cannot assume that LAPACK functions are available.")
    message(STATUS "*** NOTE: To forcefully assume that OpenBLAS provides LAPACK functions,")
    message(STATUS "*** NOTE: rerun cmake with the OPENBLAS_PROVIDES_LAPACK option enabled:")
    message(STATUS "*** NOTE: cmake -D OPENBLAS_PROVIDES_LAPACK=true .")
  endif()

  message(FATAL_ERROR "no LAPACK compatible library has been found")
endif()

################################################################################
# Backend configuration

if(FIND_OPENCL)
  find_package(OpenCL)
  if(OpenCL_FOUND)
    set(COOT_USE_OPENCL true)

    set(COOT_LIBS ${COOT_LIBS} ${OpenCL_LIBRARIES})
    set(COOT_INCLUDE_DIRS ${COOT_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

    # Now try to find clBLAS.
    include(COOT_FindCLBLAS)
    if(CLBLAS_FOUND)
      set(COOT_LIBS ${COOT_LIBS} ${CLBLAS_LIBRARIES})
      set(COOT_INCLUDE_DIRS ${COOT_INCLUDE_DIRS} ${CLBLAS_INCLUDE_DIRS})
    else()
      message(STATUS "")
      message(STATUS "*** ERROR: clBLAS not found.")
      message(STATUS "")
      message(STATUS "*** NOTE: if you do not want OpenCL support, rerun CMake with the FIND_OPENCL option off:")
      message(STATUS "*** NOTE: cmake -D FIND_OPENCL=OFF .")

      message(FATAL_ERROR "clBLAS has not been found")
    endif()
  endif()
else()
  message(STATUS "*** NOTE: not searching for OpenCL because FIND_OPENCL was set to OFF")
endif()

if(FIND_CUDA)

  if(${CMAKE_VERSION} VERSION_LESS 3.17)
    # Use older FindCUDA module because FindCUDAToolkit is not available.
    find_package(CUDA)

    if(CUDA_FOUND)
      # We also need NVRTC and also libcuda itself, which the old FindCUDA package do not find.
      find_library(CUDA_cuda_LIBRARY cuda
          HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      find_library(CUDA_nvrtc_LIBRARY nvrtc
          HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

      set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_nvrtc_LIBRARY} ${CUDA_cuda_LIBRARY})
      set(CUDA_INCLUDE_DIRS_PRINT ${CUDA_INCLUDE_DIRS})
    endif()
  else()
    # Use newer FindCUDAToolkit module.
    find_package(CUDAToolkit)

    set(CUDA_FOUND CUDAToolkit_FOUND)

    if(CUDAToolkit_FOUND)
      # Note that these variables will all contain the literal strings listed here.
      # That will cause CMake to link against the imported target, instead of just the library.
      # It's good to do that, because it's possible that each imported target may need to link against more than just the individual library.
      # (That is, it may have dependencies itself, and so linking against "CUDA::cudart" may link against more than one library.)
      set(CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver)
      set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
      set(CUDA_curand_LIBRARY CUDA::curand)
      set(CUDA_cusolver_LIBRARY CUDA::cusolver)
      set(CUDA_nvrtc_LIBRARY CUDA::nvrtc)

      # However, we want to print to the user the specific things they are linking against, because they may read the output.
      # We have to therefore recover the path explicitly.
      get_target_property(CUDADRIVER_LOC CUDA::cuda_driver IMPORTED_LOCATION)
      get_target_property(CUDART_LOC CUDA::cudart IMPORTED_LOCATION)
      get_target_property(CUBLAS_LOC CUDA::cublas IMPORTED_LOCATION)
      get_target_property(CURAND_LOC CUDA::curand IMPORTED_LOCATION)
      get_target_property(CUSOLVER_LOC CUDA::cusolver IMPORTED_LOCATION)
      get_target_property(NVRTC_LOC CUDA::nvrtc IMPORTED_LOCATION)

      get_target_property(CUDADRIVER_INCL_LOC CUDA::cuda_driver INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(CUDART_INCL_LOC CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(CUBLAS_INCL_LOC CUDA::cublas INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(CURAND_INCL_LOC CUDA::curand INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(CUSOLVER_INCL_LOC CUDA::cusolver INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(NVRTC_INCL_LOC CUDA::nvrtc INTERFACE_INCLUDE_DIRECTORIES)

      set(CUDA_LIBS_NOPRINT CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand CUDA::cusolver CUDA::nvrtc)
      set(CUDA_LIBS_PRINT ${CUDART_LOC} ${CUDADRIVER_LOC} ${CUBLAS_LOC} ${CURAND_LOC} ${CUSOLVER_LOC} ${NVRTC_LOC})
      set(CUDA_INCLUDE_DIRS CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand CUDA::cusolver CUDA::nvrtc)
      set(CUDA_INCLUDE_DIRS_PRINT ${CUDART_INCL_LOC} ${CUDADRIVER_INCL_LOC} ${CUBLAS_INCL_LOC} ${CURAND_INCL_LOC} ${CUSOLVER_INCL_LOC} ${NVRTC_INCL_LOC})

      # Set these so the next section succeeds.
      if(NOT CUDA_LIBRARIES)
        set(CUDA_LIBRARIES TRUE)
      endif()
      if(NOT CUDA_CUBLAS_LIBRARIES)
        set(CUDA_CUBLAS_LIBRARIES TRUE)
      endif()
      if(NOT CUDA_curand_LIBRARY)
        set(CUDA_curand_LIBRARY TRUE)
      endif()
      if(NOT CUDA_cusolver_LIBRARY)
        set(CUDA_cusolver_LIBRARY TRUE)
      endif()
      if(NOT CUDA_nvrtc_LIBRARY)
        set(CUDA_nvrtc_LIBRARY TRUE)
      endif()
    endif()
  endif()

  if(CUDA_FOUND)
    # Make sure all dependencies are located.
    set(COOT_USE_CUDA TRUE)

    if(NOT CUDA_LIBRARIES)
      set(COOT_USE_CUDA false)
      message(STATUS "*** WARNING: CUDA runtime libraries (libcudart) not found")
    endif()

    if(NOT CUDA_CUBLAS_LIBRARIES)
      set(COOT_USE_CUDA false)
      message(STATUS "*** WARNING: CUDA cuBLAS library not found")
    endif()

    if(NOT CUDA_curand_LIBRARY)
      set(COOT_USE_CUDA false)
      message(STATUS "*** WARNING: CUDA curand library not found")
    endif()

    if(NOT CUDA_cusolver_LIBRARY)
      set(COOT_USE_CUDA false)
      message(STATUS "*** WARNING: CUDA cuSolver library not found")
    endif()

    if(NOT CUDA_nvrtc_LIBRARY)
      set(COOT_USE_CUDA false)
      message(STATUS "*** WARNING: CUDA NVRTC library not found")
    endif()

    if(NOT COOT_USE_CUDA)
      message(STATUS "*** WARNING: CUDA backend not enabled")
    else()
      set(COOT_LIBS ${COOT_LIBS} ${CUDA_LIBS})
      set(COOT_LIBS_NOPRINT ${COOT_LIBS_NOPRINT} ${CUDA_LIBS_NOPRINT})
      set(COOT_LIBS_PRINT ${COOT_LIBS_PRINT} ${CUDA_LIBS_PRINT})
      set(COOT_INCLUDE_DIRS ${COOT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS_PRINT})
    endif()

endif()
else()
  message(STATUS "*** NOTE: not searching for CUDA backend because FIND_CUDA was set to OFF")
endif()

if(NOT DEFAULT_BACKEND)
  # Configure default backend.
  if(COOT_USE_CUDA AND COOT_USE_OPENCL)
    # If we have a CUDA device, the OpenCL driver is almost certainly better.
    set(COOT_DEFAULT_BACKEND "CUDA_BACKEND")
  elseif(COOT_USE_CUDA)
    set(COOT_DEFAULT_BACKEND "CUDA_BACKEND")
  elseif(COOT_USE_OPENCL)
    set(COOT_DEFAULT_BACKEND} "CL_BACKEND")
  endif()
else()
  # Check the user's specified default backend.
  if((NOT ${DEFAULT_BACKEND} MATCHES "CUDA_BACKEND") AND
     (NOT ${DEFAULT_BACKEND} MATCHES "CL_BACKEND"))
    message(STATUS "")
    message(STATUS "*** ERROR: unknown DEFAULT_BACKEND \"${DEFAULT_BACKEND}\"")
    message(STATUS "*** ERROR: must be \"CUDA_BACKEND\" or \"CL_BACKEND\"")
    message(STATUS "")
    message(FATAL_ERROR "unknown DEFAULT_BACKEND; see output above")
  elseif(${DEFAULT_BACKEND} MATCHES "CUDA_BACKEND")
    # Make sure the CUDA backend is available.
    if(NOT COOT_USE_CUDA)
      message(STATUS "")
      message(STATUS "*** ERROR: cannot specify \"CUDA_BACKEND\" as DEFAULT_BACKEND: the CUDA backend is not enabled")
      message(STATUS "*** ERROR: if you want to use the CUDA backend as the default backend, ensure that CUDA is found")
      message(STATUS "")
      message(FATAL_ERROR "cannot use CUDA_BACKEND as DEFAULT_BACKEND: CUDA not found")
    endif()
  else() # CL_BACKEND
    if(NOT COOT_USE_OPENCL)
      message(STATUS "")
      message(STATUS "*** ERROR: cannot specify \"CL_BACKEND\" as DEFAULT_BACKEND; the OpenCL backend is not enabled")
      message(STATUS "*** ERROR: if you want to use the OpenCL backend as the default backend, ensure that OpenCL is found")
      message(STATUS "")
      message(FATAL_ERROR "cannot use CL_BACKEND as DEFAULT_BACKEND: OpenCL not found")
    endif()
  endif()

  set(COOT_DEFAULT_BACKEND ${DEFAULT_BACKEND})
endif()

if(NOT COOT_USE_CUDA AND NOT COOT_USE_OPENCL)
  message(STATUS "")
  message(STATUS "*** ERROR: at least one backend must be enabled")
  message(STATUS "*** ERROR: make sure either OpenCL or CUDA can be found")
  message(STATUS "")
  message(FATAL_ERROR "no OpenCL or CUDA backend found")
endif()

# Clean up include directories.
list(REMOVE_DUPLICATES COOT_INCLUDE_DIRS)

message(STATUS "")
message(STATUS "*** Result of configuration:")
message(STATUS "*** COOT_USE_WRAPPER     = ${COOT_USE_WRAPPER}")
message(STATUS "*** COOT_USE_OPENCL      = ${COOT_USE_OPENCL}")
message(STATUS "*** COOT_USE_CUDA        = ${COOT_USE_CUDA}")
message(STATUS "*** COOT_DEFAULT_BACKEND = ${COOT_DEFAULT_BACKEND}")
message(STATUS "")
message(STATUS "*** Bandicoot wrapper library will use the following libraries:")
message(STATUS "*** COOT_LIBS = ${COOT_LIBS};${COOT_LIBS_PRINT}")
message(STATUS "*** COOT_INCLUDES = ${COOT_INCLUDE_DIRS}")
message(STATUS "")


message(STATUS "Copying ${PROJECT_SOURCE_DIR}/include/ to ${PROJECT_BINARY_DIR}/tmp/include/")
file(COPY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${PROJECT_BINARY_DIR}/tmp/include/)

message(STATUS "Generating ${PROJECT_BINARY_DIR}/tmp/include/config.hpp")
configure_file(${PROJECT_BINARY_DIR}/tmp/include/bandicoot_bits/config.hpp.cmake ${PROJECT_BINARY_DIR}/tmp/include/bandicoot_bits/config.hpp)

## necessary when linking with Intel MKL on Linux systems
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()


if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
  if(DEFINED CMAKE_CXX_COMPILER_ID)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    endif()
  endif()
endif()


message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )


add_library(
  bandicoot
  ${PROJECT_SOURCE_DIR}/src/wrapper_blas.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_blas_hidden_args.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_lapack.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_lapack_hidden_args.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_opencl.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_clblas.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_cuda.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_nvrtc.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_cublas.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_curand.cpp
  ${PROJECT_SOURCE_DIR}/src/wrapper_cusolver.cpp
  )

target_link_libraries(bandicoot PRIVATE ${COOT_LIBS} ${COOT_LIBS_NOPRINT})
target_include_directories(bandicoot PRIVATE ${PROJECT_BINARY_DIR}/tmp/include/ ${CMAKE_REQUIRED_INCLUDES} ${COOT_INCLUDE_DIRS})
target_include_directories(bandicoot INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/tmp/include> $<INSTALL_INTERFACE:include>)
set_target_properties(bandicoot PROPERTIES VERSION ${COOT_VERSION_MAJOR}.${COOT_VERSION_MINOR}.${COOT_VERSION_PATCH} SOVERSION ${COOT_VERSION_MAJOR})

if(BUILD_TESTS)
  message(STATUS "")
  message(STATUS "*** configuring tests")
  message(STATUS "")
  enable_testing()
  add_subdirectory(tests)
endif()


################################################################################
# INSTALL CONFIGURATION


message(STATUS ""                                                      )
message(STATUS "CMAKE_INSTALL_PREFIX     = ${CMAKE_INSTALL_PREFIX}"    )
message(STATUS "CMAKE_INSTALL_LIBDIR     = ${CMAKE_INSTALL_LIBDIR}"    )
message(STATUS "CMAKE_INSTALL_INCLUDEDIR = ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_DATADIR    = ${CMAKE_INSTALL_DATADIR}"   )
message(STATUS "CMAKE_INSTALL_BINDIR     = ${CMAKE_INSTALL_BINDIR}"    )


# Note that the trailing / character in "include/" is critical

install(
  DIRECTORY ${PROJECT_BINARY_DIR}/tmp/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN ".git" EXCLUDE
  PATTERN "*.cmake" EXCLUDE
  PATTERN "*~" EXCLUDE
  PATTERN "*orig" EXCLUDE
  PATTERN "*.sw*" EXCLUDE
  )

install(TARGETS bandicoot EXPORT BandicootLibraryDepends
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE bandicoot)

## LOCAL FILES
# Create ArmadilloConfig.cmake file for the use from the build tree
set(BANDICOOT_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/tmp/include")
set(BANDICOOT_LIB_DIR      "${PROJECT_BINARY_DIR}")
set(BANDICOOT_CMAKE_DIR    "${PROJECT_BINARY_DIR}")


message(STATUS "Generating '${PROJECT_BINARY_DIR}/BandicootConfig.cmake'")
# copy/change config and configVersion file (modify only the @xyz@ variables)
configure_file(${PROJECT_SOURCE_DIR}/cmake_aux/InstallFiles/BandicootConfig.cmake.in
  "${PROJECT_BINARY_DIR}/BandicootConfig.cmake" @ONLY)

message(STATUS "Generating '${PROJECT_BINARY_DIR}/BandicootConfigVersion.cmake'")
configure_file(${PROJECT_SOURCE_DIR}/cmake_aux/InstallFiles/BandicootConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/BandicootConfigVersion.cmake" @ONLY)

# Install the export set for use with the install-tree
install(EXPORT BandicootLibraryDepends DESTINATION
  "${CMAKE_INSTALL_DATADIR}/Bandicoot/CMake"
  COMPONENT dev)


## GLOBAL INSTALL FILES
# Create ArmadilloConfig.cmake file for the use from the install tree
# and install it
set(BANDICOOT_INCLUDE_DIRS "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
set(BANDICOOT_LIB_DIR      "${CMAKE_INSTALL_FULL_LIBDIR}")
set(BANDICOOT_CMAKE_DIR    "${CMAKE_INSTALL_FULL_DATADIR}/Bandicoot/CMake")


message(STATUS "Generating '${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfig.cmake'")
# copy/change config and configVersion file (modify only the @xyz@ variables)
configure_file(${PROJECT_SOURCE_DIR}/cmake_aux/InstallFiles/BandicootConfig.cmake.in
  "${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfig.cmake" @ONLY)

message(STATUS "Generating '${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfigVersion.cmake'")
configure_file(${PROJECT_SOURCE_DIR}/cmake_aux/InstallFiles/BandicootConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfigVersion.cmake" @ONLY)

# Install files to be found by cmake users with find_package()
install(FILES
  "${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfig.cmake"
  "${PROJECT_BINARY_DIR}/InstallFiles/BandicootConfigVersion.cmake"
  DESTINATION "${BANDICOOT_CMAKE_DIR}" COMPONENT dev)


# pkgconfig related

message(STATUS "Copying ${PROJECT_SOURCE_DIR}/misc/ to ${PROJECT_BINARY_DIR}/tmp/misc/")
file(COPY ${PROJECT_SOURCE_DIR}/misc/ DESTINATION ${PROJECT_BINARY_DIR}/tmp/misc/)

message(STATUS "Generating '${PROJECT_BINARY_DIR}/tmp/misc/bandicoot.pc'")
configure_file(${PROJECT_BINARY_DIR}/tmp/misc/bandicoot.pc.in "${PROJECT_BINARY_DIR}/tmp/misc/bandicoot.pc" @ONLY)

install(FILES "${PROJECT_BINARY_DIR}/tmp/misc/bandicoot.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
