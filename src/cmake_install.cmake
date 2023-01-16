# Install script for directory: /home/belleya/projects/def-holt/belleya/imsrg/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/belleya/projects/def-holt/belleya/imsrg/src/half/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/belleya/projects/def-holt/belleya/imsrg/src/boost_src/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/belleya/projects/def-holt/belleya/imsrg/src/profiling/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/belleya/projects/def-holt/belleya/imsrg/src/pybind11/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/belleya/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/belleya/bin/imsrg++")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/belleya/bin/imsrg++"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/belleya/bin/imsrg++")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/belleya/bin" TYPE EXECUTABLE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/belleya/projects/def-holt/belleya/imsrg/src/imsrg++")
  if(EXISTS "$ENV{DESTDIR}/home/belleya/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/belleya/bin/imsrg++")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/strip" "$ENV{DESTDIR}/home/belleya/bin/imsrg++")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/belleya/include/AngMom.hh;/home/belleya/include/AngMomCache.hh;/home/belleya/include/Commutator.hh;/home/belleya/include/Commutator232.hh;/home/belleya/include/DarkMatterNREFT.hh;/home/belleya/include/GaussLaguerre.hh;/home/belleya/include/Generator.hh;/home/belleya/include/HFMBPT.hh;/home/belleya/include/HartreeFock.hh;/home/belleya/include/Helicity.hh;/home/belleya/include/IMSRG.hh;/home/belleya/include/IMSRGProfiler.hh;/home/belleya/include/IMSRGSolver.hh;/home/belleya/include/Jacobi3BME.hh;/home/belleya/include/M0nu.hh;/home/belleya/include/ModelSpace.hh;/home/belleya/include/Operator.hh;/home/belleya/include/Parameters.hh;/home/belleya/include/PhysicalConstants.hh;/home/belleya/include/Pwd.hh;/home/belleya/include/RPA.hh;/home/belleya/include/ReadWrite.hh;/home/belleya/include/ReferenceImplementations.hh;/home/belleya/include/ThreeBodyME.hh;/home/belleya/include/ThreeBodyStorage.hh;/home/belleya/include/ThreeBodyStorage_iso.hh;/home/belleya/include/ThreeBodyStorage_mono.hh;/home/belleya/include/ThreeBodyStorage_no2b.hh;/home/belleya/include/ThreeBodyStorage_pn.hh;/home/belleya/include/ThreeLegME.hh;/home/belleya/include/TwoBodyME.hh;/home/belleya/include/UnitTest.hh;/home/belleya/include/imsrg_util.hh;/home/belleya/include/version.hh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/belleya/include" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/home/belleya/projects/def-holt/belleya/imsrg/src/AngMom.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/AngMomCache.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Commutator.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Commutator232.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/DarkMatterNREFT.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/GaussLaguerre.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Generator.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/HFMBPT.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/HartreeFock.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Helicity.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/IMSRG.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/IMSRGProfiler.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/IMSRGSolver.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Jacobi3BME.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/M0nu.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ModelSpace.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Operator.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Parameters.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/PhysicalConstants.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/Pwd.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/RPA.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ReadWrite.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ReferenceImplementations.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyME.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyStorage.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyStorage_iso.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyStorage_mono.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyStorage_no2b.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeBodyStorage_pn.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/ThreeLegME.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/TwoBodyME.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/UnitTest.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/imsrg_util.hh"
    "/home/belleya/projects/def-holt/belleya/imsrg/src/version.hh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/belleya/projects/def-holt/belleya/imsrg/src/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
