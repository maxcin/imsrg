# Install script for directory: /home/belleya/projects/rrg-holt/belleya/imsrg/src

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
  set(CMAKE_OBJDUMP "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/bin/imsrg++")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/bin/imsrg++"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/bin/imsrg++")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/bin" TYPE EXECUTABLE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/belleya/projects/rrg-holt/belleya/imsrg/src/imsrg++")
  if(EXISTS "$ENV{DESTDIR}/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/bin/imsrg++")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/strip" "$ENV{DESTDIR}/bin/imsrg++")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/include/AngMom.hh;/include/AngMomCache.hh;/include/BCH.hh;/include/Commutator.hh;/include/Commutator232.hh;/include/DaggerCommutators.hh;/include/DarkMatterNREFT.hh;/include/FactorizedDoubleCommutator.hh;/include/GaussLaguerre.hh;/include/Generator.hh;/include/GeneratorPV.hh;/include/HFMBPT.hh;/include/HartreeFock.hh;/include/Helicity.hh;/include/IMSRG.hh;/include/IMSRG3Commutators.hh;/include/IMSRGProfiler.hh;/include/IMSRGSolver.hh;/include/IMSRGSolverPV.hh;/include/Jacobi3BME.hh;/include/M0nu.hh;/include/ModelSpace.hh;/include/Operator.hh;/include/Parameters.hh;/include/PhysicalConstants.hh;/include/Pwd.hh;/include/RPA.hh;/include/ReadWrite.hh;/include/ReferenceImplementations.hh;/include/TensorCommutators.hh;/include/ThreeBodyME.hh;/include/ThreeBodyStorage.hh;/include/ThreeBodyStorage_iso.hh;/include/ThreeBodyStorage_mono.hh;/include/ThreeBodyStorage_no2b.hh;/include/ThreeBodyStorage_pn.hh;/include/ThreeLegME.hh;/include/TwoBodyME.hh;/include/UnitTest.hh;/include/imsrg_util.hh;/include/version.hh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/include" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/AngMom.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/AngMomCache.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/BCH.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Commutator.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Commutator232.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/DaggerCommutators.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/DarkMatterNREFT.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/FactorizedDoubleCommutator.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/GaussLaguerre.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Generator.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/GeneratorPV.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/HFMBPT.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/HartreeFock.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Helicity.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/IMSRG.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/IMSRG3Commutators.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/IMSRGProfiler.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/IMSRGSolver.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/IMSRGSolverPV.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Jacobi3BME.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/M0nu.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ModelSpace.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Operator.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Parameters.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/PhysicalConstants.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/Pwd.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/RPA.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ReadWrite.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ReferenceImplementations.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/TensorCommutators.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyME.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyStorage.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyStorage_iso.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyStorage_mono.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyStorage_no2b.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeBodyStorage_pn.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/ThreeLegME.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/TwoBodyME.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/UnitTest.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/imsrg_util.hh"
    "/home/belleya/projects/rrg-holt/belleya/imsrg/src/version.hh"
    )
endif()

