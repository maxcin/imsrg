# Install script for directory: /Users/antoinebelley/Documents/TRIUMF/imsrg/src

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

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/bin/imsrg++")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/bin" TYPE EXECUTABLE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/imsrg++")
  if(EXISTS "$ENV{DESTDIR}/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/bin/imsrg++")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}/bin/imsrg++")
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
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/AngMom.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/AngMomCache.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/BCH.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Commutator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Commutator232.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/DaggerCommutators.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/DarkMatterNREFT.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/FactorizedDoubleCommutator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/GaussLaguerre.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Generator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/GeneratorPV.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/HFMBPT.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/HartreeFock.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Helicity.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRG.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRG3Commutators.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRGProfiler.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRGSolver.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRGSolverPV.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Jacobi3BME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/M0nu.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ModelSpace.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Operator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Parameters.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/PhysicalConstants.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Pwd.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/RPA.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ReadWrite.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ReferenceImplementations.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/TensorCommutators.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_iso.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_mono.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_no2b.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_pn.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeLegME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/TwoBodyME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/UnitTest.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/imsrg_util.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/version.hh"
    )
endif()

