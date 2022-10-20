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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  message("                                                     ____                                     
             _________________          _____________/   /\               _________________    
           /____/_____/_____/|         /____/_____/ /___/  \             /____/_____/_____/|   
          /____/_____/__G_ /||        /____/_____/|/   /\  /\           /____/_____/____ /||   
         /____/_____/__+__/|||       /____/_____/|/ G /  \/  \         /____/_____/_____/|||   
        |     |     |     ||||      |     |     |/___/   /\  /\       |     |     |     ||||   
        |  I  |  M  |     ||/|      |  I  |  M  /   /\  /  \/  \      |  I  |  M  |     ||/|   
        |_____|_____|_____|/||      |_____|____/ + /  \/   /\  /      |_____|_____|_____|/||   
        |     |     |     ||||      |     |   / __/   /\  /  \/       |     |     |     ||||   
        |  S  |  R  |     ||/|      |  S  |   \   \  /  \/   /        |  S  |  R  |  G  ||/|   
        |_____|_____|_____|/||      |_____|____\ __\/   /\  /         |_____|_____|_____|/||   
        |     |     |     ||||      |     |     \   \  /  \/          |     |     |     ||||   
        |     |  +  |     ||/       |     |  +  |\ __\/   /           |     |  +  |  +  ||/    
        |_____|_____|_____|/        |_____|_____|/\   \  /            |_____|_____|_____|/     
                                                   \___\/                                      
                                                                                               
")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/antoinebelley/bin/imsrg++")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/antoinebelley/bin" TYPE EXECUTABLE MESSAGE_LAZY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/Users/antoinebelley/Documents/TRIUMF/imsrg/build/imsrg++")
  if(EXISTS "$ENV{DESTDIR}/Users/antoinebelley/bin/imsrg++" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/antoinebelley/bin/imsrg++")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/antoinebelley/Documents/TRIUMF/imsrg/build"
      -add_rpath "/Users/antoinebelley/lib"
      "$ENV{DESTDIR}/Users/antoinebelley/bin/imsrg++")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}/Users/antoinebelley/bin/imsrg++")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/antoinebelley/lib/libIMSRG.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/antoinebelley/lib" TYPE SHARED_LIBRARY MESSAGE_LAZY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/Users/antoinebelley/Documents/TRIUMF/imsrg/build/libIMSRG.dylib")
  if(EXISTS "$ENV{DESTDIR}/Users/antoinebelley/lib/libIMSRG.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/antoinebelley/lib/libIMSRG.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/Users/antoinebelley/lib"
      "$ENV{DESTDIR}/Users/antoinebelley/lib/libIMSRG.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}/Users/antoinebelley/lib/libIMSRG.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/antoinebelley/include/AngMom.hh;/Users/antoinebelley/include/Commutator.hh;/Users/antoinebelley/include/DarkMatterNREFT.hh;/Users/antoinebelley/include/GaussLaguerre.hh;/Users/antoinebelley/include/Generator.hh;/Users/antoinebelley/include/HFMBPT.hh;/Users/antoinebelley/include/HartreeFock.hh;/Users/antoinebelley/include/Helicity.hh;/Users/antoinebelley/include/IMSRG.hh;/Users/antoinebelley/include/IMSRGProfiler.hh;/Users/antoinebelley/include/IMSRGSolver.hh;/Users/antoinebelley/include/Jacobi3BME.hh;/Users/antoinebelley/include/M0nu.hh;/Users/antoinebelley/include/ModelSpace.hh;/Users/antoinebelley/include/Operator.hh;/Users/antoinebelley/include/Parameters.hh;/Users/antoinebelley/include/PhysicalConstants.hh;/Users/antoinebelley/include/Pwd.hh;/Users/antoinebelley/include/ReadWrite.hh;/Users/antoinebelley/include/ThreeBodyME.hh;/Users/antoinebelley/include/ThreeBodyStorage.hh;/Users/antoinebelley/include/ThreeBodyStorage_iso.hh;/Users/antoinebelley/include/ThreeBodyStorage_no2b.hh;/Users/antoinebelley/include/ThreeBodyStorage_pn.hh;/Users/antoinebelley/include/ThreeLegME.hh;/Users/antoinebelley/include/TwoBodyME.hh;/Users/antoinebelley/include/UnitTest.hh;/Users/antoinebelley/include/imsrg_util.hh;/Users/antoinebelley/include/version.hh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/antoinebelley/include" TYPE FILE MESSAGE_LAZY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/AngMom.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Commutator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/DarkMatterNREFT.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/GaussLaguerre.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Generator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/HFMBPT.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/HartreeFock.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Helicity.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRG.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRGProfiler.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/IMSRGSolver.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Jacobi3BME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/M0nu.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ModelSpace.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Operator.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Parameters.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/PhysicalConstants.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/Pwd.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ReadWrite.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_iso.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_no2b.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeBodyStorage_pn.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/ThreeLegME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/TwoBodyME.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/UnitTest.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/imsrg_util.hh"
    "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/version.hh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/antoinebelley/include/armadillo")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/antoinebelley/include" TYPE DIRECTORY MESSAGE_LAZY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DIR_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/Users/antoinebelley/Documents/TRIUMF/imsrg/src/armadillo")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("Done installing.")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("*********************************************************************")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("* Make sure libIMSRG.so is in your LIBRARY_PATH and LD_LIBRARY_PATH *")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("*********************************************************************")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/antoinebelley/lib/pyIMSRG.cpython-310-darwin.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Users/antoinebelley/lib" TYPE MODULE MESSAGE_LAZY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ FILES "/Users/antoinebelley/Documents/TRIUMF/imsrg/build/pyIMSRG.cpython-310-darwin.so")
  if(EXISTS "$ENV{DESTDIR}/Users/antoinebelley/lib/pyIMSRG.cpython-310-darwin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/antoinebelley/lib/pyIMSRG.cpython-310-darwin.so")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/Users/antoinebelley/lib"
      "$ENV{DESTDIR}/Users/antoinebelley/lib/pyIMSRG.cpython-310-darwin.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -x "$ENV{DESTDIR}/Users/antoinebelley/lib/pyIMSRG.cpython-310-darwin.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/antoinebelley/Documents/TRIUMF/imsrg/build/pybind11/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/antoinebelley/Documents/TRIUMF/imsrg/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
