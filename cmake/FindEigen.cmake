#.rst:
# FindEigen
# --------
#
# Find eigen
#
# Find the eigen libraries
#
# ::
#
#   This module defines the following variables:
#      Eigen_FOUND       - True if Eigen_INCLUDE_DIRS & Eigen_LIBRARY are found
#      Eigen_LIBRARIES   - Set when Eigen_LIBRARY is found
#      Eigen_INCLUDE_DIRSS - Set when Eigen_INCLUDE_DIRS is found
#
#
#
# ::
#
#      Eigen_INCLUDE_DIRS - where to find header files
#      Eigen_LIBRARY     - the eigen library
#      Eigen_VERSION_STRING - the version of eigen found (since CMake 2.8.8)

#=============================================================================
# Copyright 2014 Simone Baratta < Conte91 at gmail dot com >
#
# You can do evereything you want with this file, as long as it legal.
# You must only keep this copyright notice when redistributing it or a modified version of it.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#=============================================================================

#message("Searching for eigen includes into ${CMAKE_INCLUDE_PATH}...")
find_path(Eigen_INCLUDE_DIRS NAMES Eigen/src/Core/EigenBase.h PATHS "/usr/include/eigen3"
          DOC "The Eigen include directory"
)

#message("Eigen include dir: ${Eigen_INCLUDE_DIRS}")
set(Eigen_VERSION_STRING "3.2.8")

# handle the QUIETLY and REQUIRED arguments and set Eigen_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Eigen
                                  REQUIRED_VARS Eigen_INCLUDE_DIRS
                                  VERSION_VAR Eigen_VERSION_STRING)

if(Eigen_FOUND)
  #set( Eigen_LIBRARIES ${Eigen_LIBRARY} )
  set( Eigen_INCLUDE_DIRSS ${Eigen_INCLUDE_DIRS} )
endif()

mark_as_advanced(Eigen_INCLUDE_DIRS Eigen_LIBRARY)
