cmake_minimum_required(VERSION 3.1)
project(adkalmanfilter)

########################################
# Enable C++11 and other C++ flags
########################################
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")

#find_package(Eigen3 REQUIRED )
# Hack for now...
set( EIGEN33_INCLUDE_DIRS "/home/emifre/Git/eigen" )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${EIGEN33_INCLUDE_DIRS}
  include
)

## Declare a C++ executable
add_executable(adkalmanfilter
  src/test/adkf_test.cpp
)

## Add cmake target dependencies of the executable
## same as for the library above
#add_dependencies(adkalmanfilter
#  ${${PROJECT_NAME}_EXPORTED_TARGETS}
#  ${catkin_EXPORTED_TARGETS}
#)

## Specify libraries to link a library or executable target against
target_link_libraries(adkalmanfilter
  ${catkin_LIBRARIES}
)

## Printing
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
