cmake_minimum_required(VERSION 3.1)
project(adkalmanfilter)

########################################
# Options
########################################
option(DO_TESTS "Enable testing using GTest" ON)

########################################
# Enable C++11 and other C++ flags
########################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-statement-expression -Wall -Wno-long-long -pedantic")



########################################
# Build
########################################

# No build, it's a template library

########################################
# Testing
########################################

if (DO_TESTS)

  enable_testing()

  ######################################
  # Find packages needed
  ######################################
  find_package(GTest REQUIRED)
  #find_package(Eigen 3.3 REQUIRED )

  # Hack for now Eigen 3.3 is soon released!
  set( EIGEN33_INCLUDE_DIRS "/home/emifre/Git/eigen" )

  ######################################
  # Set up the testing
  ######################################


  include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${EIGEN33_INCLUDE_DIRS}
    include
  )

  #
  # Unit Tests
  #
  add_executable(runKFTests src/test/adkf_test.cpp)
  target_link_libraries(runKFTests ${GTEST_BOTH_LIBRARIES})
  add_test(allTests runKFTests)

endif()

