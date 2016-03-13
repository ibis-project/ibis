# Copyright 2012 Cloudera Inc.
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

cmake_minimum_required(VERSION 2.6)

# where to put generated libraries
set(LIBRARY_OUTPUT_PATH "build")
# where to put generated binaries
set(EXECUTABLE_OUTPUT_PATH "build")

find_program(CLANG_EXECUTABLE clang++)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb")

# find boost headers and libs
if ($ENV{ENABLE_NATIVE_TOOLCHAIN})
  message(STATUS "Found Impala Toolchain, enabling toolchain boost")
  set(ENV{BOOST_ROOT} "$ENV{IMPALA_TOOLCHAIN}/boost-$ENV{BOOST_VERSION}")
  set(Boost_USE_STATIC_LIBS NOT ${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_RUNTIME ON)
else()
  set(Boost_DEBUG TRUE)
  set(Boost_USE_MULTITHREADED ON)
endif()

find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})
message(STATUS "Boost include dir: " ${Boost_INCLUDE_DIRS})
message(STATUS "Boost libraries: " ${Boost_LIBRARIES})

set(CLANG_INCLUDE_FLAGS)

if ($ENV{ENABLE_NATIVE_TOOLCHAIN})
  set(IMPALA_TOOLCHAIN ON)

  # Set the root directory for the toolchain
  set(TOOLCHAIN_ROOT $ENV{IMPALA_TOOLCHAIN})

  if($ENV{USE_SYSTEM_GCC} EQUAL "0")
	# If Impala is built with the GCC from the toolchain, change compiler and link paths
	set(GCC_ROOT $ENV{IMPALA_TOOLCHAIN}/gcc-$ENV{IMPALA_GCC_VERSION})
	set(CMAKE_C_COMPILER ${GCC_ROOT}/bin/gcc)
	set(CMAKE_CXX_COMPILER ${GCC_ROOT}/bin/g++)

	# The rpath is needed to be able to run the binaries produced by the toolchain without
	# specifying an LD_LIBRARY_PATH
	set(TOOLCHAIN_LINK_FLAGS "-Wl,-rpath,${GCC_ROOT}/lib64")
	set(TOOLCHAIN_LINK_FLAGS "${TOOLCHAIN_LINK_FLAGS} -L${GCC_ROOT}/lib64")
  endif()

  if(NOT $ENV{IMPALA_CXX_COMPILER} STREQUAL "default")
	set(CMAKE_CXX_COMPILER $ENV{IMPALA_CXX_COMPILER})
  endif()
endif()

message(STATUS "Setup toolchain link flags ${TOOLCHAIN_LINK_FLAGS}")

if (IMPALA_TOOLCHAIN)
  # When the Toolchain is used we use LLVM 3.3 that was built in a different path that it
  # is invoked from, and a GCC that resides in a different location. LVVM 3.3 relies on
  # hard-coded path information about where to find the system headers and does not support
  # specifying the -gcc-toolchain flag to dynamically provide this information. Because of
  # these reasons we need to manually add the system c++ headers to the path when we
  # compile the IR code with clang.
  set(CLANG_BASE_FLAGS
    "-I${GCC_ROOT}/include/c++/$ENV{IMPALA_GCC_VERSION}"
    "-I${GCC_ROOT}/include/c++/$ENV{IMPALA_GCC_VERSION}/backward"
    "-I${GCC_ROOT}/include/c++/$ENV{IMPALA_GCC_VERSION}/x86_64-unknown-linux-gnu")
  set(CLANG_INCLUDE_FLAGS ${CLANG_BASE_FLAGS})
endif()

set(CLANG_INCLUDE_FLAGS
  ${CLANG_INCLUDE_FLAGS}
  "-I${Boost_INCLUDE_DIRS}"
)

set(CLANG_IR_CXX_FLAGS "-emit-llvm" "-c" "-DIR_COMPILE" "-DNDEBUG"
  "-DHAVE_INTTYPES_H" "-DHAVE_NETINET_IN_H" "-DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG"
  "-Wno-deprecated" "-Wno-c++11-extensions" "-O1")

# Function to generate rule to cross compile a source file to an IR module.
# This should be called with the .cc src file and it will generate a
# src-file-ir target that can be built.
# e.g. COMPILE_TO_IR(test.cc) generates the "test-ir" make target.
function(COMPILE_TO_IR SRC_FILE)
  get_filename_component(BASE_NAME ${SRC_FILE} NAME_WE)
  set(OUTPUT_FILE "build/${BASE_NAME}.ll")
  add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND ${CLANG_EXECUTABLE} ${CLANG_INCLUDE_FLAGS} ${CLANG_IR_CXX_FLAGS} ${SRC_FILE} -o ${OUTPUT_FILE}
    DEPENDS ${SRC_FILE})
  add_custom_target(${BASE_NAME}-ir ALL DEPENDS ${OUTPUT_FILE})
endfunction(COMPILE_TO_IR)

# Build the UDA/UDFs into a shared library.
add_library(udfsample SHARED udf-sample.cc)
add_library(udasample SHARED uda-sample.cc)

# Custom targest to cross compile UDA/UDF to ir
if (CLANG_EXECUTABLE)
  COMPILE_TO_IR(udf-sample.cc )
  COMPILE_TO_IR(uda-sample.cc )
endif(CLANG_EXECUTABLE)
