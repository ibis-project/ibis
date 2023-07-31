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

cmake_minimum_required(VERSION 3.22)

project(impala_test_udfs LANGUAGES CXX)
set(CMAKE_CXX_COMPILER clang++)

# where to put generated libraries and binaries
set(LIBRARY_OUTPUT_PATH "build")
set(EXECUTABLE_OUTPUT_PATH "build")

set(CLANG_IR_CXX_FLAGS "-emit-llvm" "-c" "-DIR_COMPILE" "-DNDEBUG"
  "-DHAVE_INTTYPES_H" "-DHAVE_NETINET_IN_H"
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
    COMMAND ${CMAKE_CXX_COMPILER} ${CLANG_INCLUDE_FLAGS} ${CLANG_IR_CXX_FLAGS} ${SRC_FILE} -o ${OUTPUT_FILE}
    DEPENDS ${SRC_FILE})
  add_custom_target(${BASE_NAME}-ir ALL DEPENDS ${OUTPUT_FILE})
endfunction(COMPILE_TO_IR)

# Build the UDA/UDFs into a shared library.
add_library(udfsample SHARED udf-sample.cc)
add_library(udasample SHARED uda-sample.cc)

# Custom targets to cross compile UDA/UDF to ir
COMPILE_TO_IR(udf-sample.cc)
COMPILE_TO_IR(uda-sample.cc)
