// Copyright 2012 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef IMPALA_UDF_UDF_DEBUG_H
#define IMPALA_UDF_UDF_DEBUG_H

#include "udf.h"

#include <string>
#include <sstream>

namespace impala_udf {

template<typename T>
inline std::string DebugString(const T& val) {
  if (val.is_null) return "NULL";
  std::stringstream ss;
  ss << val.val;
  return ss.str();
}

template<>
inline std::string DebugString(const StringVal& val) {
  if (val.is_null) return "NULL";
  return std::string(reinterpret_cast<const char*>(val.ptr), val.len);
}

}

#endif

