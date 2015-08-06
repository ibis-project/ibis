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


#ifndef IMPALA_UDF_TEST_HARNESS_H
#define IMPALA_UDF_TEST_HARNESS_H

#include <iostream>
#include <vector>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>

#include "udf.h"
#include "udf-debug.h"

namespace impala_udf {

// Utility class to help test UDFs.
class UdfTestHarness {
 public:
  // Create a test FunctionContext object. 'arg_types' should contain a TypeDesc for each
  // argument of the UDF not including the FunctionContext*. The caller is responsible
  // for calling delete on it. This context has additional debugging validation enabled.
  static FunctionContext* CreateTestContext(const FunctionContext::TypeDesc& return_type,
      const std::vector<FunctionContext::TypeDesc>& arg_types);

  // Use with test contexts to test use of IsArgConstant() and GetConstantArg().
  // constant_args should contain an AnyVal* for each argument of the UDF not including
  // the FunctionContext*; constant_args[i] corresponds to the i-th argument.
  // Non-constant arguments should be set to NULL, and constant arguments should be set
  // to the constant value.
  //
  // The AnyVal* values are owned by the caller.
  //
  // Can only be called on contexts created by CreateTestContext().
  static void SetConstantArgs(
      FunctionContext* context, const std::vector<AnyVal*>& constant_args);

  // Test contexts should be closed in order to check for UDF memory leaks. Leaks cause
  // the error to be set on context.
  static void CloseContext(FunctionContext* context);

  // Template function to execute a UDF and validate the result. They should be
  // used like:
  // ValidateUdf(udf_fn, arg1, arg2, ..., expected_result);
  // Only functions with up to 8 arguments are supported
  //
  // For variable argument udfs, the variable arguments should be passed as
  // a std::vector:
  //   ValidateUdf(udf_fn, arg1, arg2, const vector<arg3>& args, expected_result);
  template<typename RET>
  static bool ValidateUdf(boost::function<RET(FunctionContext*)> fn,
      const RET& expected, UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types;
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get());
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1>
  static bool ValidateUdf(boost::function<RET(FunctionContext*, const A1&)> fn,
      const A1& a1, const RET& expected, UdfPrepare init_fn = NULL,
      UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1>
  static bool ValidateUdf(boost::function<RET(FunctionContext*, int, const A1*)> fn,
      const std::vector<A1>& a1, const RET& expected, UdfPrepare init_fn = NULL,
      UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1.size(), &a1[0]);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&)> fn,
      const A1& a1, const A2& a2, const RET& expected, UdfPrepare init_fn = NULL,
      UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, int, const A2*)> fn,
      const A1& a1, const std::vector<A2>& a2, const RET& expected,
      UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2.size(), &a2[0]);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const RET& expected,
      UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, int, const A3*)> fn,
      const A1& a1, const A2& a2, const std::vector<A3>& a3, const RET& expected,
      UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3.size(), &a3[0]);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          const A4&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const A4& a4, const RET& expected,
      UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          int, const A4*)> fn,
      const A1& a1, const A2& a2, const A3& a3, const std::vector<A4>& a4,
      const RET& expected, UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4.size(), &a4[0]);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4,
      typename A5>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          const A4&, const A5&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const A4& a4,const A5& a5,
      const RET& expected, UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4, a5);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4,
      typename A5, typename A6>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          const A4&, const A5&, const A6&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5,
      const A6& a6, const RET& expected, UdfPrepare init_fn = NULL,
      UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4, a5, a6);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4,
      typename A5, typename A6, typename A7>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          const A4&, const A5&, const A6&, const A7&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5,
      const A6& a6, const A7& a7, const RET& expected, UdfPrepare init_fn = NULL,
      UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4, a5, a6, a7);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

  template<typename RET, typename A1, typename A2, typename A3, typename A4,
      typename A5, typename A6, typename A7, typename A8>
  static bool ValidateUdf(
      boost::function<RET(FunctionContext*, const A1&, const A2&, const A3&,
          const A4&, const A5&, const A6&, const A7&)> fn,
      const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5,
      const A6& a6, const A7& a7, const A8& a8, const RET& expected,
      UdfPrepare init_fn = NULL, UdfClose close_fn = NULL,
      const std::vector<AnyVal*>& constant_args = std::vector<AnyVal*>()) {
    FunctionContext::TypeDesc return_type; // TODO
    std::vector<FunctionContext::TypeDesc> arg_types; // TODO
    boost::scoped_ptr<FunctionContext> context(CreateTestContext(return_type, arg_types));
    SetConstantArgs(context.get(), constant_args);
    if (!RunPrepareFn(init_fn, context.get())) return false;
    RET ret = fn(context.get(), a1, a2, a3, a4, a5, a6, a7, a8);
    RunCloseFn(close_fn, context.get());
    return Validate(context.get(), expected, ret);
  }

 private:
  static bool ValidateError(FunctionContext* context) {
    if (context->has_error()) {
      std::cerr << "Udf Failed: " << context->error_msg() << std::endl;
      return false;
    }
    return true;
  }

  template<typename RET>
  static bool Validate(FunctionContext* context, const RET& expected, const RET& actual) {
    bool valid = true;
    if (!context->has_error() && actual != expected) {
      std::cerr << "UDF did not return the correct result:" << std::endl
                << "  Expected: " << DebugString(expected) << std::endl
                << "  Actual: " << DebugString(actual) << std::endl;
      valid = false;
    }
    CloseContext(context);
    if (!ValidateError(context)) valid = false;
    return valid;
  }

  static bool RunPrepareFn(UdfPrepare prepare_fn, FunctionContext* context) {
    if (prepare_fn != NULL) {
      // TODO: FRAGMENT_LOCAL
      prepare_fn(context, FunctionContext::THREAD_LOCAL);
      if (!ValidateError(context)) return false;
    }
    return true;
  }

  static void RunCloseFn(UdfClose close_fn, FunctionContext* context) {
    if (close_fn != NULL) {
      // TODO: FRAGMENT_LOCAL
      close_fn(context, FunctionContext::THREAD_LOCAL);
    }
  }
};

}

#endif

