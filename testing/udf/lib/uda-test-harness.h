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


#ifndef IMPALA_UDA_TEST_HARNESS_H
#define IMPALA_UDA_TEST_HARNESS_H

#include <string>
#include <sstream>
#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "udf.h"
#include "udf-debug.h"
#include "udf-test-harness.h"

namespace impala_udf {

enum UdaExecutionMode {
  ALL = 0,
  SINGLE_NODE = 1,
  ONE_LEVEL = 2,
  TWO_LEVEL = 3,
};

template<typename RESULT, typename INTERMEDIATE>
class UdaTestHarnessBase {
 public:
  typedef void (*InitFn)(FunctionContext* context, INTERMEDIATE* result);

  typedef void (*MergeFn)(FunctionContext* context, const INTERMEDIATE& src,
      INTERMEDIATE* dst);

  typedef const INTERMEDIATE (*SerializeFn)(FunctionContext* context,
      const INTERMEDIATE& type);

  typedef RESULT (*FinalizeFn)(FunctionContext* context, const INTERMEDIATE& value);

  // UDA test harness allows for custom comparator to validate results. UDAs
  // can specify a custom comparator to, for example, tolerate numerical imprecision.
  // Returns true if x and y should be treated as equal.
  typedef bool (*ResultComparator)(const RESULT& x, const RESULT& y);

  void SetResultComparator(ResultComparator fn) {
    result_comparator_fn_ = fn;
  }

  // This must be called if the INTERMEDIATE is TYPE_FIXED_BUFFER
  void SetIntermediateSize(int byte_size) {
    fixed_buffer_byte_size_ = byte_size;
  }

  // Returns the failure string if any.
  const std::string& GetErrorMsg() const { return error_msg_; }

 protected:
  UdaTestHarnessBase(InitFn init_fn, MergeFn merge_fn,
      SerializeFn serialize_fn, FinalizeFn finalize_fn)
    : init_fn_(init_fn),
      merge_fn_(merge_fn),
      serialize_fn_(serialize_fn),
      finalize_fn_(finalize_fn),
      result_comparator_fn_(NULL),
      num_input_values_(0) {
  }

  struct ScopedFunctionContext {
    ScopedFunctionContext(FunctionContext* context, UdaTestHarnessBase* harness)
        : context_(context), harness_(harness) { }

    ~ScopedFunctionContext() {
      UdfTestHarness::CloseContext(context_);
      harness_->CheckContext(context_);
      delete context_;
    }

    FunctionContext* get() { return context_; }

   private:
    FunctionContext* context_;
    UdaTestHarnessBase* harness_;
  };

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  bool Execute(const RESULT& expected, UdaExecutionMode mode);

  // Returns false if there is an error set in the context.
  bool CheckContext(FunctionContext* context);

  // Verifies x == y, using the custom comparator if set.
  bool CheckResult(const RESULT& x, const RESULT& y);

  // Runs the UDA on a single node. The entire execution happens in 1 context.
  // The UDA does a update on all the input values and then a finalize.
  RESULT ExecuteSingleNode(ScopedFunctionContext* result_context);

  // Runs the UDA, simulating a single level aggregation. The values are processed
  // on num_nodes + 1 contexts. There are num_nodes that do update and serialize.
  // There is a final context that does merge and finalize.
  RESULT ExecuteOneLevel(int num_nodes, ScopedFunctionContext* result_context);

  // Runs the UDA, simulating a two level aggregation with num1 in the first level and
  // num2 in the second. The values are processed in num1 + num2 contexts.
  RESULT ExecuteTwoLevel(int num1, int num2, ScopedFunctionContext* result_context);

  virtual void Update(int idx, FunctionContext* context, INTERMEDIATE* dst) = 0;

  // UDA functions
  InitFn init_fn_;
  MergeFn merge_fn_;
  SerializeFn serialize_fn_;
  FinalizeFn finalize_fn_;

  // Customer comparator, NULL if default == should be used.
  ResultComparator result_comparator_fn_;

  // Set during Execute() by subclass
  int num_input_values_;

  // Buffer len for intermediate results if the type is TYPE_FIXED_BUFFER
  int fixed_buffer_byte_size_;

  // Error message if anything went wrong during the execution.
  std::string error_msg_;
};

template<typename RESULT, typename INTERMEDIATE, typename INPUT>
class UdaTestHarness : public UdaTestHarnessBase<RESULT, INTERMEDIATE> {
 public:
  typedef void (*UpdateFn)(FunctionContext* context, const INPUT& input,
      INTERMEDIATE* result);

  typedef UdaTestHarnessBase<RESULT, INTERMEDIATE> BaseClass;

  UdaTestHarness(
      typename BaseClass::InitFn init_fn,
      UpdateFn update_fn,
      typename BaseClass::MergeFn merge_fn,
      typename BaseClass::SerializeFn serialize_fn,
      typename BaseClass::FinalizeFn finalize_fn)
    : BaseClass(init_fn, merge_fn, serialize_fn, finalize_fn),
      update_fn_(update_fn) {
  }

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  bool Execute(const std::vector<INPUT>& values, const RESULT& expected,
      UdaExecutionMode mode = ALL);

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  // T needs to be compatible (i.e. castable to) with INPUT
  template<typename T>
  bool Execute(const std::vector<T>& values, const RESULT& expected,
      UdaExecutionMode mode = ALL) {
    input_.resize(values.size());
    BaseClass::num_input_values_ = input_.size();
    for (int i = 0; i < values.size(); ++i) {
      input_[i] = &values[i];
    }
    return BaseClass::Execute(expected, mode);
  }

 protected:
  virtual void Update(int idx, FunctionContext* context, INTERMEDIATE* dst);

 private:
  UpdateFn update_fn_;
  // Set during Execute()
  std::vector<const INPUT*> input_;
};

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2>
class UdaTestHarness2 : public UdaTestHarnessBase<RESULT, INTERMEDIATE> {
 public:
  typedef void (*UpdateFn)(FunctionContext* context, const INPUT1& input1,
      const INPUT2& input2, INTERMEDIATE* result);

  typedef UdaTestHarnessBase<RESULT, INTERMEDIATE> BaseClass;

  UdaTestHarness2(
      typename BaseClass::InitFn init_fn,
      UpdateFn update_fn,
      typename BaseClass::MergeFn merge_fn,
      typename BaseClass::SerializeFn serialize_fn,
      typename BaseClass::FinalizeFn finalize_fn)
    : BaseClass(init_fn, merge_fn, serialize_fn, finalize_fn),
      update_fn_(update_fn) {
  }

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  bool Execute(const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
      const RESULT& expected, UdaExecutionMode mode = ALL);

 protected:
  virtual void Update(int idx, FunctionContext* context, INTERMEDIATE* dst);

 private:
  UpdateFn update_fn_;
  const std::vector<INPUT1>* input1_;
  const std::vector<INPUT2>* input2_;
};

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3>
class UdaTestHarness3 : public UdaTestHarnessBase<RESULT, INTERMEDIATE> {
 public:
  typedef void (*UpdateFn)(FunctionContext* context, const INPUT1& input1,
      const INPUT2& input2, const INPUT3& input3, INTERMEDIATE* result);

  typedef UdaTestHarnessBase<RESULT, INTERMEDIATE> BaseClass;

  UdaTestHarness3(
      typename BaseClass::InitFn init_fn,
      UpdateFn update_fn,
      typename BaseClass::MergeFn merge_fn,
      typename BaseClass::SerializeFn serialize_fn,
      typename BaseClass::FinalizeFn finalize_fn)
    : BaseClass(init_fn, merge_fn, serialize_fn, finalize_fn),
      update_fn_(update_fn) {
  }

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  bool Execute(const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
      const std::vector<INPUT3>& values3,
      const RESULT& expected, UdaExecutionMode mode = ALL);

 protected:
  virtual void Update(int idx, FunctionContext* context, INTERMEDIATE* dst);

 private:
  UpdateFn update_fn_;
  const std::vector<INPUT1>* input1_;
  const std::vector<INPUT2>* input2_;
  const std::vector<INPUT3>* input3_;
};

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3, typename INPUT4>
class UdaTestHarness4 : public UdaTestHarnessBase<RESULT, INTERMEDIATE> {
 public:
  typedef void (*UpdateFn)(FunctionContext* context, const INPUT1& input1,
      const INPUT2& input2, const INPUT3& input3, const INPUT4& input4,
      INTERMEDIATE* result);

  typedef UdaTestHarnessBase<RESULT, INTERMEDIATE> BaseClass;

  UdaTestHarness4(
      typename BaseClass::InitFn init_fn,
      UpdateFn update_fn,
      typename BaseClass::MergeFn merge_fn,
      typename BaseClass::SerializeFn serialize_fn,
      typename BaseClass::FinalizeFn finalize_fn)
    : BaseClass(init_fn, merge_fn, serialize_fn, finalize_fn),
      update_fn_(update_fn) {
  }

  // Runs the UDA in all the modes, validating the result is 'expected' each time.
  bool Execute(const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
      const std::vector<INPUT3>& values3, const std::vector<INPUT4>& values4,
      const RESULT& expected, UdaExecutionMode mode = ALL);

 protected:
  virtual void Update(int idx, FunctionContext* context, INTERMEDIATE* dst);

 private:
  UpdateFn update_fn_;
  const std::vector<INPUT1>* input1_;
  const std::vector<INPUT2>* input2_;
  const std::vector<INPUT3>* input3_;
  const std::vector<INPUT4>* input4_;
};

}

#include <impala_udf/uda-test-harness-impl.h>

#endif

