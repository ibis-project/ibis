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


#ifndef IMPALA_UDA_TEST_HARNESS_IMPL_H
#define IMPALA_UDA_TEST_HARNESS_IMPL_H

#include <string>
#include <sstream>
#include <vector>

#include <boost/shared_ptr.hpp>

namespace impala_udf {

// Utility class to help test UDAs. This can be used to test the correctness of the
// UDA, simulating multiple possible distributed execution setups.
// For example, the harness will run in the UDA in single node mode (so merge and
// serialize are unused), single level merge and multi-level merge.
// The test application is responsible for providing the data and expected result.
class UdaTestHarnessUtil {
 public:
  template<typename T>
  static T CreateIntermediate(FunctionContext* context, int byte_size) {
    return T();
  }

  template<typename T>
  static void FreeIntermediate(FunctionContext* context, const T& v) {
    // No-op
    return;
  }

  // Copy src value into context, returning the new copy. This simulates
  // copying the bytes between nodes.
  template<typename T>
  static T CopyIntermediate(FunctionContext* context, int byte_size, const T& src) {
    return src;
  }
};

template<>
BufferVal UdaTestHarnessUtil::CreateIntermediate(
    FunctionContext* context, int byte_size) {
  return reinterpret_cast<BufferVal>(context->Allocate(byte_size));
}

template<>
void UdaTestHarnessUtil::FreeIntermediate(
    FunctionContext* context, const BufferVal& v) {
  context->Free(v);
}

template<>
BufferVal UdaTestHarnessUtil::CopyIntermediate(
    FunctionContext* context, int byte_size, const BufferVal& src) {
  BufferVal v = reinterpret_cast<BufferVal>(context->Allocate(byte_size));
  memcpy(v, src, byte_size);
  return v;
}

// Returns false if there is an error set in the context.
template<typename RESULT, typename INTERMEDIATE>
bool UdaTestHarnessBase<RESULT, INTERMEDIATE>::CheckContext(FunctionContext* context) {
  if (context->has_error()) {
    std::stringstream ss;
    ss << "UDA set error to: " << context->error_msg();
    error_msg_ = ss.str();
    return false;
  }
  return true;
}

template<typename RESULT, typename INTERMEDIATE>
bool UdaTestHarnessBase<RESULT, INTERMEDIATE>::CheckResult(
    const RESULT& x, const RESULT& y) {
  if (result_comparator_fn_ == NULL) return x == y;
  return result_comparator_fn_(x, y);
}

// Runs the UDA in all the modes, validating the result is 'expected' each time.
template<typename RESULT, typename INTERMEDIATE>
bool UdaTestHarnessBase<RESULT, INTERMEDIATE>::Execute(
    const RESULT& expected, UdaExecutionMode mode) {
  error_msg_ = "";
  RESULT result;

  FunctionContext::TypeDesc return_type;
  std::vector<FunctionContext::TypeDesc> arg_types; // TODO
  if (mode == ALL || mode == SINGLE_NODE) {
    {
      ScopedFunctionContext context(
          UdfTestHarness::CreateTestContext(return_type, arg_types), this);
      result = ExecuteSingleNode(&context);
      if (error_msg_.empty() && !CheckResult(result, expected)) {
        std::stringstream ss;
        ss << "UDA failed running in single node execution." << std::endl
            << "Expected: " << DebugString(expected)
            << " Actual: " << DebugString(result);
        error_msg_ = ss.str();
      }
    }
    if (!error_msg_.empty()) return false;
  }

  const int num_nodes[] = { 1, 2, 10, 20, 100 };
  if (mode == ALL || mode == ONE_LEVEL) {
    for (int i = 0; i < sizeof(num_nodes) / sizeof(int); ++i) {
      ScopedFunctionContext context(
          UdfTestHarness::CreateTestContext(return_type, arg_types), this);
      result = ExecuteOneLevel(num_nodes[i], &context);
      if (error_msg_.empty() && !CheckResult(result, expected)) {
        std::stringstream ss;
        ss << "UDA failed running in one level distributed mode with "
            << num_nodes[i] << " nodes." << std::endl
            << "Expected: " << DebugString(expected)
            << " Actual: " << DebugString(result);
        error_msg_ = ss.str();
        return false;
      }
    }
    if (!error_msg_.empty()) return false;
  }

  if (mode == ALL || mode == TWO_LEVEL) {
    for (int i = 0; i < sizeof(num_nodes) / sizeof(int); ++i) {
      for (int j = 0; j <= i; ++j) {
        ScopedFunctionContext context(
            UdfTestHarness::CreateTestContext(return_type, arg_types), this);
        result = ExecuteTwoLevel(num_nodes[i], num_nodes[j], &context);
        if (error_msg_.empty() && !CheckResult(result, expected)) {
          std::stringstream ss;
          ss << "UDA failed running in two level distributed mode with "
            << num_nodes[i] << " nodes in the first level and "
            << num_nodes[j] << " nodes in the second level." << std::endl
            << "Expected: " << DebugString(expected)
            << " Actual: " << DebugString(result);
          error_msg_ = ss.str();
          return false;
        }
      }
    }
    if (!error_msg_.empty()) return false;
  }
  return true;
}

template<typename RESULT, typename INTERMEDIATE>
RESULT UdaTestHarnessBase<RESULT, INTERMEDIATE>::ExecuteSingleNode(
    ScopedFunctionContext* context) {
  INTERMEDIATE intermediate =
      UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
          context->get(), fixed_buffer_byte_size_);

  init_fn_(context->get(), &intermediate);
  if (!CheckContext(context->get())) return RESULT::null();

  for (int i = 0; i < num_input_values_; ++i) {
    Update(i, context->get(), &intermediate);
  }
  if (!CheckContext(context->get())) return RESULT::null();

  // Single node doesn't need merge or serialize
  RESULT result = finalize_fn_(context->get(), intermediate);
  UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(context->get(), intermediate);
  if (!CheckContext(context->get())) return RESULT::null();
  return result;
}

template<typename RESULT, typename INTERMEDIATE>
RESULT UdaTestHarnessBase<RESULT, INTERMEDIATE>::ExecuteOneLevel(int num_nodes,
    ScopedFunctionContext* result_context) {
  std::vector<boost::shared_ptr<ScopedFunctionContext> > contexts;
  std::vector<INTERMEDIATE> intermediates;
  contexts.resize(num_nodes);

  FunctionContext::TypeDesc return_type;
  std::vector<FunctionContext::TypeDesc> arg_types; // TODO

  for (int i = 0; i < num_nodes; ++i) {
    FunctionContext* cxt = UdfTestHarness::CreateTestContext(return_type, arg_types);
    contexts[i].reset(new ScopedFunctionContext(cxt, this));
    intermediates.push_back(UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
          cxt, fixed_buffer_byte_size_));
    init_fn_(cxt, &intermediates[i]);
    if (!CheckContext(cxt)) return RESULT::null();
  }

  INTERMEDIATE merged =
      UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
          result_context->get(), fixed_buffer_byte_size_);
  init_fn_(result_context->get(), &merged);
  if (!CheckContext(result_context->get())) return RESULT::null();

  // Process all the values in the single level num_nodes contexts
  for (int i = 0; i < num_input_values_; ++i) {
    int target = i % num_nodes;
    Update(i, contexts[target].get()->get(), &intermediates[target]);
  }

  // Merge them all into the final
  for (int i = 0; i < num_nodes; ++i) {
    if (!CheckContext(contexts[i].get()->get())) return RESULT::null();
    INTERMEDIATE serialized = intermediates[i];
    if (serialize_fn_ != NULL) {
      serialized = serialize_fn_(contexts[i].get()->get(), intermediates[i]);
    }
    INTERMEDIATE copy =
        UdaTestHarnessUtil::CopyIntermediate<INTERMEDIATE>(
            result_context->get(), fixed_buffer_byte_size_, serialized);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
        contexts[i].get()->get(), intermediates[i]);
    merge_fn_(result_context->get(), copy, &merged);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(result_context->get(), copy);
    if (!CheckContext(contexts[i].get()->get())) return RESULT::null();
    contexts[i].reset();
  }
  if (!CheckContext(result_context->get())) return RESULT::null();

  RESULT result = finalize_fn_(result_context->get(), merged);
  UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(result_context->get(), merged);
  if (!CheckContext(result_context->get())) return RESULT::null();
  return result;
}

template<typename RESULT, typename INTERMEDIATE>
RESULT UdaTestHarnessBase<RESULT, INTERMEDIATE>::ExecuteTwoLevel(
    int num1, int num2, ScopedFunctionContext* result_context) {
  std::vector<boost::shared_ptr<ScopedFunctionContext> > level1_contexts, level2_contexts;
  std::vector<INTERMEDIATE> level1_intermediates, level2_intermediates;
  level1_contexts.resize(num1);
  level2_contexts.resize(num2);

  FunctionContext::TypeDesc return_type;
  std::vector<FunctionContext::TypeDesc> arg_types; // TODO

  // Initialize the intermediate contexts and intermediate result buffers
  for (int i = 0; i < num1; ++i) {
    FunctionContext* cxt = UdfTestHarness::CreateTestContext(return_type, arg_types);
    level1_contexts[i].reset(new ScopedFunctionContext(cxt, this));
    level1_intermediates.push_back(
        UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
            cxt, fixed_buffer_byte_size_));
    init_fn_(cxt, &level1_intermediates[i]);
    if (!CheckContext(cxt)) return RESULT::null();
  }
  for (int i = 0; i < num2; ++i) {
    FunctionContext* cxt = UdfTestHarness::CreateTestContext(return_type, arg_types);
    level2_contexts[i].reset(new ScopedFunctionContext(cxt, this));
    level2_intermediates.push_back(
        UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
            cxt, fixed_buffer_byte_size_));
    init_fn_(cxt, &level2_intermediates[i]);
    if (!CheckContext(cxt)) return RESULT::null();
  }

  // Initialize the final context and final intermediate buffer
  INTERMEDIATE final_intermediate =
      UdaTestHarnessUtil::CreateIntermediate<INTERMEDIATE>(
          result_context->get(), fixed_buffer_byte_size_);
  init_fn_(result_context->get(), &final_intermediate);
  if (!CheckContext(result_context->get())) return RESULT::null();

  // Assign all the input values to level 1 updates
  for (int i = 0; i < num_input_values_; ++i) {
    int target = i % num1;
    Update(i, level1_contexts[target].get()->get(), &level1_intermediates[target]);
  }

  // Serialize the level 1 intermediates and merge them with a level 2 intermediate
  for (int i = 0; i < num1; ++i) {
    if (!CheckContext(level1_contexts[i].get()->get())) return RESULT::null();
    int target = i % num2;
    INTERMEDIATE serialized = level1_intermediates[i];
    if (serialize_fn_ != NULL) {
      serialized = serialize_fn_(level1_contexts[i].get()->get(), level1_intermediates[i]);
    }
    INTERMEDIATE copy =
        UdaTestHarnessUtil::CopyIntermediate<INTERMEDIATE>(
            level2_contexts[target].get()->get(), fixed_buffer_byte_size_, serialized);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
        level1_contexts[i].get()->get(), level1_intermediates[i]);
    merge_fn_(level2_contexts[target].get()->get(),
        copy, &level2_intermediates[target]);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
        level2_contexts[target].get()->get(), copy);
    if (!CheckContext(level1_contexts[i].get()->get())) return RESULT::null();
    level1_contexts[i].reset();
  }

  // Merge all the level twos into the final
  for (int i = 0; i < num2; ++i) {
    if (!CheckContext(level2_contexts[i].get()->get())) return RESULT::null();
    INTERMEDIATE serialized = level2_intermediates[i];
    if (serialize_fn_ != NULL) {
      serialized = serialize_fn_(level2_contexts[i].get()->get(), level2_intermediates[i]);
    }
    INTERMEDIATE copy =
        UdaTestHarnessUtil::CopyIntermediate<INTERMEDIATE>(
            result_context->get(), fixed_buffer_byte_size_, serialized);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
        level2_contexts[i].get()->get(), level2_intermediates[i]);
    merge_fn_(result_context->get(), copy, &final_intermediate);
    UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
        result_context->get(), copy);
    if (!CheckContext(level2_contexts[i].get()->get())) return RESULT::null();
    level2_contexts[i].reset();
  }
  if (!CheckContext(result_context->get())) return RESULT::null();

  RESULT result = finalize_fn_(result_context->get(), final_intermediate);
  UdaTestHarnessUtil::FreeIntermediate<INTERMEDIATE>(
      result_context->get(), final_intermediate);
  if (!CheckContext(result_context->get())) return RESULT::null();
  return result;
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT>
bool UdaTestHarness<RESULT, INTERMEDIATE, INPUT>::Execute(
    const std::vector<INPUT>& values, const RESULT& expected,
    UdaExecutionMode mode) {
  input_.resize(values.size());
  BaseClass::num_input_values_ = values.size();
  for (int i = 0; i < values.size(); ++i) {
    input_[i] = &values[i];
  }
  return BaseClass::Execute(expected, mode);
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT>
void UdaTestHarness<RESULT, INTERMEDIATE, INPUT>::Update(
    int idx, FunctionContext* context, INTERMEDIATE* dst) {
  update_fn_(context, *input_[idx], dst);
}

// Runs the UDA in all the modes, validating the result is 'expected' each time.
template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2>
bool UdaTestHarness2<RESULT, INTERMEDIATE, INPUT1, INPUT2>::Execute(
    const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
    const RESULT& expected, UdaExecutionMode mode) {
  if (values1.size() != values2.size()) {
    BaseClass::error_msg_ =
        "UdaTestHarness::Execute: values1 and values2 must be the same size.";
    return false;
  }
  input1_ = &values1;
  input2_ = &values2;
  BaseClass::num_input_values_ = input1_->size();
  return BaseClass::Execute(expected, mode);
}


template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2>
void UdaTestHarness2<RESULT, INTERMEDIATE, INPUT1, INPUT2>::Update(
    int idx, FunctionContext* context, INTERMEDIATE* dst) {
  update_fn_(context, (*input1_)[idx], (*input2_)[idx], dst);
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3>
bool UdaTestHarness3<RESULT, INTERMEDIATE, INPUT1, INPUT2, INPUT3>::Execute(
    const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
    const std::vector<INPUT3>& values3, const RESULT& expected, UdaExecutionMode mode) {
  if (values1.size() != values2.size() || values1.size() != values3.size()) {
    BaseClass::error_msg_ =
        "UdaTestHarness::Execute: input values vectors must be the same size.";
    return false;
  }
  input1_ = &values1;
  input2_ = &values2;
  input3_ = &values3;
  BaseClass::num_input_values_ = input1_->size();
  return BaseClass::Execute(expected, mode);
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3>
void UdaTestHarness3<RESULT, INTERMEDIATE, INPUT1, INPUT2, INPUT3>::Update(
    int idx, FunctionContext* context, INTERMEDIATE* dst) {
  update_fn_(context, (*input1_)[idx], (*input2_)[idx], (*input3_)[idx], dst);
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3, typename INPUT4>
bool UdaTestHarness4<RESULT, INTERMEDIATE, INPUT1, INPUT2, INPUT3, INPUT4>::Execute(
    const std::vector<INPUT1>& values1, const std::vector<INPUT2>& values2,
    const std::vector<INPUT3>& values3, const std::vector<INPUT4>& values4,
    const RESULT& expected, UdaExecutionMode mode) {
  if (values1.size() != values2.size() || values1.size() != values3.size() ||
      values1.size() != values4.size()) {
    BaseClass::error_msg_ =
        "UdaTestHarness::Execute: input values vectors must be the same size.";
    return false;
  }
  input1_ = &values1;
  input2_ = &values2;
  input3_ = &values3;
  input4_ = &values4;
  BaseClass::num_input_values_ = input1_->size();
  return BaseClass::Execute(expected, mode);
}

template<typename RESULT, typename INTERMEDIATE, typename INPUT1, typename INPUT2,
    typename INPUT3, typename INPUT4>
void UdaTestHarness4<RESULT, INTERMEDIATE, INPUT1, INPUT2, INPUT3, INPUT4>::Update(
    int idx, FunctionContext* context, INTERMEDIATE* dst) {
  update_fn_(context, (*input1_)[idx], (*input2_)[idx], (*input3_)[idx], (*input4_)[idx],
      dst);
}

}

#endif

