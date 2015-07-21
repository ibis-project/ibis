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

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <impala_udf/udf.h>

#include "uda-sample.h"

using namespace std;
using namespace impala_udf;

// An implementation of a simple single pass variance algorithm. A standard UDA must
// be single pass (i.e. does not scan the table more than once), so the most canonical
// two pass approach is not practical.
// This algorithms suffers from numerical precision issues if the input values are
// large due to floating point rounding.
struct VarianceState {
  // Sum of all input values.
  double sum;
  // Sum of the square of all input values.
  double sum_squared;
  // The number of input values.
  int64_t count;
};

void VarianceInit(FunctionContext* ctx, StringVal* dst) {
  dst->is_null = false;
  dst->len = sizeof(VarianceState);
  dst->ptr = ctx->Allocate(dst->len);
  memset(dst->ptr, 0, dst->len);
}

void VarianceUpdate(FunctionContext* ctx, const DoubleVal& src, StringVal* dst) {
  if (src.is_null) return;
  VarianceState* state = reinterpret_cast<VarianceState*>(dst->ptr);
  state->sum += src.val;
  state->sum_squared += src.val * src.val;
  ++state->count;
}

void VarianceMerge(FunctionContext* ctx, const StringVal& src, StringVal* dst) {
  VarianceState* src_state = reinterpret_cast<VarianceState*>(src.ptr);
  VarianceState* dst_state = reinterpret_cast<VarianceState*>(dst->ptr);
  dst_state->sum += src_state->sum;
  dst_state->sum_squared += src_state->sum_squared;
  dst_state->count += src_state->count;
}

// A serialize function is necessary to free the intermediate state allocation.
const StringVal VarianceSerialize(FunctionContext* ctx, const StringVal& src) {
  StringVal result(ctx, src.len);
  memcpy(result.ptr, src.ptr, src.len);
  ctx->Free(src.ptr);
  return result;
}

StringVal VarianceFinalize(FunctionContext* ctx, const StringVal& src) {
  VarianceState state = *reinterpret_cast<VarianceState*>(src.ptr);
  ctx->Free(src.ptr);
  if (state.count == 0 || state.count == 1) return StringVal::null();
  double mean = state.sum / state.count;
  double variance =
      (state.sum_squared - state.sum * state.sum / state.count) / (state.count - 1);
  return ToStringVal(ctx, variance);
}

struct KnuthVarianceState {
  int64_t count;
  double mean;
  double m2;
};

void KnuthVarianceInit(FunctionContext* ctx, StringVal* dst) {
  dst->is_null = false;
  dst->len = sizeof(KnuthVarianceState);
  dst->ptr = ctx->Allocate(dst->len);
  memset(dst->ptr, 0, dst->len);
}

void KnuthVarianceUpdate(FunctionContext* ctx, const DoubleVal& src, StringVal* dst) {
  if (src.is_null) return;
  KnuthVarianceState* state = reinterpret_cast<KnuthVarianceState*>(dst->ptr);
  double temp = 1 + state->count;
  double delta = src.val - state->mean;
  double r = delta / temp;
  state->mean += r;
  state->m2 += state->count * delta * r;
  state->count = temp;
}

void KnuthVarianceMerge(FunctionContext* ctx, const StringVal& src, StringVal* dst) {
  KnuthVarianceState* src_state = reinterpret_cast<KnuthVarianceState*>(src.ptr);
  KnuthVarianceState* dst_state = reinterpret_cast<KnuthVarianceState*>(dst->ptr);
  if (src_state->count == 0) return;
  double delta = dst_state->mean - src_state->mean;
  double sum_count = dst_state->count + src_state->count;
  dst_state->mean = src_state->mean + delta * (dst_state->count / sum_count);
  dst_state->m2 = (src_state->m2) + dst_state->m2 +
      (delta * delta) * (src_state->count * dst_state->count / sum_count);
  dst_state->count = sum_count;
}

// Same as VarianceSerialize(). Create a wrapper function so automatic symbol resolution
// still works.
const StringVal KnuthVarianceSerialize(FunctionContext* ctx, const StringVal& state_sv) {
  return VarianceSerialize(ctx, state_sv);
}

// TODO: this can be used as the actual variance finalize function once the return type
// doesn't need to match the intermediate type in Impala 2.0.
DoubleVal KnuthVarianceFinalize(const StringVal& state_sv) {
  KnuthVarianceState* state = reinterpret_cast<KnuthVarianceState*>(state_sv.ptr);
  if (state->count == 0 || state->count == 1) return DoubleVal::null();
  double variance_n = state->m2 / state->count;
  double variance = variance_n * state->count / (state->count - 1);
  return DoubleVal(variance);
}

StringVal KnuthVarianceFinalize(FunctionContext* ctx, const StringVal& src) {
  StringVal result =  ToStringVal(ctx, KnuthVarianceFinalize(src));
  ctx->Free(src.ptr);
  return result;
}

StringVal StdDevFinalize(FunctionContext* ctx, const StringVal& src) {
  DoubleVal variance = KnuthVarianceFinalize(src);
  ctx->Free(src.ptr);
  if (variance.is_null) return StringVal::null();
  return ToStringVal(ctx, sqrt(variance.val));
}

