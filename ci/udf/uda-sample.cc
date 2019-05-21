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

#include "uda-sample.h"
#include <assert.h>

using namespace impala_udf;

// ---------------------------------------------------------------------------
// This is a sample of implementing a COUNT aggregate function.
// ---------------------------------------------------------------------------
void CountInit(FunctionContext* context, BigIntVal* val) {
  val->is_null = false;
  val->val = 0;
}

void CountUpdate(FunctionContext* context, const IntVal& input, BigIntVal* val) {
  if (input.is_null) return;
  ++val->val;
}

void CountMerge(FunctionContext* context, const BigIntVal& src, BigIntVal* dst) {
  dst->val += src.val;
}

BigIntVal CountFinalize(FunctionContext* context, const BigIntVal& val) {
  return val;
}

// ---------------------------------------------------------------------------
// This is a sample of implementing a AVG aggregate function.
// ---------------------------------------------------------------------------
struct AvgStruct {
  double sum;
  int64_t count;
};

void AvgInit(FunctionContext* context, BufferVal* val) {
  assert(sizeof(AvgStruct) == 16);
  memset(*val, 0, sizeof(AvgStruct));
}

void AvgUpdate(FunctionContext* context, const DoubleVal& input, BufferVal* val) {
  if (input.is_null) return;
  AvgStruct* avg = reinterpret_cast<AvgStruct*>(*val);
  avg->sum += input.val;
  ++avg->count;
}

void AvgMerge(FunctionContext* context, const BufferVal& src, BufferVal* dst) {
  if (src == NULL) return;
  const AvgStruct* src_struct = reinterpret_cast<const AvgStruct*>(src);
  AvgStruct* dst_struct = reinterpret_cast<AvgStruct*>(*dst);
  dst_struct->sum += src_struct->sum;
  dst_struct->count += src_struct->count;
}

DoubleVal AvgFinalize(FunctionContext* context, const BufferVal& val) {
  if (val == NULL) return DoubleVal::null();
  AvgStruct* val_struct = reinterpret_cast<AvgStruct*>(val);
  return DoubleVal(val_struct->sum / val_struct->count);
}

// ---------------------------------------------------------------------------
// This is a sample of implementing the STRING_CONCAT aggregate function.
// Example: select string_concat(string_col, ",") from table
// ---------------------------------------------------------------------------
void StringConcatInit(FunctionContext* context, StringVal* val) {
  val->is_null = true;
}

void StringConcatUpdate(FunctionContext* context, const StringVal& arg1,
    const StringVal& arg2, StringVal* val) {
  if (val->is_null) {
    val->is_null = false;
    *val = StringVal(context, arg1.len);
    memcpy(val->ptr, arg1.ptr, arg1.len);
  } else {
    int new_len = val->len + arg1.len + arg2.len;
    StringVal new_val(context, new_len);
    memcpy(new_val.ptr, val->ptr, val->len);
    memcpy(new_val.ptr + val->len, arg2.ptr, arg2.len);
    memcpy(new_val.ptr + val->len + arg2.len, arg1.ptr, arg1.len);
    *val = new_val;
  }
}

void StringConcatMerge(FunctionContext* context, const StringVal& src, StringVal* dst) {
  if (src.is_null) return;
  StringConcatUpdate(context, src, ",", dst);
}

StringVal StringConcatFinalize(FunctionContext* context, const StringVal& val) {
  return val;
}

// ---------------------------------------------------------------------------
// This is a sample of implementing the SUM aggregate function for decimals.
// Example: select sum_small_decimal(dec_col) from table
// It is different than the builtin sum since it can easily overflow but can
// be faster for small tables.
// ---------------------------------------------------------------------------
void SumSmallDecimalInit(FunctionContext*, DecimalVal* val) {
  val->is_null = true;
  val->val4 = 0;
}

void SumSmallDecimalUpdate(FunctionContext* ctx,
    const DecimalVal& src, DecimalVal* dst) {
  assert(ctx->GetArgType(0)->scale == 2);
  assert(ctx->GetArgType(0)->precision == 9);
  if (src.is_null) return;
  dst->is_null = false;
  dst->val4 += src.val4;
}

void SumSmallDecimalMerge(FunctionContext*, const DecimalVal& src, DecimalVal* dst) {
  if (src.is_null) return;
  dst->is_null = false;
  dst->val4 += src.val4;
}
