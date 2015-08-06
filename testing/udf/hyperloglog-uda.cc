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
#include "lib/udf.h"

using namespace std;
using namespace impala_udf;

// This sample UDA implements the hyperloglog distinct estimate aggregate
// function.
// See these papers for more details.
// 1) Hyperloglog: The analysis of a near-optimal cardinality estimation algorithm (2007)
// 2) HyperLogLog in Practice

// Precision taken from the paper. Doesn't seem to matter very much when between [6,12]
const int HLL_PRECISION = 10;

void HllInit(FunctionContext* ctx, StringVal* dst) {
  int str_len = pow(2, HLL_PRECISION);
  dst->is_null = false;
  dst->ptr = ctx->Allocate(str_len);
  dst->len = str_len;
  memset(dst->ptr, 0, str_len);
}

static const uint64_t FNV64_PRIME = 1099511628211UL;
static const uint64_t FNV64_SEED = 14695981039346656037UL;

static uint64_t FnvHash(const void* data, int32_t bytes, uint64_t hash) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
  while (bytes--) {
    hash = (*ptr ^ hash) * FNV64_PRIME;
    ++ptr;
  }
  return hash;
}

static uint64_t Hash(const IntVal& v) {
  return FnvHash(&v.val, sizeof(int32_t), FNV64_SEED);
}

void HllUpdate(FunctionContext* ctx, const IntVal& src, StringVal* dst) {
  if (src.is_null) return;
  assert(dst != NULL);
  assert(!dst->is_null);
  assert(dst->len == pow(2, HLL_PRECISION));
  uint64_t hash_value = Hash(src);
  if (hash_value != 0) {
    // Use the lower bits to index into the number of streams and then
    // find the first 1 bit after the index bits.
    int idx = hash_value % dst->len;
    uint8_t first_one_bit = __builtin_ctzl(hash_value >> HLL_PRECISION) + 1;
    dst->ptr[idx] = ::max(dst->ptr[idx], first_one_bit);
  }
}

void HllMerge(FunctionContext* ctx, const StringVal& src, StringVal* dst) {
  assert(dst != NULL);
  assert(!dst->is_null);
  assert(!src.is_null);
  assert(dst->len == pow(2, HLL_PRECISION));
  assert(src.len == pow(2, HLL_PRECISION));
  for (int i = 0; i < src.len; ++i) {
    dst->ptr[i] = ::max(dst->ptr[i], src.ptr[i]);
  }
}

const StringVal HllSerialize(FunctionContext* ctx, const StringVal& src) {
  if (src.is_null) return src;
  // Copy intermediate state into memory owned by Impala and free allocated memory
  StringVal result(ctx, src.len);
  memcpy(result.ptr, src.ptr, src.len);
  ctx->Free(src.ptr);
  return result;
}

StringVal HllFinalize(FunctionContext* ctx, const StringVal& src) {
  assert(!src.is_null);
  assert(src.len == pow(2, HLL_PRECISION));

  const int num_streams = pow(2, HLL_PRECISION);
  // Empirical constants for the algorithm.
  float alpha = 0;
  if (num_streams == 16) {
    alpha = 0.673f;
  } else if (num_streams == 32) {
    alpha = 0.697f;
  } else if (num_streams == 64) {
    alpha = 0.709f;
  } else {
    alpha = 0.7213f / (1 + 1.079f / num_streams);
  }

  float harmonic_mean = 0;
  int num_zero_registers = 0;
  for (int i = 0; i < src.len; ++i) {
    harmonic_mean += powf(2.0f, -src.ptr[i]);
    if (src.ptr[i] == 0) ++num_zero_registers;
  }
  harmonic_mean = 1.0f / harmonic_mean;
  int64_t estimate = alpha * num_streams * num_streams * harmonic_mean;

  if (num_zero_registers != 0) {
    // Estimated cardinality is too low. Hll is too inaccurate here, instead use
    // linear counting.
    estimate = num_streams * log(static_cast<float>(num_streams) / num_zero_registers);
  }

  // Free allocated memory
  ctx->Free(src.ptr);

  // Output the estimate as ascii string
  stringstream out;
  out << estimate;
  string out_str = out.str();
  StringVal result_str(ctx, out_str.size());
  memcpy(result_str.ptr, out_str.c_str(), result_str.len);
  return result_str;
}

