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


#ifndef SAMPLES_UDA_H
#define SAMPLES_UDA_H

#include "lib/udf.h"

using namespace impala_udf;

// Note: As of Impala 1.2, UDAs must have the same intermediate and result types (see the
// udf.h header for the full Impala UDA specification, which can be found at
// https://github.com/cloudera/impala/blob/master/be/src/udf/udf.h). Some UDAs naturally
// conform to this limitation, such as Count and StringConcat. However, other UDAs return
// a numeric value but use a custom intermediate struct type that must be stored in a
// StringVal or BufferVal, such as Variance.
//
// As a workaround for now, these UDAs that require an intermediate buffer use StringVal
// for the intermediate and result type. In the UDAs' finalize functions, the numeric
// result is serialized to an ASCII string (see the ToStringVal() utility function
// provided with these samples). The returned StringVal is then cast back to the correct
// numeric type (see the Usage examples below).
//
// This restriction will be lifted in Impala 2.0.


// This is an example of the COUNT aggregate function.
//
// Usage: > create aggregate function my_count(int) returns bigint
//          location '/user/cloudera/libudasample.so' update_fn='CountUpdate';
//        > select my_count(col) from tbl;
void CountInit(FunctionContext* context, BigIntVal* val);
void CountUpdate(FunctionContext* context, const IntVal& input, BigIntVal* val);
void CountMerge(FunctionContext* context, const BigIntVal& src, BigIntVal* dst);
BigIntVal CountFinalize(FunctionContext* context, const BigIntVal& val);

// This is an example of the AVG(double) aggregate function. This function needs to
// maintain two pieces of state, the current sum and the count. We do this using
// the BufferVal intermediate type. When this UDA is registered, it would specify
// 16 bytes (8 byte sum + 8 byte count) as the size for this buffer.
//
// Usage: > create aggregate function my_avg(double) returns string 
//          location '/user/cloudera/libudasample.so' update_fn='AvgUpdate';
//        > select cast(my_avg(col) as double) from tbl;
//
// TODO: The StringVal intermediate type should be replaced by a prealloacted BufferVal
// and the return type changed to DoubleVal in Impala 2.0
void AvgInit(FunctionContext* context, StringVal* val);
void AvgUpdate(FunctionContext* context, const DoubleVal& input, StringVal* val);
void AvgMerge(FunctionContext* context, const StringVal& src, StringVal* dst);
const StringVal AvgSerialize(FunctionContext* context, const StringVal& val);
StringVal AvgFinalize(FunctionContext* context, const StringVal& val);

// This is a sample of implementing the STRING_CONCAT aggregate function.
//
// Usage: > create aggregate function string_concat(string, string) returns string
//          location '/user/cloudera/libudasample.so' update_fn='StringConcatUpdate';
//        > select string_concat(string_col, ",") from table;
void StringConcatInit(FunctionContext* context, StringVal* val);
void StringConcatUpdate(FunctionContext* context, const StringVal& arg1,
    const StringVal& arg2, StringVal* val);
void StringConcatMerge(FunctionContext* context, const StringVal& src, StringVal* dst);
const StringVal StringConcatSerialize(FunctionContext* context, const StringVal& val);
StringVal StringConcatFinalize(FunctionContext* context, const StringVal& val);

// This is a example of the variance aggregate function.
//
// Usage: > create aggregate function var(double) returns string
//          location '/user/cloudera/libudasample.so' update_fn='VarianceUpdate';
//        > select cast(var(col) as double) from tbl;
//
// TODO: The StringVal intermediate type should be replaced by a prealloacted BufferVal
// and the return type changed to DoubleVal in Impala 2.0
void VarianceInit(FunctionContext* context, StringVal* val);
void VarianceUpdate(FunctionContext* context, const DoubleVal& input, StringVal* val);
void VarianceMerge(FunctionContext* context, const StringVal& src, StringVal* dst);
const StringVal VarianceSerialize(FunctionContext* context, const StringVal& val);
StringVal VarianceFinalize(FunctionContext* context, const StringVal& val);

// An implementation of the Knuth online variance algorithm, which is also single pass and
// more numerically stable.
//
// Usage: > create aggregate function knuth_var(double) returns string
//          location '/user/cloudera/libudasample.so' update_fn='KnuthVarianceUpdate';
//        > select cast(knuth_var(col) as double) from tbl;
//
// TODO: The StringVal intermediate type should be replaced by a prealloacted BufferVal
// and the return type changed to DoubleVal in Impala 2.0
void KnuthVarianceInit(FunctionContext* context, StringVal* val);
void KnuthVarianceUpdate(FunctionContext* context, const DoubleVal& input, StringVal* val);
void KnuthVarianceMerge(FunctionContext* context, const StringVal& src, StringVal* dst);
const StringVal KnuthVarianceSerialize(FunctionContext* context, const StringVal& val);
StringVal KnuthVarianceFinalize(FunctionContext* context, const StringVal& val);

// The different steps of the UDA are composable. In this case, we'the UDA will use the
// other steps from the Knuth variance computation.
//
// Usage: > create aggregate function stddev(double) returns string
//          location '/user/cloudera/libudasample.so' update_fn='KnuthVarianceUpdate'
//          finalize_fn="StdDevFinalize";
//        > select cast(stddev(col) as double) from tbl;
//
// TODO: The StringVal intermediate type should be replaced by a prealloacted BufferVal
// and the return type changed to DoubleVal in Impala 2.0
StringVal StdDevFinalize(FunctionContext* context, const StringVal& val);

// Utility function for serialization to StringVal
// TODO: this will be unnecessary in Impala 2.0, when we will no longer have to serialize
// results to StringVals in order to match the intermediate type
template <typename T>
StringVal ToStringVal(FunctionContext* context, const T& val);

#endif
