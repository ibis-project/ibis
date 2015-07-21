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


#ifndef SAMPLES_UDF_H
#define SAMPLES_UDF_H

#include <impala_udf/udf.h>

using namespace impala_udf;

// Usage: > create function add(int, int) returns int
//          location '/user/cloudera/libudfsample.so' SYMBOL='AddUdf';
//        > select add(1, 2);
IntVal AddUdf(FunctionContext* context, const IntVal& arg1, const IntVal& arg2);

// Returns true if x is approximately equal to y.
// Usage: > create function fuzzy_equals(double, double) returns boolean
//          location '/user/cloudera/libudfsample.so' SYMBOL='FuzzyEquals';
//        > select fuzzy_equals(1, 1.00000001);
BooleanVal FuzzyEquals(FunctionContext* context, const DoubleVal& x, const DoubleVal& y);

// Perform tests, calculations, and transformations
// on a string value, using the set of letters 'aeiou'.

// Usage: > create function hasvowels(string) returns boolean
//          location '/user/cloudera/libudfsample.so' SYMBOL='HasVowels';
//        > select hasvowels('banana');
//        > select hasvowels('grr hm shhh');
//        > select hasvowels(c1) from t1;
BooleanVal HasVowels(FunctionContext* context, const StringVal& input);


// Usage: > create function countvowels(string) returns int
//          location '/user/cloudera/libudfsample.so' SYMBOL='CountVowels';
//        > select countvowels('abracadabra hocus pocus');
//        > select countvowels(c1) from t1;
IntVal CountVowels(FunctionContext* context, const StringVal& arg1);

// Usage: > create function stripvowels(string) returns string
//          location '/user/cloudera/libudfsample.so' SYMBOL='StripVowels';
//        > select stripvowels('colour color');
//        > select stripvowels(c1) from t1;
StringVal StripVowels(FunctionContext* context, const StringVal& arg1);

// If 'val' is constant, returns 'val', otherwise returns null. This is a simple toy UDF
// demonstrating how to use prepare and close functions to maintain shared state.
// Requires Impala 1.3 or higher.
// Usage: > create function constantarg(int) returns int
//          location '/user/cloudera/libudfsample.so' symbol='ReturnConstantArg'
//          prepare_fn='ReturnConstantArgPrepare' close_fn='ReturnConstantArgClose';
//        > select constantarg(1 + 1);
//        > select constantarg(c1) from t1 limit 1;
IntVal ReturnConstantArg(FunctionContext* context, const IntVal& val);
void ReturnConstantArgPrepare(
    FunctionContext* context, FunctionContext::FunctionStateScope scope);
void ReturnConstantArgClose(
    FunctionContext* context, FunctionContext::FunctionStateScope scope);

#endif
