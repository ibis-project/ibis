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

#include <iostream>
#include <math.h>

#include "lib/uda-test-harness.h"
#include "uda-sample.h"

using namespace impala;
using namespace impala_udf;
using namespace std;

bool TestCount() {
  // Use the UDA test harness to validate the COUNT UDA.
  UdaTestHarness<BigIntVal, BigIntVal, IntVal> test(
      CountInit, CountUpdate, CountMerge, NULL, CountFinalize);

  // Run the UDA over empty input
  vector<IntVal> empty;
  if (!test.Execute(empty, BigIntVal(0))) {
    cerr << "Count empty: " << test.GetErrorMsg() << endl;
    return false;
  }

  // Run the UDA over 10000 non-null values
  vector<IntVal> no_nulls;
  no_nulls.resize(10000);
  if (!test.Execute(no_nulls, BigIntVal(no_nulls.size()))) {
    cerr << "Count without nulls: " << test.GetErrorMsg() << endl;
    return false;
  }

  // Run the UDA with some nulls
  vector<IntVal> some_nulls;
  some_nulls.resize(10000);
  int expected = some_nulls.size();
  for (int i = 0; i < some_nulls.size(); i += 100) {
    some_nulls[i] = IntVal::null();
    --expected;
  }
  if (!test.Execute(some_nulls, BigIntVal(expected))) {
    cerr << "Count with nulls: " << test.GetErrorMsg() << endl;
    return false;
  }

  return true;
}

bool TestAvg() {
  UdaTestHarness<StringVal, StringVal, DoubleVal> test(
      AvgInit, AvgUpdate, AvgMerge, AvgSerialize, AvgFinalize);
  test.SetIntermediateSize(16);

  vector<DoubleVal> vals;

  // Test empty input
  if (!test.Execute<DoubleVal>(vals, StringVal::null())) {
    cerr << "Avg empty: " << test.GetErrorMsg() << endl;
    return false;
  }

  // Test values
  for (int i = 0; i < 1001; ++i) {
    vals.push_back(DoubleVal(i));
  }
  if (!test.Execute<DoubleVal>(vals, StringVal("500"))) {
    cerr << "Avg: " << test.GetErrorMsg() << endl;
    return false;
  }
  return true;
}

bool TestStringConcat() {
  // Use the UDA test harness to validate the COUNT UDA.
  UdaTestHarness2<StringVal, StringVal, StringVal, StringVal> test(
      StringConcatInit, StringConcatUpdate, StringConcatMerge, StringConcatSerialize,
      StringConcatFinalize);

  vector<StringVal> values;
  vector<StringVal> separators;

  // Test empty input
  if (!test.Execute(values, separators, StringVal::null())) {
    cerr << "String concat empty: " << test.GetErrorMsg() << endl;
    return false;
  }

  // Test values
  values.push_back("Hello");
  values.push_back("World");

  for(int i = 0; i < values.size(); ++i) {
    separators.push_back(",");
  }
  if (!test.Execute(values, separators, StringVal("Hello,World"))) {
    cerr << "String concat: " << test.GetErrorMsg() << endl;
    return false;
  }

  return true;
}

// For algorithms that work on floating point values, the results might not match
// exactly due to floating point inprecision. The test harness allows passing a
// custom equality comparator. Here's an example of one that can tolerate some small
// error.
// This is particularly true  for distributed execution since the order the values
// are processed is variable.
bool FuzzyCompare(const DoubleVal& x, const DoubleVal& y) {
  if (x.is_null && y.is_null) return true;
  if (x.is_null || y.is_null) return false;
  return fabs(x.val - y.val) < 0.00001;
}

// Reimplementation of FuzzyCompare that parses doubles encoded as StringVals.
// TODO: This can be removed when separate intermediate types are supported in Impala 2.0
bool FuzzyCompareStrings(const StringVal& x, const StringVal& y) {
  if (x.is_null && y.is_null) return true;
  if (x.is_null || y.is_null) return false;
  // Note that atof expects null-terminated strings, which is not guaranteed by
  // StringVals. However, since our UDAs serialize double to StringVals via stringstream,
  // we know the serialized StringVals will be null-terminated in this case.
  double x_val = atof(reinterpret_cast<char*>(x.ptr));
  double y_val = atof(reinterpret_cast<char*>(y.ptr));
  return fabs(x_val - y_val) < 0.00001;
}

bool TestVariance() {
  // Setup the test UDAs.
  UdaTestHarness<StringVal, StringVal, DoubleVal> simple_variance(
      VarianceInit, VarianceUpdate, VarianceMerge, VarianceSerialize, VarianceFinalize);
  simple_variance.SetResultComparator(FuzzyCompareStrings);

  UdaTestHarness<StringVal, StringVal, DoubleVal> knuth_variance(
      KnuthVarianceInit, KnuthVarianceUpdate, KnuthVarianceMerge, KnuthVarianceSerialize,
      KnuthVarianceFinalize);
  knuth_variance.SetResultComparator(FuzzyCompareStrings);

  UdaTestHarness<StringVal, StringVal, DoubleVal> stddev(
      KnuthVarianceInit, KnuthVarianceUpdate, KnuthVarianceMerge, KnuthVarianceSerialize,
      StdDevFinalize);
  stddev.SetResultComparator(FuzzyCompareStrings);

  // Test empty input
  vector<DoubleVal> vals;
  if (!simple_variance.Execute(vals, StringVal::null())) {
    cerr << "Simple variance: " << simple_variance.GetErrorMsg() << endl;
    return false;
  }
  if (!knuth_variance.Execute(vals, StringVal::null())) {
    cerr << "Knuth variance: " << knuth_variance.GetErrorMsg() << endl;
    return false;
  }
  if (!stddev.Execute(vals, StringVal::null())) {
    cerr << "Stddev: " << stddev.GetErrorMsg() << endl;
    return false;
  }

  // Initialize the test values.
  double sum = 0;
  for (int i = 0; i < 1001; ++i) {
    vals.push_back(DoubleVal(i));
    sum += i;
  }
  double mean = sum / vals.size();
  double expected_variance = 0;
  for (int i = 0; i < vals.size(); ++i) {
    double d = mean - vals[i].val;
    expected_variance += d * d;
  }
  expected_variance /= (vals.size() - 1);
  double expected_stddev = sqrt(expected_variance);

  stringstream expected_variance_ss;
  expected_variance_ss << expected_variance;
  string expected_variance_str = expected_variance_ss.str();
  StringVal expected_variance_sv(expected_variance_str.c_str());

  stringstream expected_stddev_ss;
  expected_stddev_ss << expected_stddev;
  string expected_stddev_str = expected_stddev_ss.str();
  StringVal expected_stddev_sv(expected_stddev_str.c_str());

  // Run the tests
  if (!simple_variance.Execute(vals, expected_variance_sv)) {
    cerr << "Simple variance: " << simple_variance.GetErrorMsg() << endl;
    return false;
  }
  if (!knuth_variance.Execute(vals, expected_variance_sv)) {
    cerr << "Knuth variance: " << knuth_variance.GetErrorMsg() << endl;
    return false;
  }
  if (!stddev.Execute(vals, expected_stddev_sv)) {
    cerr << "Stddev: " << stddev.GetErrorMsg() << endl;
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  bool passed = true;
  passed &= TestCount();
  passed &= TestAvg();
  passed &= TestStringConcat();
  passed &= TestVariance();
  cerr << (passed ? "Tests passed." : "Tests failed.") << endl;
  return 0;
}
