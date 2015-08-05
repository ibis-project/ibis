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

#include <impala_udf/udf-test-harness.h>
#include "udf-sample.h"

using namespace impala;
using namespace impala_udf;
using namespace std;

int main(int argc, char** argv) {
  bool passed = true;
  // Using the test harness helpers, validate the UDF returns correct results.
  // These tests validate:
  //  AddUdf(1, 2) == 3
  //  AddUdf(null, 2) == null
  passed &= UdfTestHarness::ValidateUdf<IntVal, IntVal, IntVal>(
      AddUdf, IntVal(1), IntVal(2), IntVal(3));
  passed &= UdfTestHarness::ValidateUdf<IntVal, IntVal, IntVal>(
      AddUdf, IntVal::null(), IntVal(2), IntVal::null());

  // Test FuzzyEquals sample.
  passed &= UdfTestHarness::ValidateUdf<BooleanVal, DoubleVal, DoubleVal>(
      FuzzyEquals, DoubleVal(1.0), DoubleVal(1.0000000001), BooleanVal(true));
  passed &= UdfTestHarness::ValidateUdf<BooleanVal, DoubleVal, DoubleVal>(
      FuzzyEquals, DoubleVal(1.1), DoubleVal(1.0), BooleanVal(false));

  // Test ReturnConstantArg sample
  // Test constant arg
  vector<AnyVal*> constant_args;
  constant_args.push_back(new IntVal(1));
  passed &= UdfTestHarness::ValidateUdf<IntVal, IntVal>(
      ReturnConstantArg, IntVal(1), IntVal(1), ReturnConstantArgPrepare,
      ReturnConstantArgClose, constant_args);
  delete constant_args[0];
  constant_args.clear();

  // Test non-constant arg
  passed &= UdfTestHarness::ValidateUdf<IntVal, IntVal>(
      ReturnConstantArg, IntVal(1), IntVal::null(), ReturnConstantArgPrepare,
      ReturnConstantArgClose);

  cout << "Tests " << (passed ? "Passed." : "Failed.") << endl;
  return !passed;
}
