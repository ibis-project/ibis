# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ibis.expr.base import Schema, Literal
from ibis.expr.types import (ArrayExpr, ScalarExpr, TableExpr,
                             Int8Scalar, Int8Array,
                             Int16Scalar, Int16Array,
                             Int32Scalar, Int32Array,
                             Int64Scalar, Int64Array,
                             NullScalar,
                             BooleanScalar, BooleanArray,
                             FloatScalar, FloatArray,
                             DoubleScalar, DoubleArray,
                             StringScalar, StringArray,
                             DecimalScalar, DecimalArray,
                             TimestampScalar, TimestampArray,
                             table, literal, null, value_list, desc, unnamed)
import ibis.expr.operations as ops


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)
    """
    return ops.SearchedCaseBuilder()


def now():
    return ops.TimestampNow().to_expr()
