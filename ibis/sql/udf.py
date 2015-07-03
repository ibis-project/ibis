# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib

from ibis.expr.types import (Schema, Expr,  # noqa
                             ValueExpr, ScalarExpr, ArrayExpr,
                             TableExpr,
                             NumericValue, NumericArray,
                             IntegerValue,
                             Int8Value, Int8Scalar, Int8Array,
                             Int16Value, Int16Scalar, Int16Array,
                             Int32Value, Int32Scalar, Int32Array,
                             Int64Value, Int64Scalar, Int64Array,
                             NullScalar,
                             BooleanValue, BooleanScalar, BooleanArray,
                             FloatValue, FloatScalar, FloatArray,
                             DoubleValue, DoubleScalar, DoubleArray,
                             StringValue, StringScalar, StringArray,
                             DecimalValue, DecimalScalar, DecimalArray,
                             TimestampValue, TimestampScalar, TimestampArray,
                             CategoryValue, unnamed, as_value_expr, literal,
                             null, sequence)

# __all__ is defined
from ibis.expr.temporal import *  # noqa
import ibis.common as _com

from ibis.compat import py_string
import ibis.expr.analytics as _analytics
import ibis.expr.analysis as _L
import ibis.expr.types as ir
import ibis.expr.operations as _ops
import ibis.expr.rules as rules
import ibis.sql.exprs as _expr
import ibis.util as util


class UDFInfo(object):

    def __init__(self, hdfs_file, so_symbol, input_type,
                 output_type, name=None):
        if hdfs_file[-3:] != '.so':
            raise ValueError('File is not a .so file')
        self.hdfs_file = hdfs_file
        self.so_symbol = so_symbol
        self.inputs = [ir._validate_type(x) for x in input_type]
        self.output = ir._validate_type(output_type)
        if name:
            self.name = name
        else:
            self.name = hashlib.sha1(self.so_symbol).hexdigest()

    def to_operation(self, name=None):
        """
        Creates and returns a ValueOp subclass

        Parameters
        ----------
        name : string (optional)

        Returns
        -------
        op : UdfOp object
        """
        (in_values, out_value) = _operation_type_conversion(self.inputs,
                                                            self.output)
        class_name = name
        if self.name and not name:
            class_name = self.name
        elif not (name or self.name):
            class_name = 'UDF_{0}'.format(util.guid())

        class UdfOp(_ops.ValueOp):
            input_type = in_values
            output_type = out_value
        UdfOp.__name__ = class_name
        return UdfOp

    def get_name(self):
        return self.name


def _operation_type_conversion(inputs, output):
    in_type = [ir._validate_type(x) for x in inputs]
    in_values = [rules.value_typed_as(_scalar_conversion_types[x])
                 for x in in_type]
    out_type = ir._validate_type(output)
    out_value = rules.shape_like_arg(0, out_type)
    return (in_values, out_value)


def scalar_function(inputs, output, name=None):
    (in_values, out_value) = _operation_type_conversion(inputs, output)
    class_name = name
    if not name:
        class_name = 'UDF_{0}'.format(util.guid())

    class UdfOp(_ops.ValueOp):
        input_type = in_values
        output_type = out_value

    UdfOp.__name__ = class_name
    return UdfOp


def add_impala_operation(op, func_name, db):
    _expr._operation_registry[op] = _expr._udf_op(func_name, db)


_scalar_conversion_types = {
    'boolean': (BooleanScalar),
    'int8': (Int8Scalar),
    'int16': (Int8Scalar, Int16Scalar),
    'int32': (Int8Scalar, Int16Scalar, Int32Scalar),
    'int64': (Int8Scalar, Int16Scalar, Int32Scalar, Int64Scalar),
    'float': (FloatScalar),
    'double': (FloatScalar, DoubleScalar),
    'string': (StringScalar),
    'timestamp': (TimestampScalar),
}
