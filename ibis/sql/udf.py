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

from hashlib import sha1

from ibis.common import IbisTypeError

from ibis.expr.datatypes import validate_type
import ibis.expr.datatypes as _dt
import ibis.expr.operations as _ops
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.sql.exprs as _expr
import ibis.util as util


class UDFInfo(object):

    def __init__(self, input_type, output_type, name):
        self.inputs = input_type
        self.output = output_type
        self.name = name

    def __repr__(self):
        return ('{0}({1}) returns {2}'.format(
            self.name,
            ', '.join(self.inputs),
            self.output))


class UDFCreatorParent(UDFInfo):

    def __init__(self, hdfs_file, input_type,
                 output_type, name=None):
        file_suffix = hdfs_file[-3:]
        if not(file_suffix == '.so' or file_suffix == '.ll'):
            raise ValueError('Invalid file type. Must be .so or .ll ')
        self.hdfs_file = hdfs_file
        inputs = [validate_type(x) for x in input_type]
        output = validate_type(output_type)
        new_name = name
        if not name:
            string = self.so_symbol
            for in_type in inputs:
                string += in_type.name()
            new_name = sha1(string).hexdigest()

        UDFInfo.__init__(self, inputs, output, new_name)

    def to_operation(self, name=None):
        """
        Creates and returns an operator class that can
        be passed to add_impala_operation()

        Parameters
        ----------
        name : string (optional). Used internally to track function

        Returns
        -------
        op : an operator class to use in constructing function
        """
        (in_values, out_value) = _operation_type_conversion(self.inputs,
                                                            self.output)
        class_name = name
        if self.name and not name:
            class_name = self.name
        elif not (name or self.name):
            class_name = 'UDF_{0}'.format(util.guid())
        func_dict = {
            'input_type': in_values,
            'output_type': out_value,
            }
        UdfOp = type(class_name, (_ops.ValueOp,), func_dict)
        return UdfOp

    def get_name(self):
        return self.name


class UDFCreator(UDFCreatorParent):

    def __init__(self, hdfs_file, input_type, output_type,
                 so_symbol, name=None):
        self.so_symbol = so_symbol
        UDFCreatorParent.__init__(self, hdfs_file, input_type,
                                  output_type, name=name)


class UDACreator(UDFCreatorParent):

    def __init__(self, hdfs_file, input_type, output_type, init_fn,
                 update_fn, merge_fn, finalize_fn, name=None):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn
        UDFCreatorParent.__init__(self, hdfs_file, input_type,
                                  output_type, name=name)


def _validate_impala_type(t):
    if t in _impala_to_ibis_type:
        return t
    elif _dt._DECIMAL_RE.match(t):
        return t
    raise IbisTypeError("Not a valid Impala type for UDFs")


def _operation_type_conversion(inputs, output):
    in_type = [validate_type(x) for x in inputs]
    in_values = [rules.value_typed_as(_convert_types(x)) for x in in_type]
    out_type = validate_type(output)
    out_value = rules.shape_like_flatargs(out_type)
    return (in_values, out_value)


def scalar_function(inputs, output, name=None):
    (in_values, out_value) = _operation_type_conversion(inputs, output)
    class_name = name
    if not name:
        class_name = 'UDF_{0}'.format(util.guid())

    func_dict = {
        'input_type': in_values,
        'output_type': out_value,
    }
    UdfOp = type(class_name, (_ops.ValueOp,), func_dict)
    return UdfOp


def add_impala_operation(op, func_name, db):
    full_name = '{0}.{1}'.format(db, func_name)
    arity = len(op.input_type.types)
    _expr._operation_registry[op] = _expr._fixed_arity_call(full_name, arity)


def _impala_type_to_ibis(tval):
    if tval in _impala_to_ibis_type:
        return _impala_to_ibis_type[tval]
    result = _dt._parse_decimal(tval)
    if result:
        return result.__repr__()
    raise Exception('Not a valid Impala type')


def _ibis_string_to_impala(tval):
    if tval in _expr._sql_type_names:
        return _expr._sql_type_names[tval]
    result = _dt._parse_decimal(tval)
    if result:
        return result.__repr__()


def _convert_types(t):
    name = t.name()
    return _conversion_types[name]


_conversion_types = {
    'boolean': (ir.BooleanValue),
    'int8': (ir.Int8Value),
    'int16': (ir.Int8Value, ir.Int16Value),
    'int32': (ir.Int8Value, ir.Int16Value, ir.Int32Value),
    'int64': (ir.Int8Value, ir.Int16Value, ir.Int32Value, ir.Int64Value),
    'float': (ir.FloatValue, ir.DoubleValue),
    'double': (ir.FloatValue, ir.DoubleValue),
    'string': (ir.StringValue),
    'timestamp': (ir.TimestampValue),
    'decimal': (ir.DecimalValue, ir.FloatValue, ir.DoubleValue)
}


_impala_to_ibis_type = {
    'boolean': 'boolean',
    'tinyint': 'int8',
    'smallint': 'int16',
    'int': 'int32',
    'bigint': 'int64',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'timestamp': 'timestamp',
    'decimal': 'decimal'
}
