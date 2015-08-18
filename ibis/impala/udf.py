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

from ibis.common import IbisTypeError

from ibis.expr.datatypes import validate_type
import ibis.expr.datatypes as _dt
import ibis.expr.operations as _ops
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.sql.exprs as _expr
import ibis.util as util


class FunctionWrapper(object):

    def __init__(self, input_type, output_type, name=None, lib_path=None):
        self.lib_path = lib_path

        self.inputs = [validate_type(x) for x in input_type]
        self.output = validate_type(output_type)

        if name is None:
            name = self.hash()

        self.name = name

        if lib_path is not None:
            self._check_library()

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix not in ['.so', '.ll']:
            raise ValueError('Invalid file type. Must be .so or .ll ')

    def hash(self):
        raise NotImplementedError

    def __repr__(self):
        klass = type(self).__name__
        return ('{0}({1}, {2!r}, {3!r})'
                .format(klass, self.name, self.inputs, self.output))


class ImpalaUDF(FunctionWrapper):

    def __init__(self, input_type, output_type, so_symbol,
                 lib_path=None, name=None):
        self.so_symbol = so_symbol
        FunctionWrapper.__init__(self, input_type, output_type,
                                 lib_path=lib_path, name=name)

    def hash(self):
        from hashlib import sha1
        val = self.so_symbol
        for in_type in self.inputs:
            val += in_type.name()

        return sha1(val).hexdigest()


class ImpalaUDAF(FunctionWrapper):

    def __init__(self, input_type, output_type, init_fn, update_fn,
                 merge_fn, finalize_fn,
                 lib_path=None, name=None):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn

        FunctionWrapper.__init__(self, input_type, output_type,
                                 lib_path=lib_path, name=name)


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


def wrap_uda(hdfs_file, inputs, output, init_fn, update_fn,
             merge_fn, finalize_fn, name=None):
    """
    Creates and returns a useful container object that can be used to
    issue a create_uda() statement and register the uda within ibis

    Parameters
    ----------
    hdfs_file: .so file that contains relevant UDA
    inputs: list of strings denoting ibis datatypes
    output: string denoting ibis datatype
    init_fn: string, C++ function name for initialization function
    update_fn: string, C++ function name for update function
    merge_fn: string, C++ function name for merge function
    finalize_fn: C++ function name for finalize function
    name: string, optional
        Used internally to track function

    Returns
    -------
    container : UDA object
    """
    wrapper = ImpalaUDAF(inputs, output, init_fn, update_fn,
                         merge_fn, finalize_fn,
                         name=name, lib_path=hdfs_file)
    op = aggregate_function(inputs, output, name=name)
    return wrapper, op


def wrap_udf(hdfs_file, inputs, output, so_symbol, name=None):
    """
    Creates and returns a useful container object that can be used to
    issue a create_udf() statement and register the udf within ibis

    Parameters
    ----------
    hdfs_file: .so file that contains relevant UDF
    inputs: list of strings denoting ibis datatypes
    output: string denoting ibis datatype
    so_symbol: string, C++ function name for relevant UDF
    name: string (optional). Used internally to track function

    Returns
    -------
    container : UDF object
    """
    wrapper = ImpalaUDF(inputs, output, so_symbol, name=name,
                        lib_path=hdfs_file)
    op = scalar_function(inputs, output, name=name)
    return wrapper, op


def scalar_function(inputs, output, name=None):
    """
    Creates and returns an operator class that can be passed to add_operation()

    Parameters:
    inputs: list of strings
      Ibis data type names
    output: string
      Ibis data type
    name: string, optional
        Used internally to track function

    Returns
    -------
    op : operator class to use in construction function
    """
    input_type, output_type = _operation_type_conversion(inputs, output)
    if name is None:
        name = util.guid()

    class_name = 'UDF_{0}'.format(name)
    return _create_operation_class(class_name, input_type, output_type)


def aggregate_function(inputs, output, name=None):
    in_values, out_value = _operation_type_conversion(inputs, output)
    if name is None:
        name = util.guid()

    class_name = 'UDF_{0}'.format(name)

    func_dict = {
        'input_type': in_values,
        'output_type': out_value,
    }
    klass = type(class_name, (_ops.ValueOp,), func_dict)
    return klass


def _create_operation_class(class_name, input_type, output_type):
    func_dict = {
        'input_type': input_type,
        'output_type': output_type,
    }
    klass = type(class_name, (_ops.ValueOp,), func_dict)
    return klass


def add_operation(op, func_name, db):
    """
    Registers the given operation within the Ibis SQL translation toolchain

    Parameters
    ----------
    op: operator class
    name: used in issuing statements to SQL engine
    database: database the relevant operator is registered to
    """
    full_name = '{0}.{1}'.format(db, func_name)
    arity = len(op.input_type.types)
    _expr._operation_registry[op] = _expr._fixed_arity_call(full_name, arity)


def _impala_type_to_ibis(tval):
    if tval in _impala_to_ibis_type:
        return _impala_to_ibis_type[tval]
    return tval


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
