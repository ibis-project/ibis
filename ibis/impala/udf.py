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

from ibis.expr.datatypes import validate_type
import ibis.expr.datatypes as _dt
import ibis.expr.operations as _ops
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.sql.exprs as _expr
import ibis.common as com
import ibis.util as util


__all__ = ['add_operation', 'scalar_function', 'aggregate_function',
           'wrap_udf', 'wrap_uda']


class Function(object):

    def __init__(self, inputs, output, name):
        self.inputs = inputs
        self.output = output

        (self.input_type,
         self.output_type) = self._type_signature(inputs, output)
        self._klass = self._create_operation(name)

    def _create_operation(self, name):
        class_name = self._get_class_name(name)
        return _create_operation_class(class_name, self.input_type,
                                       self.output_type)

    def __repr__(self):
        klass = type(self).__name__
        return ('{0}({1}, {2!r}, {3!r})'
                .format(klass, self.name, self.inputs, self.output))

    def __call__(self, *args):
        return self._klass(*args).to_expr()

    def register(self, name, database):
        """
        Registers the given operation within the Ibis SQL translation
        toolchain. Can also use add_operation API

        Parameters
        ----------
        name: used in issuing statements to SQL engine
        database: database the relevant operator is registered to
        """
        add_operation(self._klass, name, database)


class ScalarFunction(Function):

    def _get_class_name(self, name):
        if name is None:
            name = util.guid()
        return 'UDF_{0}'.format(name)

    def _type_signature(self, inputs, output):
        input_type = _to_input_sig(inputs)
        output = validate_type(output)
        output_type = rules.shape_like_flatargs(output)
        return input_type, output_type


class AggregateFunction(Function):

    def _create_operation(self, name):
        klass = Function._create_operation(self, name)
        klass._reduction = True
        return klass

    def _get_class_name(self, name):
        if name is None:
            name = util.guid()
        return 'UDA_{0}'.format(name)

    def _type_signature(self, inputs, output):
        input_type = _to_input_sig(inputs)
        output = validate_type(output)
        output_type = rules.scalar_output(output)
        return input_type, output_type


class ImpalaFunction(object):

    def __init__(self, name=None, lib_path=None):
        self.lib_path = lib_path
        self.name = name or util.guid()

        if lib_path is not None:
            self._check_library()

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix not in ['.so', '.ll']:
            raise ValueError('Invalid file type. Must be .so or .ll ')

    def hash(self):
        raise NotImplementedError


class ImpalaUDF(ScalarFunction, ImpalaFunction):
    """
    Feel free to customize my __doc__ or wrap in a nicer user API
    """
    def __init__(self, inputs, output, so_symbol, lib_path=None,
                 name=None):
        self.so_symbol = so_symbol
        ImpalaFunction.__init__(self, name=name, lib_path=lib_path)
        ScalarFunction.__init__(self, inputs, output, name=self.name)

    def hash(self):
        # TODO: revisit this later
        # from hashlib import sha1
        # val = self.so_symbol
        # for in_type in self.inputs:
        #     val += in_type.name()

        # return sha1(val).hexdigest()
        pass


class ImpalaUDAF(AggregateFunction, ImpalaFunction):

    def __init__(self, inputs, output, init_fn, update_fn, merge_fn,
                 finalize_fn, serialize_fn=None, lib_path=None, name=None):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn
        self.serialize_fn = serialize_fn

        ImpalaFunction.__init__(self, name=name, lib_path=lib_path)
        AggregateFunction.__init__(self, inputs, output, name=self.name)

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix == '.ll':
            raise com.IbisInputError('LLVM IR UDAs are not yet supported')
        elif suffix != '.so':
            raise ValueError('Invalid file type. Must be .so')


def wrap_uda(hdfs_file, inputs, output, init_fn, update_fn,
             merge_fn, finalize_fn, serialize_fn=None, name=None):
    """
    Creates and returns a useful container object that can be used to
    issue a create_uda() statement and register the uda within ibis

    Parameters
    ----------
    hdfs_file: .so file that contains relevant UDA
    inputs: list of strings denoting ibis datatypes
    output: string denoting ibis datatype
    init_fn: string
      Library symbol name for initialization function
    update_fn: string
      Library symbol name for update function
    merge_fn: string
      Library symbol name for merge function
    finalize_fn: string
      Library symbol name for finalize function
    serialize_fn : string, optional
      Library symbol name for serialize UDA API function. Not required for all
      UDAs; see documentation for more.
    name: string, optional
      Used internally to track function

    Returns
    -------
    container : UDA object
    """
    func = ImpalaUDAF(inputs, output, init_fn, update_fn,
                      merge_fn, finalize_fn,
                      serialize_fn=serialize_fn,
                      name=name, lib_path=hdfs_file)
    return func


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
    func = ImpalaUDF(inputs, output, so_symbol, name=name,
                     lib_path=hdfs_file)
    return func


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
    klass, user_api : class, function
    """
    return ScalarFunction(inputs, output, name=name)


def aggregate_function(inputs, output, name=None):
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
    klass, user_api : class, function
    """
    return AggregateFunction(inputs, output, name=name)


def _to_input_sig(inputs):
    in_type = [validate_type(x) for x in inputs]
    return [rules.value_typed_as(_convert_types(x)) for x in in_type]


def _create_operation_class(name, input_type, output_type):
    func_dict = {
        'input_type': input_type,
        'output_type': output_type,
    }
    klass = type(name, (_ops.ValueOp,), func_dict)
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
