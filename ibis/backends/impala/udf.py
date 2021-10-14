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

import abc
import re

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.udf.validate as v
import ibis.util as util
from ibis.backends.base.sql.registry import fixed_arity, sql_type_names

from .compiler import ImpalaExprTranslator

__all__ = [
    'add_operation',
    'scalar_function',
    'aggregate_function',
    'wrap_udf',
    'wrap_uda',
]


class Function(metaclass=abc.ABCMeta):
    def __init__(self, inputs, output, name):
        self.inputs = tuple(map(dt.dtype, inputs))
        self.output = dt.dtype(output)
        self.name = name or util.guid()
        self._klass = self._create_operation_class()

    @abc.abstractmethod
    def _create_operation_class(self):
        pass

    def __repr__(self):
        klass = type(self).__name__
        return '{}({}, {!r}, {!r})'.format(
            klass, self.name, self.inputs, self.output
        )

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
    def _create_operation_class(self):
        fields = {
            f'_{i}': rlz.value(dtype) for i, dtype in enumerate(self.inputs)
        }
        fields['output_type'] = rlz.shape_like('args', self.output)
        return type(f"UDF_{self.name}", (ops.ValueOp,), fields)


class AggregateFunction(Function):
    def _create_operation_class(self):
        fields = {
            f'_{i}': rlz.value(dtype) for i, dtype in enumerate(self.inputs)
        }
        fields['output_type'] = lambda op: self.output.scalar_type()
        fields['_reduction'] = True
        return type(f"UDA_{self.name}", (ops.ValueOp,), fields)


class ImpalaFunction:
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

    def __init__(
        self, inputs, output, so_symbol=None, lib_path=None, name=None
    ):
        v.validate_output_type(output)
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


class ImpalaUDA(AggregateFunction, ImpalaFunction):
    def __init__(
        self,
        inputs,
        output,
        update_fn=None,
        init_fn=None,
        merge_fn=None,
        finalize_fn=None,
        serialize_fn=None,
        lib_path=None,
        name=None,
    ):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn
        self.serialize_fn = serialize_fn

        v.validate_output_type(output)

        ImpalaFunction.__init__(self, name=name, lib_path=lib_path)
        AggregateFunction.__init__(self, inputs, output, name=self.name)

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix == '.ll':
            raise com.IbisInputError('LLVM IR UDAs are not yet supported')
        elif suffix != '.so':
            raise ValueError('Invalid file type. Must be .so')


def wrap_uda(
    hdfs_file,
    inputs,
    output,
    update_fn,
    init_fn=None,
    merge_fn=None,
    finalize_fn=None,
    serialize_fn=None,
    close_fn=None,
    name=None,
):
    """
    Creates a callable aggregation function object. Must be created in Impala
    to be used

    Parameters
    ----------
    hdfs_file: .so file that contains relevant UDA
    inputs: list of strings denoting ibis datatypes
    output: string denoting ibis datatype
    update_fn: string
      Library symbol name for update function
    init_fn: string, optional
      Library symbol name for initialization function
    merge_fn: string, optional
      Library symbol name for merge function
    finalize_fn: string, optional
      Library symbol name for finalize function
    serialize_fn : string, optional
      Library symbol name for serialize UDA API function. Not required for all
      UDAs; see documentation for more.
    close_fn : string, optional
    name: string, optional
      Used internally to track function

    Returns
    -------
    container : UDA object
    """
    func = ImpalaUDA(
        inputs,
        output,
        update_fn,
        init_fn,
        merge_fn,
        finalize_fn,
        serialize_fn=serialize_fn,
        name=name,
        lib_path=hdfs_file,
    )
    return func


def wrap_udf(hdfs_file, inputs, output, so_symbol, name=None):
    """
    Creates a callable scalar function object. Must be created in Impala to be
    used

    Parameters
    ----------
    hdfs_file: .so file that contains relevant UDF
    inputs: list of strings or sig.TypeSignature
      Input types to UDF
    output: string
      Ibis data type
    so_symbol: string, C++ function name for relevant UDF
    name: string (optional). Used internally to track function

    Returns
    -------
    container : UDF object
    """
    func = ImpalaUDF(inputs, output, so_symbol, name=name, lib_path=hdfs_file)
    return func


def scalar_function(inputs, output, name=None):
    """
    Creates an operator class that can be passed to add_operation()

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
    Creates an operator class that can be passed to add_operation()

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


def add_operation(op, func_name, db):
    """
    Registers the given operation within the Ibis SQL translation toolchain

    Parameters
    ----------
    op: operator class
    name: used in issuing statements to SQL engine
    database: database the relevant operator is registered to
    """
    full_name = f'{db}.{func_name}'
    # TODO
    # if op.input_type is rlz.listof:
    #     translator = comp.varargs(full_name)
    # else:
    arity = len(op.__signature__.parameters)
    translator = fixed_arity(full_name, arity)

    ImpalaExprTranslator._registry[op] = translator


def parse_type(t):
    t = t.lower()
    if t in _impala_to_ibis_type:
        return _impala_to_ibis_type[t]
    else:
        if 'varchar' in t or 'char' in t:
            return 'string'
        elif 'decimal' in t:
            result = dt.dtype(t)
            if result:
                return t
            else:
                return ValueError(t)
        else:
            raise Exception(t)


_VARCHAR_RE = re.compile(r'varchar\((\d+)\)')


def _parse_varchar(t):
    m = _VARCHAR_RE.match(t)
    if m:
        return 'string'


def _impala_type_to_ibis(tval):
    if tval in _impala_to_ibis_type:
        return _impala_to_ibis_type[tval]
    return tval


def _ibis_string_to_impala(tval):
    if tval in sql_type_names:
        return sql_type_names[tval]
    result = dt.validate_type(tval)
    if result:
        return repr(result)


_impala_to_ibis_type = {
    'boolean': 'boolean',
    'tinyint': 'int8',
    'smallint': 'int16',
    'int': 'int32',
    'bigint': 'int64',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'varchar': 'string',
    'char': 'string',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
}
