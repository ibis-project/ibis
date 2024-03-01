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

from __future__ import annotations

import abc
import inspect

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util

__all__ = ["scalar_function", "aggregate_function", "wrap_udf", "wrap_uda"]


class Function(abc.ABC):
    def __init__(self, inputs, output, name, database):
        self.inputs = tuple(map(dt.dtype, inputs))
        self.output = dt.dtype(output)
        self.name = name or util.guid()
        self.database = database
        self._klass = self._create_operation_class()

    @abc.abstractmethod
    def _create_operation_class(self):
        pass

    def __repr__(self):
        ident = ".".join(filter(None, (self.database, self.name)))
        return f"{ident}({self.inputs!r}, {self.output!r})"

    def __call__(self, *args):
        return self._klass(*args)

    def _make_fn(self):
        def fn(*args, **kwargs): ...

        fn.__name__ = self.name
        fn.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    f"input{i:d}",
                    annotation=input,
                    kind=inspect.Parameter.POSITIONAL_ONLY,
                )
                for i, input in enumerate(self.inputs)
            ],
            return_annotation=self.output,
        )

        return fn


class ScalarFunction(Function):
    def _create_operation_class(self):
        return ops.scalar.builtin(
            fn=self._make_fn(),
            name=self.name,
            signature=(self.inputs, self.output),
            schema=self.database,
        )


class AggregateFunction(Function):
    def _create_operation_class(self):
        return ops.agg.builtin(
            fn=self._make_fn(),
            name=self.name,
            signature=(self.inputs, self.output),
            schema=self.database,
        )


class ImpalaFunction:
    def __init__(self, name=None, lib_path=None):
        self.lib_path = lib_path
        self.name = name or util.guid()

        if lib_path is not None:
            self._check_library()

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix not in [".so", ".ll"]:
            raise ValueError("Invalid file type. Must be .so or .ll ")


class ImpalaUDF(ScalarFunction, ImpalaFunction):
    """Feel free to customize my __doc__ or wrap in a nicer user API."""

    def __init__(
        self, inputs, output, so_symbol=None, lib_path=None, name=None, database=None
    ):
        from ibis.legacy.udf.validate import validate_output_type

        validate_output_type(output)
        self.so_symbol = so_symbol
        ImpalaFunction.__init__(self, name=name, lib_path=lib_path)
        ScalarFunction.__init__(self, inputs, output, name=self.name, database=database)


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
        database=None,
    ):
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.merge_fn = merge_fn
        self.finalize_fn = finalize_fn
        self.serialize_fn = serialize_fn

        from ibis.legacy.udf.validate import validate_output_type

        validate_output_type(output)

        ImpalaFunction.__init__(self, name=name, lib_path=lib_path)
        AggregateFunction.__init__(
            self, inputs, output, name=self.name, database=database
        )

    def _check_library(self):
        suffix = self.lib_path[-3:]
        if suffix == ".ll":
            raise com.IbisInputError("LLVM IR UDAs are not yet supported")
        elif suffix != ".so":
            raise ValueError("Invalid file type. Must be .so")


def wrap_uda(
    hdfs_file: str,
    inputs: str,
    output: str,
    update_fn: str,
    init_fn: str | None = None,
    merge_fn: str | None = None,
    finalize_fn: str | None = None,
    serialize_fn: str | None = None,
    name: str | None = None,
    database: str | None = None,
):
    """Creates a callable aggregation function object.

    Must be created in Impala to be used.

    Parameters
    ----------
    hdfs_file
        .so file that contains relevant UDA
    inputs
        list of strings denoting ibis datatypes
    output
        string denoting ibis datatype
    update_fn
        Library symbol name for update function
    init_fn
        Library symbol name for initialization function
    merge_fn
        Library symbol name for merge function
    finalize_fn
        Library symbol name for finalize function
    serialize_fn
        Library symbol name for serialize UDA API function. Not required for all
        UDAs.
    name
        Used internally to track function
    database
        Name of database

    """
    return ImpalaUDA(
        inputs,
        output,
        update_fn,
        init_fn,
        merge_fn,
        finalize_fn,
        serialize_fn=serialize_fn,
        name=name,
        lib_path=hdfs_file,
        database=database,
    )


def wrap_udf(hdfs_file, inputs, output, so_symbol, name=None, database=None):
    """Creates a callable scalar function object.

    Must be created in Impala to be used.

    Parameters
    ----------
    hdfs_file
        .so file that contains relevant UDF
    inputs
        Input types to UDF
    output
        Ibis data type
    so_symbol
        C++ function name for relevant UDF
    name
        Used internally to track function
    database
        Name of database

    """
    func = ImpalaUDF(
        inputs, output, so_symbol, name=name, lib_path=hdfs_file, database=database
    )
    return func


def scalar_function(inputs, output, name=None, database=None):
    """Create an operator class.

    Parameters
    ----------
    inputs
        Ibis data type names
    output
        Ibis data type
    name
        Used internally to track function
    database
        Name of database

    """
    return ScalarFunction(inputs, output, name=name, database=database)


def aggregate_function(inputs, output, name=None, database=None):
    """Create an operator class.

    Parameters
    ----------
    inputs
        Ibis data type names
    output
        Ibis data type
    name
        Used internally to track function
    database
        Name of database

    """
    return AggregateFunction(inputs, output, name=name, database=database)
