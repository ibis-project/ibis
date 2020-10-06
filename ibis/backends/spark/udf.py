"""
APIs for creating user-defined element-wise, reduction and analytic
functions.
"""

import collections
import functools
import itertools

import pyspark.sql.functions as f
import pyspark.sql.types as pt

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.signature as sig
import ibis.udf.validate as v

from .compiler import SparkUDAFNode, SparkUDFNode, compiles
from .datatypes import spark_dtype

_udf_name_cache = collections.defaultdict(itertools.count)


class SparkUDF:
    base_class = SparkUDFNode

    def __init__(self, input_type, output_type):
        self.input_type = list(map(dt.dtype, input_type))
        self.output_type = dt.dtype(output_type)
        self.spark_output_type = spark_dtype(self.output_type)

    def validate_func_and_types(self, func):
        if not callable(func):
            raise TypeError('func must be callable, got {}'.format(func))

        # Validate that the input_type argument and the function signature
        # match and that the output_type is valid
        v.validate_input_type(self.input_type, func)
        v.validate_output_type(self.output_type)

        if not self.output_type.nullable:
            raise com.IbisTypeError(
                'Spark does not support non-nullable output types'
            )

    def pyspark_udf(self, func):
        return f.udf(func, self.spark_output_type)

    def create_udf_node(self, udf_func):
        """Create a new UDF node type and adds a corresponding compile rule.

        Parameters
        ----------
        udf_func : function
            Should be the result of calling pyspark.sql.functions.udf or
            pyspark.sql.functions.pandas_udf on the user-specified func

        Returns
        -------
        result : type
            A new SparkUDFNode or SparkUDAFNode subclass
        """
        name = udf_func.__name__
        definition = next(_udf_name_cache[name])
        external_name = '{}_{:d}'.format(name, definition)

        UDFNode = type(
            external_name,
            (self.base_class,),
            {
                'signature': sig.TypeSignature.from_dtypes(self.input_type),
                'return_type': self.output_type,
            },
        )

        # Add udf_func as a property. If added to the class namespace dict, it
        # would be incorrectly used as a bound method, i.e.
        # udf_func(t.column) would be a call to bound method func with t.column
        # interpreted as self.
        UDFNode.udf_func = property(lambda self, udf_func=udf_func: udf_func)

        @compiles(UDFNode)
        def compiles_udf_node(t, expr):
            return '{}({})'.format(
                UDFNode.__name__, ', '.join(map(t.translate, expr.op().args))
            )

        return UDFNode

    def __call__(self, func):
        self.validate_func_and_types(func)
        udf_func = self.pyspark_udf(func)
        UDFNode = self.create_udf_node(udf_func)

        @functools.wraps(func)
        def wrapped(*args):
            node = UDFNode(*args)
            casted_args = [
                arg.cast(typ) for arg, typ in zip(node.args, self.input_type)
            ]
            new_node = UDFNode(*casted_args)
            return new_node.to_expr()

        return wrapped


class SparkPandasUDF(SparkUDF):
    pandas_udf_type = f.PandasUDFType.SCALAR

    def validate_func_and_types(self, func):
        if isinstance(self.spark_output_type, (pt.MapType, pt.StructType)):
            raise com.IbisTypeError(
                'Spark does not support MapType or StructType output for \
Pandas UDFs'
            )
        if not self.input_type:
            raise com.UnsupportedArgumentError(
                'Spark does not support 0-arg pandas UDFs. Instead, create \
a 1-arg pandas UDF and ignore the arg in your function'
            )
        super().validate_func_and_types(func)

    def pyspark_udf(self, func):
        return f.pandas_udf(func, self.spark_output_type, self.pandas_udf_type)


class SparkPandasAggregateUDF(SparkPandasUDF):
    base_class = SparkUDAFNode
    pandas_udf_type = f.PandasUDFType.GROUPED_AGG


class udf:
    class elementwise:
        def __init__(self, input_type, output_type):
            self._input_type = input_type
            self._output_type = output_type

        def __call__(self, func):
            """Define a UDF (user-defined function) that operates element wise
            on a Spark DataFrame.

            Parameters
            ----------
            input_type : List[ibis.expr.datatypes.DataType]
                A list of the types found in :mod:`~ibis.expr.datatypes`. The
                length of this list must match the number of arguments to the
                function. Variadic arguments are not yet supported.
            output_type : ibis.expr.datatypes.DataType
                The return type of the function.

            Examples
            --------
            >>> import ibis
            >>> import ibis.expr.datatypes as dt
            >>> from ibis.spark.udf import udf
            >>> @udf.elementwise(input_type=[dt.string], output_type=dt.int64)
            ... def my_string_length(x):
            ...     return len(x) * 2
            """
            return SparkUDF(self._input_type, self._output_type)(func)

        @staticmethod
        def pandas(input_type, output_type):
            """Define a Pandas UDF (user-defined function) that operates
            element-wise on a Spark DataFrame. The content of the function
            should operate on a pandas.Series.

            Examples
            --------
            >>> import ibis
            >>> import ibis.expr.datatypes as dt
            >>> from ibis.spark.udf import udf
            >>> @udf.elementwise.pandas([dt.string], dt.int64)
            ... def my_string_length(x):
            ...     return x.str.len() * 2
            """
            return SparkPandasUDF(input_type, output_type)

    @staticmethod
    def reduction(input_type, output_type):
        """Define a user-defined reduction function that takes N pandas Series
        or scalar values as inputs and produces one row of output.

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.datatypes as dt
        >>> from ibis.spark.udf import udf
        >>> @udf.reduction(input_type=[dt.string], output_type=dt.int64)
        ... def my_string_length_agg(series, **kwargs):
        ...     return (series.str.len() * 2).sum()
        """
        return SparkPandasAggregateUDF(input_type, output_type)
