from __future__ import annotations

import collections
import inspect
import itertools
from typing import TYPE_CHECKING, Callable, Literal

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.backends.bigquery.datatypes import BigQueryType, spread_type
from ibis.backends.bigquery.operations import BigQueryUDFNode
from ibis.backends.bigquery.udf.core import PythonToJavaScriptTranslator
from ibis.legacy.udf.validate import validate_output_type

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ("udf",)

_udf_name_cache: dict[str, Iterable[int]] = collections.defaultdict(itertools.count)


def _make_udf_name(name):
    definition = next(_udf_name_cache[name])
    external_name = f"{name}_{definition:d}"
    return external_name


class _BigQueryUDF:
    def __call__(self, *args, **kwargs):
        return self.python(*args, **kwargs)

    def python(
        self,
        input_type: Iterable[dt.DataType],
        output_type: dt.DataType,
        strict: bool = True,
        libraries: Iterable[str] | None = None,
        determinism: bool | None = None,
    ) -> Callable:
        '''Define a UDF for BigQuery.

        The function is transpiled to JS.

        `INT64` is not supported as an argument type or a return type, as per
        [the BigQuery documentation](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions#sql-type-encodings-in-javascript).

        Parameters
        ----------
        input_type
            Iterable of types, one per argument.
        output_type
            Return type of the UDF.
        strict
            Whether or not to put a ``'use strict';`` string at the beginning of
            the UDF. Setting to ``False`` is probably a bad idea.
        libraries
            An iterable of Google Cloud Storage URIs containing to JavaScript source
            code. Note that any symbols (functions, classes, variables, etc.) that
            are exposed in these JavaScript files will be visible inside the UDF.
        determinism
            Provides a hint to BigQuery as to whether the query result can be cached.

        Returns
        -------
        Callable
            The wrapped user-defined function.

        Examples
        --------
        >>> from ibis.backends.bigquery import udf
        >>> import ibis.expr.datatypes as dt
        >>> @udf.python(input_type=[dt.double], output_type=dt.double)
        ... def add_one(x):
        ...     return x + 1
        ...
        >>> print(add_one.sql)
        CREATE TEMPORARY FUNCTION add_one_0(x FLOAT64)
        RETURNS FLOAT64
        LANGUAGE js AS """
        'use strict';
        function add_one(x) {
            return (x + 1);
        }
        return add_one(x);
        """;
        >>> @udf.python(
        ...     input_type=[dt.double, dt.double], output_type=dt.Array(dt.double)
        ... )
        ... def my_range(start, stop):
        ...     def gen(start, stop):
        ...         curr = start
        ...         while curr < stop:
        ...             yield curr
        ...             curr += 1
        ...
        ...     result = []
        ...     for value in gen(start, stop):
        ...         result.append(value)
        ...     return result
        >>> print(my_range.sql)
        CREATE TEMPORARY FUNCTION my_range_0(start FLOAT64, stop FLOAT64)
        RETURNS ARRAY<FLOAT64>
        LANGUAGE js AS """
        'use strict';
        function my_range(start, stop) {
            function* gen(start, stop) {
                let curr = start;
                while ((curr < stop)) {
                    yield curr;
                    curr += 1;
                }
            }
            let result = [];
            for (let value of gen(start, stop)) {
                result.push(value);
            }
            return result;
        }
        return my_range(start, stop);
        """;
        >>> @udf.python(
        ...     input_type=[dt.double, dt.double],
        ...     output_type=dt.Struct.from_tuples(
        ...         [("width", "double"), ("height", "double")]
        ...     ),
        ... )
        ... def my_rectangle(width, height):
        ...     class Rectangle:
        ...         def __init__(self, width, height):
        ...             self.width = width
        ...             self.height = height
        ...
        ...         @property
        ...         def area(self):
        ...             return self.width * self.height
        ...
        ...         def perimeter(self):
        ...             return 2 * (self.width + self.height)
        ...
        ...     return Rectangle(width, height)
        >>> print(my_rectangle.sql)
        CREATE TEMPORARY FUNCTION my_rectangle_0(width FLOAT64, height FLOAT64)
        RETURNS STRUCT<width FLOAT64, height FLOAT64>
        LANGUAGE js AS """
        'use strict';
        function my_rectangle(width, height) {
            class Rectangle {
                constructor(width, height) {
                    this.width = width;
                    this.height = height;
                }
                get area() {
                    return (this.width * this.height);
                }
                perimeter() {
                    return (2 * (this.width + this.height));
                }
            }
            return (new Rectangle(width, height));
        }
        return my_rectangle(width, height);
        """;
        '''
        validate_output_type(output_type)

        if libraries is None:
            libraries = []

        def wrapper(f):
            if not callable(f):
                raise TypeError(f"f must be callable, got {f}")

            signature = inspect.signature(f)
            parameter_names = signature.parameters.keys()
            source = PythonToJavaScriptTranslator(f).compile()
            args = ", ".join(parameter_names)
            strict_str = repr("use strict") + ";\n" if strict else ""
            function_body = f"""\
{strict_str}{source}
return {f.__name__}({args});\
"""

            return self.js(
                name=f.__name__,
                params=(dict(zip(parameter_names, input_type))),
                output_type=output_type,
                body=function_body,
                libraries=libraries,
                determinism=determinism,
            )

        return wrapper

    @staticmethod
    def js(
        name: str,
        params: Mapping[str, dt.DataType],
        output_type: dt.DataType,
        body: str,
        libraries: Iterable[str] | None = None,
        determinism: bool | None = None,
    ) -> Callable:
        '''Define a Javascript UDF for BigQuery.

        `INT64` is not supported as an argument type or a return type, as per
        [the BigQuery documentation](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions#sql-type-encodings-in-javascript).

        Parameters
        ----------
        name:
            The name of the function.
        params
            Mapping of names and types of parameters
        output_type
            Return type of the UDF.
        body:
            The code of the function.
        libraries
            An iterable of Google Cloud Storage URIs containing to JavaScript source
            code. Note that any symbols (functions, classes, variables, etc.) that
            are exposed in these JavaScript files will be visible inside the UDF.
        determinism
            Provides a hint to BigQuery as to whether the query result can be cached.

        Returns
        -------
        Callable
            The user-defined function.

        Examples
        --------
        >>> from ibis.backends.bigquery import udf
        >>> import ibis.expr.datatypes as dt
        >>> add_one = udf.js(
        ...     name="add_one",
        ...     params={"a": dt.double},
        ...     output_type=dt.double,
        ...     body="return x + 1",
        ... )
        >>> print(add_one.sql)
        CREATE TEMPORARY FUNCTION add_one_0(x FLOAT64)
        RETURNS FLOAT64
        LANGUAGE js AS """
        return x + 1
        """;
        '''
        validate_output_type(output_type)
        if any(
            type_ == dt.int64
            for param_type in params.values()
            for type_ in spread_type(param_type)
        ) or any(type_ == dt.int64 for type_ in spread_type(output_type)):
            raise TypeError(
                "BigQuery does not support INT64 as an argument type or a return type "
                "for UDFs. Replace INT64 with FLOAT64 in your UDF signature and "
                "cast all INT64 inputs to FLOAT64."
            )

        if libraries is None:
            libraries = []

        bigquery_signature = ", ".join(
            f"{name} {BigQueryType.from_ibis(dt.dtype(type_))}"
            for name, type_ in params.items()
        )
        return_type = BigQueryType.from_ibis(dt.dtype(output_type))
        libraries_opts = (
            f"\nOPTIONS (\n    library={list(libraries)!r}\n)" if libraries else ""
        )
        determinism_formatted = {
            True: "DETERMINISTIC\n",
            False: "NOT DETERMINISTIC\n",
            None: "",
        }.get(determinism)

        name = _make_udf_name(name)
        sql_code = f'''\
CREATE TEMPORARY FUNCTION {name}({bigquery_signature})
RETURNS {return_type}
{determinism_formatted}LANGUAGE js AS """
{body}
"""{libraries_opts};'''

        udf_node_fields = {
            name: rlz.ValueOf(None if type_ == "ANY TYPE" else type_)
            for name, type_ in params.items()
        }

        udf_node_fields["dtype"] = output_type
        udf_node_fields["shape"] = rlz.shape_like("args")
        udf_node_fields["sql"] = sql_code

        udf_node = type(name, (BigQueryUDFNode,), udf_node_fields)

        from ibis.backends.bigquery.compiler import compiles

        @compiles(udf_node)
        def compiles_udf_node(t, op):
            args = ", ".join(map(t.translate, op.args))
            return f"{udf_node.__name__}({args})"

        def wrapped(*args, **kwargs):
            node = udf_node(*args, **kwargs)
            return node.to_expr()

        wrapped.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
                for param in params.keys()
            ]
        )
        wrapped.__name__ = name
        wrapped.sql = sql_code
        return wrapped

    @staticmethod
    def sql(
        name: str,
        params: Mapping[str, dt.DataType | Literal["ANY TYPE"]],
        output_type: dt.DataType,
        sql_expression: str,
    ) -> Callable:
        """Define a SQL UDF for BigQuery.

        Parameters
        ----------
        name:
            The name of the function.
        params
            Mapping of names and types of parameters
        output_type
            Return type of the UDF.
        sql_expression
            The SQL expression that defines the function.

        Returns
        -------
        Callable
            The wrapped user-defined function.

        Examples
        --------
        >>> from ibis.backends.bigquery import udf
        >>> import ibis.expr.datatypes as dt
        >>> add_one = udf.sql(
        ...     name="add_one",
        ...     params={"x": dt.double},
        ...     output_type=dt.double,
        ...     sql_expression="x + 1",
        ... )
        >>> print(add_one.sql)
        CREATE TEMPORARY FUNCTION add_one_0(x FLOAT64)
        RETURNS FLOAT64
        AS (x + 1)
        """
        validate_output_type(output_type)
        udf_node_fields = {
            name: rlz.ValueOf(None if type_ == "ANY TYPE" else type_)
            for name, type_ in params.items()
        }
        return_type = BigQueryType.from_ibis(dt.dtype(output_type))

        bigquery_signature = ", ".join(
            "{name} {type}".format(
                name=name,
                type="ANY TYPE"
                if type_ == "ANY TYPE"
                else BigQueryType.from_ibis(dt.dtype(type_)),
            )
            for name, type_ in params.items()
        )
        name = _make_udf_name(name)
        sql_code = f"""\
CREATE TEMPORARY FUNCTION {name}({bigquery_signature})
RETURNS {return_type}
AS ({sql_expression});"""

        udf_node_fields["dtype"] = output_type
        udf_node_fields["shape"] = rlz.shape_like("args")
        udf_node_fields["sql"] = sql_code

        udf_node = type(name, (BigQueryUDFNode,), udf_node_fields)

        from ibis.backends.bigquery.compiler import compiles

        @compiles(udf_node)
        def compiles_udf_node(t, op):
            args = ", ".join(map(t.translate, op.args))
            return f"{udf_node.__name__}({args})"

        def wrapper(*args, **kwargs):
            node = udf_node(*args, **kwargs)
            return node.to_expr()

        return wrapper


udf = _BigQueryUDF()
