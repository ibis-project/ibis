from __future__ import annotations

import collections
import functools
import inspect
import itertools
from typing import Callable, Iterable, Mapping

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.backends.bigquery.compiler import compiles
from ibis.backends.bigquery.datatypes import UDFContext, ibis_type_to_bigquery_type
from ibis.backends.bigquery.operations import BigQueryUDFNode
from ibis.backends.bigquery.udf.core import PythonToJavaScriptTranslator
from ibis.udf.validate import validate_output_type

__all__ = ("udf",)

_udf_name_cache: Mapping[str, Iterable[int]] = collections.defaultdict(itertools.count)


def create_udf_node(name, fields):
    """Create a new UDF node type.

    Parameters
    ----------
    name : str
        Then name of the UDF node
    fields : OrderedDict
        Mapping of class member name to definition

    Returns
    -------
    result : type
        A new BigQueryUDFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = f"{name}_{definition:d}"
    return type(external_name, (BigQueryUDFNode,), fields)


def udf(
    input_type: Iterable[dt.DataType],
    output_type: dt.DataType,
    strict: bool = True,
    libraries: Iterable[str] | None = None,
) -> Callable:
    '''Define a UDF for BigQuery.

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

    Returns
    -------
    Callable
        The wrapped user-defined function.

    Examples
    --------
    >>> from ibis.backends.bigquery import udf
    >>> import ibis.expr.datatypes as dt
    >>> @udf(input_type=[dt.double], output_type=dt.double)
    ... def add_one(x):
    ...     return x + 1
    >>> print(add_one.js)
    CREATE TEMPORARY FUNCTION add_one_0(x FLOAT64)
    RETURNS FLOAT64
    LANGUAGE js AS """
    'use strict';
    function add_one(x) {
        return (x + 1);
    }
    return add_one(x);
    """;
    >>> @udf(input_type=[dt.double, dt.double],
    ...      output_type=dt.Array(dt.double))
    ... def my_range(start, stop):
    ...     def gen(start, stop):
    ...         curr = start
    ...         while curr < stop:
    ...             yield curr
    ...             curr += 1
    ...     result = []
    ...     for value in gen(start, stop):
    ...         result.append(value)
    ...     return result
    >>> print(my_range.js)
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
    >>> @udf(
    ...     input_type=[dt.double, dt.double],
    ...     output_type=dt.Struct.from_tuples([
    ...         ('width', 'double'), ('height', 'double')
    ...     ])
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
    >>> print(my_rectangle.js)
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

        udf_node_fields = {
            name: rlz.value(type) for name, type in zip(parameter_names, input_type)
        }

        udf_node_fields["output_dtype"] = output_type
        udf_node_fields["output_shape"] = rlz.shape_like("args")
        udf_node_fields["__slots__"] = ("js",)

        udf_node = create_udf_node(f.__name__, udf_node_fields)

        @compiles(udf_node)
        def compiles_udf_node(t, op):
            args = ", ".join(map(t.translate, op.args))
            return f"{udf_node.__name__}({args})"

        type_translation_context = UDFContext()
        return_type = ibis_type_to_bigquery_type(
            dt.dtype(output_type), type_translation_context
        )
        bigquery_signature = ", ".join(
            "{name} {type}".format(
                name=name,
                type=ibis_type_to_bigquery_type(
                    dt.dtype(type), type_translation_context
                ),
            )
            for name, type in zip(parameter_names, input_type)
        )
        source = PythonToJavaScriptTranslator(f).compile()
        args = ", ".join(parameter_names)
        strict_str = repr("use strict") + ";\n" if strict else ""
        libraries_opts = (
            f"\nOPTIONS (\n    library={repr(list(libraries))}\n)" if libraries else ""
        )
        js = f'''\
CREATE TEMPORARY FUNCTION {udf_node.__name__}({bigquery_signature})
RETURNS {return_type}
LANGUAGE js AS """
{strict_str}{source}
return {f.__name__}({args});
"""{libraries_opts};'''

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            node = udf_node(*args, **kwargs)
            object.__setattr__(node, "js", js)
            return node.to_expr()

        wrapped.__signature__ = signature
        wrapped.js = js
        return wrapped

    return wrapper
