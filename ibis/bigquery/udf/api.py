import collections
import inspect

import ibis.expr.rules as rlz

from ibis.compat import functools, signature
from ibis.expr.signature import Argument as Arg

from ibis.bigquery.compiler import BigQueryUDFNode, compiles

from ibis.bigquery.udf.core import (
    PythonToJavaScriptTranslator,
    UDFContext
)
from ibis.bigquery.datatypes import ibis_type_to_bigquery_type


__all__ = 'udf',


def udf(input_type, output_type, strict=True):
    '''Define a UDF for BigQuery

    Parameters
    ----------
    input_type : List[DataType]
    output_type : DataType
    strict : bool
        Whether or not to put a ``'use strict';`` string at the beginning of
        the UDF. Setting to ``False`` is a really bad idea.

    Returns
    -------
    wrapper : Callable
        The wrapped function

    Examples
    --------
    >>> from ibis.bigquery.api import udf
    >>> import ibis.expr.datatypes as dt
    >>> @udf(input_type=[dt.double], output_type=dt.double)
    ... def add_one(x):
    ...     return x + 1
    >>> print(add_one.js)
    CREATE TEMPORARY FUNCTION add_one(x FLOAT64)
    RETURNS FLOAT64
    LANGUAGE js AS """
    'use strict';
    function add_one(x) {
        return (x + 1);
    }
    return add_one(x);
    """;
    >>> @udf(input_type=[dt.int64, dt.int64],
    ...      output_type=dt.Array(dt.int64))
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
    CREATE TEMPORARY FUNCTION my_range(start FLOAT64, stop FLOAT64)
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
    CREATE TEMPORARY FUNCTION my_rectangle(width FLOAT64, height FLOAT64)
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
    def wrapper(f):
        if not callable(f):
            raise TypeError('f must be callable, got {}'.format(f))

        sig = signature(f)
        udf_node_fields = collections.OrderedDict([
            (name, Arg(rlz.value(type)))
            for name, type in zip(sig.parameters.keys(), input_type)
        ] + [
            ('output_type', output_type.array_type),
            ('__slots__', ('js',)),
        ])
        udf_node = type(f.__name__, (BigQueryUDFNode,), udf_node_fields)

        @compiles(udf_node)
        def compiles_udf_node(t, expr):
            return '{}({})'.format(
                udf_node.__name__,
                ', '.join(map(t.translate, expr.op().args))
            )

        source = PythonToJavaScriptTranslator(f).compile()
        type_translation_context = UDFContext()
        js = '''\
CREATE TEMPORARY FUNCTION {name}({signature})
RETURNS {return_type}
LANGUAGE js AS """
{strict}{source}
return {name}({args});
""";'''.format(
            name=f.__name__,
            return_type=ibis_type_to_bigquery_type(
                output_type, type_translation_context),
            source=source,
            signature=', '.join(
                '{name} {type}'.format(
                    name=name,
                    type=ibis_type_to_bigquery_type(
                        type, type_translation_context)
                ) for name, type in zip(
                   inspect.signature(f).parameters.keys(), input_type
                )
            ),
            strict=repr('use strict') + ';\n' if strict else '',
            args=', '.join(inspect.signature(f).parameters.keys())
        )

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            node = udf_node(*args, **kwargs)
            node.js = js
            return node.to_expr()

        wrapped.__signature__ = inspect.signature(f)
        wrapped.js = js

        return wrapped
    return wrapper
