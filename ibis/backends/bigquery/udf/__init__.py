import collections
import functools
import inspect
import itertools

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.udf.validate as v
from ibis.compat import PY38  # noqa: F401
from ibis.expr.signature import Argument as Arg

from ..compiler import BigQueryUDFNode, compiles
from ..datatypes import UDFContext, ibis_type_to_bigquery_type
from ..udf.core import PythonToJavaScriptTranslator

__all__ = ('udf',)

_udf_name_cache = collections.defaultdict(itertools.count)


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
    external_name = '{}_{:d}'.format(name, definition)
    return type(external_name, (BigQueryUDFNode,), fields)


def udf(input_type, output_type, strict=True, libraries=None):
    '''Define a UDF for BigQuery

    Parameters
    ----------
    input_type : List[DataType]
    output_type : DataType
    strict : bool
        Whether or not to put a ``'use strict';`` string at the beginning of
        the UDF. Setting to ``False`` is probably a bad idea.
    libraries : List[str]
        A list of Google Cloud Storage URIs containing to JavaScript source
        code. Note that any symbols (functions, classes, variables, etc.) that
        are exposed in these JavaScript files will be visible inside the UDF.

    Returns
    -------
    wrapper : Callable
        The wrapped function

    Notes
    -----
    - ``INT64`` is not supported as an argument type or a return type, as per
      `the BigQuery documentation
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions#sql-type-encodings-in-javascript>`_.
    - `The follow example doctest doesn't work for Python 3.8
      <https://github.com/ibis-project/ibis/issues/2085>`_.

    Examples
    --------
    >>> if PY38:
    ...     import pytest; pytest.skip("Issue #2085")
    >>> from ibis.bigquery import udf
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
    v.validate_output_type(output_type)

    if libraries is None:
        libraries = []

    def wrapper(f):
        if not callable(f):
            raise TypeError('f must be callable, got {}'.format(f))

        signature = inspect.signature(f)
        parameter_names = signature.parameters.keys()

        udf_node_fields = collections.OrderedDict(
            [
                (name, Arg(rlz.value(type)))
                for name, type in zip(parameter_names, input_type)
            ]
            + [
                (
                    'output_type',
                    lambda self, output_type=output_type: rlz.shape_like(
                        self.args, dtype=output_type
                    ),
                ),
                ('__slots__', ('js',)),
            ]
        )

        udf_node = create_udf_node(f.__name__, udf_node_fields)

        @compiles(udf_node)
        def compiles_udf_node(t, expr):
            return '{}({})'.format(
                udf_node.__name__, ', '.join(map(t.translate, expr.op().args))
            )

        type_translation_context = UDFContext()
        return_type = ibis_type_to_bigquery_type(
            dt.dtype(output_type), type_translation_context
        )
        bigquery_signature = ', '.join(
            '{name} {type}'.format(
                name=name,
                type=ibis_type_to_bigquery_type(
                    dt.dtype(type), type_translation_context
                ),
            )
            for name, type in zip(parameter_names, input_type)
        )
        source = PythonToJavaScriptTranslator(f).compile()
        js = '''\
CREATE TEMPORARY FUNCTION {external_name}({signature})
RETURNS {return_type}
LANGUAGE js AS """
{strict}{source}
return {internal_name}({args});
"""{libraries};'''.format(
            external_name=udf_node.__name__,
            internal_name=f.__name__,
            return_type=return_type,
            source=source,
            signature=bigquery_signature,
            strict=repr('use strict') + ';\n' if strict else '',
            args=', '.join(parameter_names),
            libraries=(
                '\nOPTIONS (\n    library={}\n)'.format(repr(list(libraries)))
                if libraries
                else ''
            ),
        )

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            node = udf_node(*args, **kwargs)
            node.js = js
            return node.to_expr()

        wrapped.__signature__ = signature
        wrapped.js = js
        return wrapped

    return wrapper
