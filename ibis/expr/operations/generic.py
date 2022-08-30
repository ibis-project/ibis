from __future__ import annotations

import abc
import datetime
import decimal
import enum
import ipaddress
import itertools
import uuid
from operator import attrgetter

import numpy as np
import pandas as pd
from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.util as util
from ibis.common import exceptions as com
from ibis.common.grounds import Singleton
from ibis.common.validators import immutable_property
from ibis.expr.operations.core import Node, Unary, Value, distinct_roots
from ibis.util import deprecated, frozendict

try:
    import shapely
except ImportError:
    BaseGeometry = type(None)
else:
    BaseGeometry = shapely.geometry.base.BaseGeometry


@public
class TableColumn(Value):
    """Selects a column from a `Table`."""

    table = rlz.table
    name = rlz.instance_of((str, int))

    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, table, name):
        schema = table.schema()

        if isinstance(name, int):
            name = schema.name_at_position(name)

        if name not in schema:
            raise com.IbisTypeError(
                f"value {name!r} is not a field in {table.columns}"
            )

        super().__init__(table=table, name=name)

    @util.deprecated(version="4.0.0", instead="Use `table` property instead")
    def parent(self):  # pragma: no cover
        return self.table

    def resolve_name(self):
        return self.name

    def has_resolved_name(self):
        return True

    def root_tables(self):
        return self.table.op().root_tables()

    @property
    def output_dtype(self):
        schema = self.table.schema()
        return schema[self.name]


@public
class RowID(Value):
    """The row number (an autonumeric) of the returned result."""

    output_shape = rlz.Shape.COLUMNAR
    output_dtype = dt.int64

    def resolve_name(self):
        return 'rowid'

    def has_resolved_name(self):
        return True


@public
@util.deprecated(version="4.0.0", instead="")
def find_all_base_tables(expr, memo=None):
    if memo is None:
        memo = {}

    node = expr.op()

    if isinstance(expr, ir.Table) and node.blocks():
        if expr not in memo:
            memo[node] = expr
        return memo

    for arg in expr.op().flat_args():
        if isinstance(arg, ir.Expr):
            find_all_base_tables(arg, memo)

    return memo


@public
class TableArrayView(Value):

    """
    (Temporary?) Helper operation class for SQL translation (fully formed table
    subqueries to be viewed as arrays)
    """

    table = rlz.table

    output_shape = rlz.Shape.COLUMNAR

    @property
    def output_dtype(self):
        schema = self.table.schema()
        return schema[self.name]

    @property
    def name(self):
        return self.table.schema().names[0]


@public
class Cast(Value):
    """Explicitly cast value to a specific data type."""

    arg = rlz.any
    to = rlz.datatype

    output_shape = rlz.shape_like("arg")
    output_dtype = property(attrgetter("to"))

    # see #396 for the issue preventing an implementation of resolve_name


@public
class TypeOf(Unary):
    output_dtype = dt.string


@public
class IsNull(Unary):
    """Return true if values are null.

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are null
    """

    output_dtype = dt.boolean


@public
class NotNull(Unary):
    """Returns true if values are not null

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are not null
    """

    output_dtype = dt.boolean


@public
class ZeroIfNull(Unary):
    output_dtype = rlz.dtype_like("arg")


@public
class IfNull(Value):
    """
    Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """

    arg = rlz.any
    ifnull_expr = rlz.any

    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class NullIf(Value):
    """Set values to NULL if they equal the null_if_expr"""

    arg = rlz.any
    null_if_expr = rlz.any
    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class CoalesceLike(Value):

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column
    arg = rlz.value_list_of(rlz.any)

    output_shape = rlz.shape_like('arg')

    @immutable_property
    def output_dtype(self):
        # filter out null types
        non_null_exprs = [arg for arg in self.arg if arg.type() != dt.null]
        if non_null_exprs:
            return rlz.highest_precedence_dtype(non_null_exprs)
        else:
            return dt.null


@public
class Coalesce(CoalesceLike):
    pass


@public
class Greatest(CoalesceLike):
    pass


@public
class Least(CoalesceLike):
    pass


@public
class Literal(Value):
    value = rlz.one_of(
        (
            rlz.instance_of(
                (
                    BaseGeometry,
                    bytes,
                    datetime.date,
                    datetime.datetime,
                    datetime.time,
                    datetime.timedelta,
                    enum.Enum,
                    float,
                    frozenset,
                    int,
                    frozendict,
                    ipaddress.IPv4Address,
                    ipaddress.IPv6Address,
                    np.generic,
                    np.ndarray,
                    pd.Timedelta,
                    pd.Timestamp,
                    str,
                    tuple,
                    type(None),
                    uuid.UUID,
                    decimal.Decimal,
                )
            ),
            rlz.is_computable_input,
        )
    )
    dtype = rlz.datatype

    output_shape = rlz.Shape.SCALAR
    output_dtype = property(attrgetter("dtype"))

    def root_tables(self):
        return []


@public
class NullLiteral(Literal, Singleton):
    """Typeless NULL literal"""

    value = rlz.optional(type(None))
    dtype = rlz.optional(rlz.instance_of(dt.Null), default=dt.null)


@public
class ScalarParameter(Value):
    _counter = itertools.count()

    dtype = rlz.datatype
    counter = rlz.optional(
        rlz.instance_of(int), default=lambda: next(ScalarParameter._counter)
    )

    output_shape = rlz.Shape.SCALAR
    output_dtype = property(attrgetter("dtype"))

    def resolve_name(self):
        return f'param_{self.counter:d}'

    def __hash__(self):
        return hash((self.dtype, self.counter))

    @property
    def inputs(self):
        return ()

    def root_tables(self):
        return []


@public
class ValueList(Value):
    """Data structure for a list of value expressions"""

    # NOTE: this proxies the Value behaviour to the underlying values

    values = rlz.tuple_of(rlz.any)

    output_type = ir.ValueList
    output_dtype = rlz.dtype_like("values")
    output_shape = rlz.shape_like("values")

    def root_tables(self):
        return distinct_roots(*self.values)


@public
class Constant(Value):
    output_shape = rlz.Shape.SCALAR


@public
class TimestampNow(Constant):
    output_dtype = dt.timestamp


@public
class RandomScalar(Constant):
    output_dtype = dt.float64


@public
class E(Constant):
    output_dtype = dt.float64


@public
class Pi(Constant):
    output_dtype = dt.float64


@public
class DecimalPrecision(Unary):
    arg = rlz.decimal
    output_dtype = dt.int32


@public
class DecimalScale(Unary):
    arg = rlz.decimal
    output_dtype = dt.int32


@public
class Hash(Value):
    arg = rlz.any
    how = rlz.isin({'fnv', 'farm_fingerprint'})

    output_dtype = dt.int64
    output_shape = rlz.shape_like("arg")


@public
class HashBytes(Value):
    arg = rlz.one_of({rlz.value(dt.string), rlz.value(dt.binary)})
    how = rlz.isin({'md5', 'sha1', 'sha256', 'sha512'})

    output_dtype = dt.binary
    output_shape = rlz.shape_like("arg")


@deprecated(
    instead="do nothing; SummaryFilter is no longer used internally",
    version="4.0.0",
)
@public
class SummaryFilter(Value):
    expr = rlz.instance_of(ir.TopK)

    output_dtype = dt.boolean
    output_shape = rlz.Shape.COLUMNAR


# TODO(kszucs): shouldn't we move this operation to either
# analytic.py or reductions.py?
@public
class TopK(Node):
    arg = rlz.column(rlz.any)
    k = rlz.non_negative_integer
    by = rlz.one_of((rlz.function_of(rlz.base_table_of("arg")), rlz.any))
    output_type = ir.TopK

    def blocks(self):  # pragma: no cover
        return True

    def root_tables(self):  # pragma: no cover
        args = (arg for arg in self.flat_args() if isinstance(arg, ir.Expr))
        return distinct_roots(*args)


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    base = rlz.any
    cases = rlz.value_list_of(rlz.any)
    results = rlz.value_list_of(rlz.any)
    default = rlz.any

    output_shape = rlz.shape_like("base")

    def __init__(self, cases, results, **kwargs):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, **kwargs)

    def root_tables(self):
        return distinct_roots(*self.flat_args())

    @immutable_property
    def output_dtype(self):
        # TODO(kszucs): we could extend the functionality of
        # rlz.shape_like to support varargs with .flat_args()
        # to define a subset of input arguments
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    cases = rlz.value_list_of(rlz.boolean)
    results = rlz.value_list_of(rlz.any)
    default = rlz.any

    output_shape = rlz.shape_like("cases")

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, default=default)

    def root_tables(self):
        return distinct_roots(*self.flat_args())

    @immutable_property
    def output_dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)


class _Negatable(abc.ABC):
    @abc.abstractmethod
    def negate(self):  # pragma: no cover
        ...
