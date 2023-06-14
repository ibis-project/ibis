from __future__ import annotations

import abc
import itertools
from typing import Any, Optional, Union
from typing import Literal as LiteralType

from public import public
from typing_extensions import TypeVar

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.grounds import Singleton
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Named, Scalar, Unary, Value
from ibis.expr.operations.relations import Relation  # noqa: TCH001


@public
class TableColumn(Value, Named):
    """Selects a column from a `Table`."""

    table: Relation
    name: Union[str, int]

    output_shape = ds.columnar

    def __init__(self, table, name):
        if isinstance(name, int):
            name = table.schema.name_at_position(name)

        if name not in table.schema:
            columns_formatted = ', '.join(map(repr, table.schema.names))
            raise com.IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )

        super().__init__(table=table, name=name)

    @property
    def output_dtype(self):
        return self.table.schema[self.name]


@public
class RowID(Value, Named):
    """The row number (an autonumeric) of the returned result."""

    name = "rowid"
    table: Relation

    output_shape = ds.columnar
    output_dtype = dt.int64


@public
class TableArrayView(Value, Named):
    """Helper operation class for creating scalar subqueries."""

    table: Relation

    output_shape = ds.columnar

    @property
    def output_dtype(self):
        return self.table.schema[self.name]

    @property
    def name(self):
        return self.table.schema.names[0]


@public
class Cast(Value):
    """Explicitly cast value to a specific data type."""

    arg: Value
    to: dt.DataType

    output_shape = rlz.shape_like("arg")

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.arg.name}, {self.to})"

    @property
    def output_dtype(self):
        return self.to


@public
class TryCast(Value):
    """Explicitly try cast value to a specific data type."""

    arg: Value
    to: dt.DataType

    output_shape = rlz.shape_like("arg")

    @property
    def output_dtype(self):
        return self.to


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
    """Returns true if values are not null.

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
    """Set values to ifnull_expr if they are equal to NULL."""

    arg: Value
    ifnull_expr: Value

    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class NullIf(Value):
    """Set values to NULL if they equal the null_if_expr."""

    arg: Value
    null_if_expr: Value

    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class Coalesce(Value):
    arg: VarTuple[Value]

    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


@public
class Greatest(Value):
    arg: VarTuple[Value]

    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


@public
class Least(Value):
    arg: VarTuple[Value]

    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


T = TypeVar("T", bound=dt.DataType, covariant=True)


@public
class Literal(Scalar):
    value: Any
    dtype: dt.DataType

    output_shape = ds.scalar

    def __init__(self, value, dtype):
        # normalize ensures that the value is a valid value for the given dtype
        value = dt.normalize(dtype, value)
        super().__init__(value=value, dtype=dtype)

    @property
    def name(self):
        return repr(self.value)

    @property
    def output_dtype(self) -> T:
        return self.dtype


@public
class ScalarParameter(Scalar, Named):
    _counter = itertools.count()

    dtype: dt.DataType
    counter: Optional[int] = None

    output_shape = ds.scalar

    def __init__(self, dtype, counter):
        if counter is None:
            counter = next(self._counter)
        super().__init__(dtype=dtype, counter=counter)

    @property
    def name(self):
        return f'param_{self.counter:d}'

    @property
    def output_dtype(self):
        return self.dtype


@public
class Constant(Scalar, Singleton):
    output_shape = ds.scalar


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
class Hash(Value):
    arg: Value

    output_dtype = dt.int64
    output_shape = rlz.shape_like("arg")


@public
class HashBytes(Value):
    arg: Value[dt.String | dt.Binary]
    how: LiteralType[
        "md5",
        "MD5",
        "sha1",
        "SHA1",
        "SHA224",
        "sha256",
        "SHA256",
        "sha512",
        "intHash32",
        "intHash64",
        "cityHash64",
        "sipHash64",
        "sipHash128",
    ]

    output_dtype = dt.binary
    output_shape = rlz.shape_like("arg")


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    base: Value
    cases: VarTuple[Value]
    results: VarTuple[Value]
    default: Value

    output_shape = rlz.shape_like("base")

    def __init__(self, cases, results, **kwargs):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, **kwargs)

    @attribute.default
    def output_dtype(self):
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    cases: VarTuple[Value[dt.Boolean]]
    results: VarTuple[Value]
    default: Value

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, default=default)

    @attribute.default
    def output_shape(self):
        # TODO(kszucs): can be removed after making Sequence iterable
        return rlz.highest_precedence_shape(self.cases)

    @attribute.default
    def output_dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)


class _Negatable(abc.ABC):
    @abc.abstractmethod
    def negate(self):  # pragma: no cover
        ...
