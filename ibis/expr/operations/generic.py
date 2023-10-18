from __future__ import annotations

import itertools
from typing import Annotated, Any, Optional, Union
from typing import Literal as LiteralType

from public import public
from typing_extensions import TypeVar

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.deferred import Deferred  # noqa: TCH001
from ibis.common.grounds import Singleton
from ibis.common.patterns import InstanceOf, Length  # noqa: TCH001
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Named, Scalar, Unary, Value
from ibis.expr.operations.relations import Relation  # noqa: TCH001


@public
class TableColumn(Value, Named):
    """Selects a column from a `Table`."""

    table: Relation
    name: Union[str, int]

    shape = ds.columnar

    def __init__(self, table, name):
        if isinstance(name, int):
            name = table.schema.name_at_position(name)

        if name not in table.schema:
            columns_formatted = ", ".join(map(repr, table.schema.names))
            raise com.IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )

        super().__init__(table=table, name=name)

    @property
    def dtype(self):
        return self.table.schema[self.name]


@public
class RowID(Value, Named):
    """The row number (an autonumeric) of the returned result."""

    name = "rowid"
    table: Relation

    shape = ds.columnar
    dtype = dt.int64


@public
class TableArrayView(Value, Named):
    """Helper operation class for creating scalar subqueries."""

    table: Relation

    shape = ds.columnar

    @property
    def dtype(self):
        return self.table.schema[self.name]

    @property
    def name(self):
        return self.table.schema.names[0]


@public
class Cast(Value):
    """Explicitly cast value to a specific data type."""

    arg: Value
    to: dt.DataType

    shape = rlz.shape_like("arg")

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.arg.name}, {self.to})"

    @property
    def dtype(self):
        return self.to


@public
class TryCast(Value):
    """Explicitly try cast value to a specific data type."""

    arg: Value
    to: dt.DataType

    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        return self.to


@public
class TypeOf(Unary):
    dtype = dt.string


@public
class IsNull(Unary):
    """Return true if values are null.

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are null
    """

    dtype = dt.boolean


@public
class NotNull(Unary):
    """Returns true if values are not null.

    Returns
    -------
    ir.BooleanValue
        Value expression indicating whether values are not null
    """

    dtype = dt.boolean


@public
class NullIf(Value):
    """Set values to NULL if they equal the null_if_expr."""

    arg: Value
    null_if_expr: Value

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class Coalesce(Value):
    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Greatest(Value):
    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Least(Value):
    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


T = TypeVar("T", bound=dt.DataType, covariant=True)


@public
class Literal(Scalar[T]):
    value: Annotated[Any, ~InstanceOf(Deferred)]
    dtype: T

    shape = ds.scalar

    def __init__(self, value, dtype):
        # normalize ensures that the value is a valid value for the given dtype
        value = dt.normalize(dtype, value)
        super().__init__(value=value, dtype=dtype)

    @property
    def name(self):
        if self.dtype.is_interval():
            return f"{self.value!r}{self.dtype.unit.short}"
        return repr(self.value)


@public
class ScalarParameter(Scalar, Named):
    _counter = itertools.count()

    dtype: dt.DataType
    counter: Optional[int] = None

    shape = ds.scalar

    def __init__(self, dtype, counter):
        if counter is None:
            counter = next(self._counter)
        super().__init__(dtype=dtype, counter=counter)

    @property
    def name(self):
        return f"param_{self.counter:d}"


@public
class Constant(Scalar, Singleton):
    shape = ds.scalar


@public
class TimestampNow(Constant):
    dtype = dt.timestamp


@public
class RandomScalar(Constant):
    dtype = dt.float64


@public
class E(Constant):
    dtype = dt.float64


@public
class Pi(Constant):
    dtype = dt.float64


@public
class Hash(Value):
    arg: Value

    dtype = dt.int64
    shape = rlz.shape_like("arg")


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

    dtype = dt.binary
    shape = rlz.shape_like("arg")


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    base: Value
    cases: VarTuple[Value]
    results: VarTuple[Value]
    default: Value

    shape = rlz.shape_like("base")

    def __init__(self, cases, results, **kwargs):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, **kwargs)

    @attribute
    def dtype(self):
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    cases: VarTuple[Value[dt.Boolean]]
    results: VarTuple[Value]
    default: Value

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        if default.dtype.is_null():
            default = Cast(default, rlz.highest_precedence_dtype(results))
        super().__init__(cases=cases, results=results, default=default)

    @attribute
    def shape(self):
        # TODO(kszucs): can be removed after making Sequence iterable
        return rlz.highest_precedence_shape(self.cases)

    @attribute
    def dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)
