"""Generic value operations."""

from __future__ import annotations

import itertools
from typing import Annotated, Any, Optional
from typing import Literal as LiteralType

from public import public
from typing_extensions import TypeVar

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.deferred import Deferred
from ibis.common.grounds import Singleton
from ibis.common.patterns import InstanceOf, Length
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Scalar, Unary, Value
from ibis.expr.operations.relations import Relation  # noqa: TC001


@public
class RowID(Value):
    """The row number of the returned result."""

    name = "rowid"
    table: Relation

    shape = ds.columnar
    dtype = dt.int64

    @attribute
    def relations(self):
        return frozenset({self.table})


@public
class Cast(Value):
    """Explicitly cast a value to a specific data type."""

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
    """Try to cast a value to a specific data type."""

    arg: Value
    to: dt.DataType

    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        return self.to


@public
class TypeOf(Unary):
    """Return the _database_ data type of the input expression."""

    dtype = dt.string


@public
class IsNull(Unary):
    """Return true if values are null."""

    dtype = dt.boolean


@public
class NotNull(Unary):
    """Returns true if values are not null."""

    dtype = dt.boolean


@public
class NullIf(Value):
    """Return NULL if an expression equals some specific value."""

    arg: Value
    null_if_expr: Value

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class Coalesce(Value):
    """Return the first non-null expression from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Greatest(Value):
    """Return the largest value from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Least(Value):
    """Return the smallest value from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


T = TypeVar("T", bound=dt.DataType, covariant=True)


@public
class Literal(Scalar[T]):
    """A constant value."""

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


NULL = Literal(None, dt.null)


@public
class ScalarParameter(Scalar):
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
    """A function that produces a constant."""

    shape = ds.scalar


@public
class Impure(Value):
    pass


@public
class TimestampNow(Constant):
    """Return the current timestamp."""

    dtype = dt.timestamp


@public
class DateNow(Constant):
    """Return the current date."""

    dtype = dt.date


@public
class RandomScalar(Impure):
    """Return a random scalar between 0 and 1."""

    dtype = dt.float64
    shape = ds.scalar


@public
class RandomUUID(Impure):
    """Return a random UUID."""

    dtype = dt.uuid
    shape = ds.scalar


@public
class E(Constant):
    """The mathematical constant e."""

    dtype = dt.float64


@public
class Pi(Constant):
    """The mathematical constant pi."""

    dtype = dt.float64


@public
class Hash(Value):
    """Return the hash of a value."""

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


@public
class HexDigest(Value):
    """Return the hexadecimal digest of a value."""

    arg: Value[dt.String | dt.Binary]
    how: LiteralType[
        "md5",
        "sha1",
        "sha256",
        "sha512",
    ]

    dtype = dt.str
    shape = rlz.shape_like("arg")


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    """Simple case statement."""

    base: Value
    cases: Annotated[VarTuple[Value], Length(at_least=1)]
    results: Annotated[VarTuple[Value], Length(at_least=1)]
    default: Value

    def __init__(self, base, cases, results, default):
        assert len(cases) == len(results)
        for case in cases:
            if not rlz.comparable(base, case):
                raise TypeError(
                    f"Base expression {rlz.arg_type_error_format(base)} and "
                    f"case {rlz.arg_type_error_format(case)} are not comparable"
                )
        super().__init__(base=base, cases=cases, results=results, default=default)

    @attribute
    def shape(self):
        exprs = [self.base, *self.cases, *self.results, self.default]
        return rlz.highest_precedence_shape(exprs)

    @attribute
    def dtype(self):
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    """Searched case statement."""

    cases: Annotated[VarTuple[Value[dt.Boolean]], Length(at_least=1)]
    results: Annotated[VarTuple[Value], Length(at_least=1)]
    default: Value

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, default=default)

    @attribute
    def shape(self):
        return rlz.highest_precedence_shape((*self.cases, *self.results, self.default))

    @attribute
    def dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)


public(NULL=NULL)
