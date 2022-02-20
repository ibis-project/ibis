from __future__ import annotations

from typing import Any

from public import public

import ibis
import ibis.common.exceptions as com

from .. import datatypes as dt
from .core import Expr


@public
class ValueExpr(Expr):
    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    def __init__(self, arg, dtype, name=None):
        super().__init__(arg)
        self._name = name
        self._dtype = dtype

    def equals(self, other, cache=None):
        return (
            isinstance(other, ValueExpr)
            and self._name == other._name
            and self._dtype == other._dtype
            and super().equals(other, cache=cache)
        )

    def has_name(self):
        if self._name is not None:
            return True
        return self.op().has_resolved_name()

    def get_name(self):
        if self._name is not None:
            # This value has been explicitly named
            return self._name

        # In some but not all cases we can get a name from the node that
        # produces the value
        return self.op().resolve_name()

    def name(self, name):
        return self._factory(self._arg, name=name)

    def type(self):
        return self._dtype

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, dtype=self.type(), name=name)

        return factory


@public
class ScalarExpr(ValueExpr):
    def _type_display(self):
        return str(self.type())

    def to_projection(self):
        """
        Promote this column expression to a table projection
        """
        from .relations import TableExpr

        roots = self.op().root_tables()
        if len(roots) > 1:
            raise com.RelationError(
                'Cannot convert scalar expression '
                'involving multiple base table references '
                'to a projection'
            )

        table = TableExpr(roots[0])
        return table.projection([self])

    def _repr_html_(self) -> str | None:
        return None


@public
class ColumnExpr(ValueExpr):
    def _type_display(self):
        return str(self.type())

    def parent(self):
        return self._arg

    def to_projection(self):
        """
        Promote this column expression to a table projection
        """
        from .relations import TableExpr

        roots = self.op().root_tables()
        if len(roots) > 1:
            raise com.RelationError(
                'Cannot convert array expression '
                'involving multiple base table references '
                'to a projection'
            )

        table = TableExpr(roots[0])
        return table.projection([self])

    def _repr_html_(self) -> str | None:
        if not ibis.options.interactive:
            return None

        return self.execute().to_frame()._repr_html_()


@public
class AnyValue(ValueExpr):
    pass  # noqa: E701,E302


@public
class AnyScalar(ScalarExpr, AnyValue):
    pass  # noqa: E701,E302


@public
class AnyColumn(ColumnExpr, AnyValue):
    pass  # noqa: E701,E302


@public
class NullValue(AnyValue):
    pass  # noqa: E701,E302


@public
class NullScalar(AnyScalar, NullValue):
    pass  # noqa: E701,E302


@public
class NullColumn(AnyColumn, NullValue):
    pass  # noqa: E701,E302


@public
class ListExpr(ColumnExpr, AnyValue):
    @property
    def values(self):
        return self.op().values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __add__(self, other):
        other_values = tuple(getattr(other, 'values', other))
        return type(self.op())(self.values + other_values).to_expr()

    def __radd__(self, other):
        other_values = tuple(getattr(other, 'values', other))
        return type(self.op())(other_values + self.values).to_expr()

    def __bool__(self):
        return bool(self.values)

    __nonzero__ = __bool__

    def __len__(self):
        return len(self.values)


_NULL = None


@public
def null():
    """Create a NULL/NA scalar"""
    import ibis.expr.operations as ops

    global _NULL
    if _NULL is None:
        _NULL = ops.NullLiteral().to_expr()

    return _NULL


@public
def literal(value: Any, type: dt.DataType | str | None = None) -> ScalarExpr:
    """Create a scalar expression from a Python value.

    !!! tip "Use specific functions for arrays, structs and maps"

        Ibis supports literal construction of arrays using the following
        functions:

        1. [`ibis.array`][ibis.array]
        1. [`ibis.struct`][ibis.struct]
        1. [`ibis.map`][ibis.map]

        Constructing these types using `literal` will be deprecated in a future
        release.

    Parameters
    ----------
    value
        A Python value
    type
        An instance of [`DataType`][ibis.expr.datatypes.DataType] or a string
        indicating the ibis type of `value`. This parameter can be used
        in cases where ibis's type inference isn't sufficient for discovering
        the type of `value`.

    Returns
    -------
    ScalarExpr
        An expression representing a literal value

    Examples
    --------
    Construct an integer literal

    >>> import ibis
    >>> x = ibis.literal(42)
    >>> x.type()
    int8

    Construct a `float64` literal from an `int`

    >>> y = ibis.literal(42, type='double')
    >>> y.type()
    float64

    Ibis checks for invalid types

    >>> ibis.literal('foobar', type='int64')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Value 'foobar' cannot be safely coerced to int64
    """
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops

    if hasattr(value, 'op') and isinstance(value.op(), ops.Literal):
        return value

    try:
        inferred_dtype = dt.infer(value)
    except com.InputTypeError:
        has_inferred = False
    else:
        has_inferred = True

    if type is None:
        has_explicit = False
    else:
        has_explicit = True
        explicit_dtype = dt.dtype(type)

    if has_explicit and has_inferred:
        try:
            # ensure type correctness: check that the inferred dtype is
            # implicitly castable to the explicitly given dtype and value
            dtype = inferred_dtype.cast(explicit_dtype, value=value)
        except com.IbisTypeError:
            raise TypeError(
                f'Value {value!r} cannot be safely coerced to {type}'
            )
    elif has_explicit:
        dtype = explicit_dtype
    elif has_inferred:
        dtype = inferred_dtype
    else:
        raise TypeError(
            'The datatype of value {!r} cannot be inferred, try '
            'passing it explicitly with the `type` keyword.'.format(value)
        )

    if dtype is dt.null:
        return null().cast(dtype)
    else:
        value = dt._normalize(dtype, value)
        return ops.Literal(value, dtype=dtype).to_expr()
