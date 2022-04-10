from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING, Iterable, Mapping

from public import public

from ibis.expr.types.generic import AnyColumn, AnyScalar, AnyValue, literal
from ibis.expr.types.typing import V

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir


@public
def struct(
    value: Iterable[tuple[str, V]] | Mapping[str, V],
    type: str | dt.DataType | None = None,
) -> StructValue:
    """Create a struct literal from a [`dict`][dict] or other mapping.

    Parameters
    ----------
    value
        The underlying data for literal struct value
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`.

    Returns
    -------
    StructScalar
        An expression representing a literal struct (compound type with fields
        of fixed types)

    Examples
    --------
    Create a struct literal from a [`dict`][dict] with the type inferred
    >>> import ibis
    >>> t = ibis.struct(dict(a=1, b='foo'))

    Create a struct literal from a [`dict`][dict] with a specified type
    >>> t = ibis.struct(dict(a=1, b='foo'), type='struct<a: float, b: string>')
    """
    return literal(collections.OrderedDict(value), type=type)


@public
class StructValue(AnyValue):
    def __dir__(self):
        return sorted(
            frozenset(itertools.chain(dir(type(self)), self.type().names))
        )

    def __getitem__(self, name: str) -> ir.ValueExpr:
        """Extract the `name` field from this struct.

        Parameters
        ----------
        name
            The name of the field to access.

        Returns
        -------
        ValueExpr
            An expression with the type of the field being accessed.

        Examples
        --------
        >>> import ibis
        >>> s = ibis.struct(dict(fruit="pear", weight=0))
        >>> s['fruit']
        fruit: StructField(frozendict({'fruit': 'pear', 'weight': 0}), field='fruit')
        """  # noqa: E501
        import ibis.expr.operations as ops

        return ops.StructField(self, name).to_expr().name(name)

    def destructure(self) -> DestructValue:
        """Destructure `self` into a `DestructValue`.

        When assigned, a destruct value will be destructured and assigned to
        multiple columns.

        Returns
        -------
        DestructValue
            A destruct value expression.
        """
        return DestructValue(self._arg)


@public
class StructScalar(AnyScalar, StructValue):
    def destructure(self) -> DestructScalar:
        """Destructure `self` into a `DestructScalar`.

        When assigned, a destruct scalar will be destructured and assigned to
        multiple columns.

        Returns
        -------
        DestructScalar
            A destruct scalar expression.
        """
        return DestructScalar(self._arg)


@public
class StructColumn(AnyColumn, StructValue):
    def destructure(self) -> DestructColumn:
        """Destructure `self` into a `DestructColumn`.

        When assigned, a destruct column will be destructured and assigned to
        multiple columns.

        Returns
        -------
        DestructColumn
            A destruct column expression.
        """
        return DestructColumn(self._arg)


@public
class DestructValue(AnyValue):
    """Class that represents a destruct value.

    When assigning a destruct column, the field inside this destruct column
    will be destructured and assigned to multiple columnns.
    """

    def name(self, name):
        res = super().name(name)
        return self.__class__(res.op())


@public
class DestructScalar(AnyScalar, DestructValue):
    pass


@public
class DestructColumn(AnyColumn, DestructValue):
    pass
