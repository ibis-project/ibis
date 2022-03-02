from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING, Iterable, Mapping

from public import public

from .generic import AnyColumn, AnyScalar, AnyValue, literal
from .typing import V

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

    def destructure(self) -> list[ir.ValueExpr]:
        """Destructure a ``StructValue`` into the corresponding struct fields.

        When assigned, a destruct value will be destructured and assigned to
        multiple columns.

        Returns
        -------
        list[AnyValue]
            Value expressions corresponding to the struct fields.
        """
        return [self[field_name] for field_name in self.type().names]


@public
class StructScalar(AnyScalar, StructValue):
    pass


@public
class StructColumn(AnyColumn, StructValue):
    pass
