from __future__ import annotations

import collections
import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from public import public

from ibis import util
from ibis.expr.types.generic import Column, Scalar, Value, literal
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
    import ibis.expr.operations as ops

    items = dict(value)
    values = items.values()
    if any(isinstance(value, Value) for value in values):
        return ops.StructColumn(
            names=tuple(items.keys()), values=tuple(values)
        ).to_expr()
    return literal(collections.OrderedDict(items), type=type)


@public
class StructValue(Value):
    def __dir__(self):
        return sorted(
            frozenset(itertools.chain(dir(type(self)), self.type().names))
        )

    def __getitem__(self, name: str) -> ir.Value:
        """Extract the `name` field from this struct.

        Parameters
        ----------
        name
            The name of the field to access.

        Returns
        -------
        Value
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

    def __setstate__(self, instance_dictionary):
        self.__dict__ = instance_dictionary

    def __getattr__(self, name: str) -> ir.Value:
        """Extract the `name` field from this struct."""
        if name in self.names:
            return self.__getitem__(name)
        raise AttributeError(name)

    @cached_property
    def names(self) -> Sequence[str]:
        """Return the field names of the struct."""
        return self.type().names

    @cached_property
    def types(self) -> Sequence[dt.DataType]:
        """Return the field types of the struct."""
        return self.type().types

    @cached_property
    def fields(self) -> Mapping[str, dt.DataType]:
        """Return a mapping from field name to field type of the struct."""
        return util.frozendict(self.type().pairs)

    def lift(self) -> ir.Table:
        """Project the fields of `self` into a table.

        This method is useful when analyzing data that has deeply nested
        structs or arrays of structs. `lift` can be chained to avoid repeating
        column names and table references.

        See also [`Table.unpack`][ibis.expr.types.relations.Table.unpack].

        Returns
        -------
        Table
            A projection with this struct expression's fields.

        Examples
        --------
        >>> schema = dict(a="struct<b: float, c: string>", d="string")
        >>> t = ibis.table(schema, name="t")
        >>> t
        UnboundTable: t
          a struct<b: float64, c: string>
          d string
        >>> t.a.lift()
        r0 := UnboundTable: t
          a struct<b: float64, c: string>
          d string

        Selection[r0]
          selections:
            b: StructField(r0.a, field='b')
            c: StructField(r0.a, field='c')
        """
        import ibis.expr.analysis as an

        table = an.find_first_base_table(self)
        return table[[self[name] for name in self.names]]

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
class StructScalar(Scalar, StructValue):
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
class StructColumn(Column, StructValue):
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
class DestructValue(Value):
    """Class that represents a destruct value.

    When assigning a destruct column, the field inside this destruct column
    will be destructured and assigned to multiple columnns.
    """

    def name(self, name):
        res = super().name(name)
        return self.__class__(res.op())


@public
class DestructScalar(Scalar, DestructValue):
    pass


@public
class DestructColumn(Column, DestructValue):
    pass
