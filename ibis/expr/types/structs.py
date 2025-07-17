from __future__ import annotations

import collections
from keyword import iskeyword
from typing import TYPE_CHECKING, Any

from public import public

import ibis.expr.operations as ops
from ibis.common.deferred import deferrable
from ibis.common.exceptions import IbisError
from ibis.expr.types.generic import Column, Scalar, Value, literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir


@public
@deferrable
def struct(
    value: Iterable[tuple[str, Value | Any]] | Mapping[str, Value | Any],
    *,
    type: str | dt.DataType | None = None,
) -> StructValue:
    """Create a struct expression.

    If any of the inputs are Columns, then the output will be a `StructColumn`.
    Otherwise, the output will be a `StructScalar`.

    Parameters
    ----------
    value
        Either a `{str: Value}` mapping, or an iterable of tuples of the form
        `(str, Value)`.
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the Ibis type of `value`. This is only used if all of the input values
        are Python literals. eg `struct<a: float, b: string>`.

    Returns
    -------
    StructValue
        An StructScalar or StructColumn expression.

    Examples
    --------
    Create a struct scalar literal from a `dict` with the type inferred

    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.struct(dict(a=1, b="foo"))
    ┌──────────────────────┐
    │ {'a': 1, 'b': 'foo'} │
    └──────────────────────┘

    Specify a type (note the 1 is now a `float`):

    >>> ibis.struct(dict(a=1, b="foo"), type="struct<a: float, b: string>")
    ┌────────────────────────┐
    │ {'a': 1.0, 'b': 'foo'} │
    └────────────────────────┘

    Create a struct column from a column and a scalar literal

    >>> t = ibis.memtable({"a": [1, 2, 3]})
    >>> ibis.struct([("a", t.a), ("b", "foo")])
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ StructColumn({'a': a, 'b': 'foo'}) ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ struct<a: int64, b: string>        │
    ├────────────────────────────────────┤
    │ {'a': 1, 'b': 'foo'}               │
    │ {'a': 2, 'b': 'foo'}               │
    │ {'a': 3, 'b': 'foo'}               │
    └────────────────────────────────────┘
    """
    import ibis.expr.operations as ops

    fields = dict(value)
    if any(isinstance(value, Value) for value in fields.values()):
        names = tuple(fields.keys())
        values = tuple(fields.values())
        return ops.StructColumn(names=names, values=values).to_expr()
    else:
        return literal(collections.OrderedDict(fields), type=type)


@public
class StructValue(Value):
    """A Struct is a nested type with ordered fields of any type.

    For example, a Struct might have a field `a` of type `int64` and a field `b`
    of type `string`.

    Structs can be constructed with [`ibis.struct()`](#ibis.expr.types.struct).

    Examples
    --------
    Construct a `Struct` column with fields `a: int64` and `b: string`

    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t = ibis.memtable({"s": [{"a": 1, "b": "foo"}, {"a": 3, "b": None}, None]})
    >>> t
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ s                           ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ struct<a: int64, b: string> │
    ├─────────────────────────────┤
    │ {'a': 1, 'b': 'foo'}        │
    │ {'a': 3, 'b': None}         │
    │ NULL                        │
    └─────────────────────────────┘

    You can use dot notation (`.`) or square-bracket syntax (`[]`) to access
    struct column fields

    >>> t.s.a
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     3 │
    │  NULL │
    └───────┘
    >>> t.s["a"]
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     3 │
    │  NULL │
    └───────┘
    """

    def __dir__(self):
        out = set(dir(type(self)))
        out.update(
            c for c in self.type().names if c.isidentifier() and not iskeyword(c)
        )
        return sorted(out)

    def _ipython_key_completions_(self) -> list[str]:
        return sorted(self.type().names)

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
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": [{"a": 1, "b": "foo"}, {"a": 3, "b": None}, None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ s                           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ struct<a: int64, b: string> │
        ├─────────────────────────────┤
        │ {'a': 1, 'b': 'foo'}        │
        │ {'a': 3, 'b': None}         │
        │ NULL                        │
        └─────────────────────────────┘
        >>> t.s["a"]
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     3 │
        │  NULL │
        └───────┘
        >>> t.s["b"]
        ┏━━━━━━━━┓
        ┃ b      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ foo    │
        │ NULL   │
        │ NULL   │
        └────────┘
        >>> t.s["foo_bar"]
        Traceback (most recent call last):
            ...
        KeyError: 'foo_bar'
        """
        if name not in self.names:
            raise KeyError(name)

        op = self.op()

        # if the underlying operation is a simple struct column access, then
        # just inline the underlying field access
        if isinstance(op, ops.StructColumn):
            return op.values[op.names.index(name)].to_expr()
        # and then do the same if the underlying value is a field access
        elif isinstance(op, ops.Literal):
            return ops.Literal(
                op.value[name] if op.value is not None else None,
                dtype=self.fields[name],
            ).to_expr()
        else:
            return ops.StructField(self, name).to_expr()

    def __setstate__(self, instance_dictionary):
        self.__dict__ = instance_dictionary

    def __getattr__(self, name: str) -> ir.Value:
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
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": [{"a": 1, "b": "foo"}, {"a": 3, "b": None}, None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ s                           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ struct<a: int64, b: string> │
        ├─────────────────────────────┤
        │ {'a': 1, 'b': 'foo'}        │
        │ {'a': 3, 'b': None}         │
        │ NULL                        │
        └─────────────────────────────┘
        >>> t.s.a
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     3 │
        │  NULL │
        └───────┘
        >>> t.s.b
        ┏━━━━━━━━┓
        ┃ b      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ foo    │
        │ NULL   │
        │ NULL   │
        └────────┘
        >>> t.s.foo_bar
        Traceback (most recent call last):
            ...
        AttributeError: foo_bar
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def names(self) -> Sequence[str]:
        """Return the field names of the struct."""
        return self.type().names

    @property
    def types(self) -> Sequence[dt.DataType]:
        """Return the field types of the struct."""
        return self.type().types

    @property
    def fields(self) -> Mapping[str, dt.DataType]:
        """Return a mapping from field name to field type of the struct."""
        return self.type().fields

    def lift(self) -> ir.Table:
        """Project the fields of `self` into a table.

        This method is useful when analyzing data that has deeply nested
        structs or arrays of structs. `lift` can be chained to avoid repeating
        column names and table references.

        Returns
        -------
        Table
            A projection with this struct expression's fields.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "pos": [
        ...             {"lat": 10.1, "lon": 30.3},
        ...             {"lat": 10.2, "lon": 30.2},
        ...             {"lat": 10.3, "lon": 30.1},
        ...         ]
        ...     }
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ pos                                ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ struct<lat: float64, lon: float64> │
        ├────────────────────────────────────┤
        │ {'lat': 10.1, 'lon': 30.3}         │
        │ {'lat': 10.2, 'lon': 30.2}         │
        │ {'lat': 10.3, 'lon': 30.1}         │
        └────────────────────────────────────┘
        >>> t.pos.lift()
        ┏━━━━━━━━━┳━━━━━━━━━┓
        ┃ lat     ┃ lon     ┃
        ┡━━━━━━━━━╇━━━━━━━━━┩
        │ float64 │ float64 │
        ├─────────┼─────────┤
        │    10.1 │    30.3 │
        │    10.2 │    30.2 │
        │    10.3 │    30.1 │
        └─────────┴─────────┘

        See Also
        --------
        [`Table.unpack`](./expression-tables.qmd#ibis.expr.types.relations.Table.unpack)
        """
        try:
            (table,) = self.op().relations
        except ValueError:
            raise IbisError("StructValue must depend on exactly one table")

        return table.to_expr().select([self[name] for name in self.names])


@public
class StructScalar(Scalar, StructValue):
    pass


@public
class StructColumn(Column, StructValue):
    def __getitem__(self, name: str) -> ir.Column:
        return StructValue.__getitem__(self, name)
