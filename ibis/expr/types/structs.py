from __future__ import annotations

import collections
from keyword import iskeyword
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.generic import Column, Scalar, Value, literal
from ibis.expr.types.typing import V
from ibis.common.deferred import deferrable

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir


@public
@deferrable
def struct(
    value: Iterable[tuple[str, V]] | Mapping[str, V],
    type: str | dt.DataType | None = None,
) -> StructValue:
    """Create a struct expression.

    If the input expressions are all column expressions, then the output will be
    a `StructColumn`.

    If the input expressions are Python literals, then the output will be a
    `StructScalar`.

    Parameters
    ----------
    value
        The underlying data for literal struct value or a pairs of field names
        and column expressions.
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`. This is only used if all of the input values
        are literals.

    Returns
    -------
    StructValue
        An expression representing a literal or column struct (compound type with
        fields of fixed types)

    Examples
    --------
    Create a struct literal from a [](`dict`) with the type inferred
    >>> import ibis
    >>> t = ibis.struct(dict(a=1, b="foo"))

    Create a struct literal from a [](`dict`) with a specified type
    >>> t = ibis.struct(dict(a=1, b="foo"), type="struct<a: float, b: string>")

    Specify a specific type for the struct literal
    >>> t = ibis.struct(dict(a=1, b=40), type="struct<a: float, b: int32>")

    Create a struct array from multiple arrays
    >>> ibis.options.interactive = True
    >>> t = ibis.memtable({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    >>> ibis.struct([("a", t.a), ("b", t.b)])
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ StructColumn()              ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ struct<a: int64, b: string> │
    ├─────────────────────────────┤
    │ {'a': 1, 'b': 'foo'}        │
    │ {'a': 2, 'b': 'bar'}        │
    │ {'a': 3, 'b': 'baz'}        │
    └─────────────────────────────┘

    Create a struct array from columns and literals
    >>> ibis.struct([("a", t.a), ("b", "foo")])
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ StructColumn()              ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ struct<a: int64, b: string> │
    ├─────────────────────────────┤
    │ {'a': 1, 'b': 'foo'}        │
    │ {'a': 2, 'b': 'foo'}        │
    │ {'a': 3, 'b': 'foo'}        │
    └─────────────────────────────┘
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
    """A struct literal or column.

    Can be constructed with [`ibis.struct()`](#ibis.expr.types.struct).

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

    Can use either `.` or `[]` to access fields:

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
        import ibis.expr.analysis as an

        # TODO(kszucs): avoid expression roundtripping
        table = an.find_first_base_table(self.op()).to_expr()
        return table[[self[name] for name in self.names]]

    def destructure(self) -> list[ir.Value]:
        """Destructure a ``StructValue`` into the corresponding struct fields.

        When assigned, a destruct value will be destructured and assigned to
        multiple columns.

        Returns
        -------
        list[AnyValue]
            Value expressions corresponding to the struct fields.

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
        >>> a, b = t.s.destructure()
        >>> a
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     3 │
        │  NULL │
        └───────┘
        >>> b
        ┏━━━━━━━━┓
        ┃ b      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ foo    │
        │ NULL   │
        │ NULL   │
        └────────┘
        """
        return [self[field_name] for field_name in self.type().names]


@public
class StructScalar(Scalar, StructValue):
    pass


@public
class StructColumn(Column, StructValue):
    def __getitem__(self, name: str) -> ir.Column:
        return StructValue.__getitem__(self, name)
