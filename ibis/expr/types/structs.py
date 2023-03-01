from __future__ import annotations

import collections
from keyword import iskeyword
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from public import public

import ibis.expr.operations as ops
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

    fields = dict(value)
    if any(isinstance(value, Value) for value in fields.values()):
        names = tuple(fields.keys())
        values = tuple(fields.values())
        return ops.StructColumn(names=names, values=values).to_expr()
    else:
        return literal(collections.OrderedDict(fields), type=type)


@public
class StructValue(Value):
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
        >>> s = ibis.struct(dict(fruit="pear", weight=0))
        >>> s['fruit']
        fruit: StructField(...)
        """
        return ops.StructField(self, name).to_expr()

    def __setstate__(self, instance_dictionary):
        self.__dict__ = instance_dictionary

    def __getattr__(self, name: str) -> ir.Value:
        """Extract the `name` field from this struct."""
        if name in self.names:
            return self.__getitem__(name)
        raise AttributeError(name)

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
        >>> lines = '''
        ...     {"pos": {"lat": 10.1, "lon": 30.3}}
        ...     {"pos": {"lat": 10.2, "lon": 30.2}}
        ...     {"pos": {"lat": 10.3, "lon": 30.1}}
        ... '''
        >>> with open("/tmp/lines.json", "w") as f:
        ...     _ = f.write(lines)
        >>> t = ibis.read_json("/tmp/lines.json")
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
        [`Table.unpack`][ibis.expr.types.relations.Table.unpack].
        """
        import ibis.expr.analysis as an

        # TODO(kszucs): avoid expression roundtripping
        table = an.find_first_base_table(self.op()).to_expr()
        return table[[self[name] for name in self.names]]

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
class StructScalar(Scalar, StructValue):
    pass


@public
class StructColumn(Column, StructValue):
    pass
