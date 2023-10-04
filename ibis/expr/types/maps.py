from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.arrays import ArrayColumn
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.common.deferred import deferrable

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class MapValue(Value):
    """A map literal or column expression.

    Can be constructed with [`ibis.map()`](#ibis.expr.types.map).

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> import pyarrow as pa
    >>> tab = pa.table(
    ...     {
    ...         "m": pa.array(
    ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
    ...             type=pa.map_(pa.utf8(), pa.int64()),
    ...         )
    ...     }
    ... )
    >>> t = ibis.memtable(tab)
    >>> t
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ m                    ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ map<!string, int64>  │
    ├──────────────────────┤
    │ {'a': 1, 'b': 2}     │
    │ {'a': 1}             │
    │ NULL                 │
    └──────────────────────┘

    Can use `[]` to access values:
    >>> t.m["a"]
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ MapGet(m, 'a', None) ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ int64                │
    ├──────────────────────┤
    │                    1 │
    │                    1 │
    │                 NULL │
    └──────────────────────┘

    To provide default values, use `get`:
    >>> t.m.get("b", 0)
    ┏━━━━━━━━━━━━━━━━━━━┓
    ┃ MapGet(m, 'b', 0) ┃
    ┡━━━━━━━━━━━━━━━━━━━┩
    │ int64             │
    ├───────────────────┤
    │                 2 │
    │                 0 │
    │                 0 │
    └───────────────────┘
    """

    def get(
        self,
        key: ir.Value,
        default: ir.Value | None = None,
    ) -> ir.Value:
        """Return the value for `key` from `expr`.

        Return `default` if `key` is not in the map.

        Parameters
        ----------
        key
            Expression to use for key
        default
            Expression to return if `key` is not a key in `expr`

        Returns
        -------
        Value
            The element type of `self`

        Examples
        --------
        >>> import ibis
        >>> import pyarrow as pa
        >>> ibis.options.interactive = True
        >>> tab = pa.table(
        ...     {
        ...         "m": pa.array(
        ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
        ...             type=pa.map_(pa.utf8(), pa.int64()),
        ...         )
        ...     }
        ... )
        >>> t = ibis.memtable(tab)
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ m                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<!string, int64>  │
        ├──────────────────────┤
        │ {'a': 1, 'b': 2}     │
        │ {'a': 1}             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.m.get("a")
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ MapGet(m, 'a', None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                │
        ├──────────────────────┤
        │                    1 │
        │                    1 │
        │                 NULL │
        └──────────────────────┘
        >>> t.m.get("b")
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ MapGet(m, 'b', None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                │
        ├──────────────────────┤
        │                    2 │
        │                 NULL │
        │                 NULL │
        └──────────────────────┘
        >>> t.m.get("b", 0)
        ┏━━━━━━━━━━━━━━━━━━━┓
        ┃ MapGet(m, 'b', 0) ┃
        ┡━━━━━━━━━━━━━━━━━━━┩
        │ int64             │
        ├───────────────────┤
        │                 2 │
        │                 0 │
        │                 0 │
        └───────────────────┘
        """

        return ops.MapGet(self, key, default).to_expr()

    def length(self) -> ir.IntegerValue:
        """Return the number of key-value pairs in the map.

        Returns
        -------
        IntegerValue
            The number of elements in `self`

        Examples
        --------
        >>> import ibis
        >>> import pyarrow as pa
        >>> ibis.options.interactive = True
        >>> tab = pa.table(
        ...     {
        ...         "m": pa.array(
        ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
        ...             type=pa.map_(pa.utf8(), pa.int64()),
        ...         )
        ...     }
        ... )
        >>> t = ibis.memtable(tab)
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ m                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<!string, int64>  │
        ├──────────────────────┤
        │ {'a': 1, 'b': 2}     │
        │ {'a': 1}             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.m.length()
        ┏━━━━━━━━━━━━━━┓
        ┃ MapLength(m) ┃
        ┡━━━━━━━━━━━━━━┩
        │ int64        │
        ├──────────────┤
        │            2 │
        │            1 │
        │         NULL │
        └──────────────┘
        """
        return ops.MapLength(self).to_expr()

    def __getitem__(self, key: ir.Value) -> ir.Value:
        """Get the value for a given map `key`.

        ::: {.callout-note}
        ## This operation may have different semantics depending on the backend.

        Some backends return `NULL` when a key is missing, others may fail
        the query.
        :::

        Parameters
        ----------
        key
            A map key

        Returns
        -------
        Value
            An element with the value type of the map

        Examples
        --------
        >>> import ibis
        >>> import pyarrow as pa
        >>> ibis.options.interactive = True
        >>> tab = pa.table(
        ...     {
        ...         "m": pa.array(
        ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
        ...             type=pa.map_(pa.utf8(), pa.int64()),
        ...         )
        ...     }
        ... )
        >>> t = ibis.memtable(tab)
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ m                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<!string, int64>  │
        ├──────────────────────┤
        │ {'a': 1, 'b': 2}     │
        │ {'a': 1}             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.m["a"]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ MapGet(m, 'a', None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                │
        ├──────────────────────┤
        │                    1 │
        │                    1 │
        │                 NULL │
        └──────────────────────┘
        """
        return ops.MapGet(self, key).to_expr()

    def contains(
        self, key: int | str | ir.IntegerValue | ir.StringValue
    ) -> ir.BooleanValue:
        """Return whether the map contains `key`.

        Parameters
        ----------
        key
            Mapping key for which to check

        Returns
        -------
        BooleanValue
            Boolean indicating the presence of `key` in the map expression

        Examples
        --------
        >>> import ibis
        >>> import pyarrow as pa
        >>> ibis.options.interactive = True
        >>> tab = pa.table(
        ...     {
        ...         "m": pa.array(
        ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
        ...             type=pa.map_(pa.utf8(), pa.int64()),
        ...         )
        ...     }
        ... )
        >>> t = ibis.memtable(tab)
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ m                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<!string, int64>   │
        ├──────────────────────┤
        │ {'a': 1, 'b': 2}     │
        │ {'a': 1}             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.m.contains("b")
        ┏━━━━━━━━━━━━━━━━━━━━━┓
        ┃ MapContains(m, 'b') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean             │
        ├─────────────────────┤
        │ True                │
        │ False               │
        │ False               │
        └─────────────────────┘
        """
        return ops.MapContains(self, key).to_expr()

    def keys(self) -> ir.ArrayValue:
        """Extract the keys of a map.

        Returns
        -------
        ArrayValue
            The keys of `self`

        Examples
        --------
        >>> import ibis
        >>> import pyarrow as pa
        >>> ibis.options.interactive = True
        >>> tab = pa.table(
        ...     {
        ...         "m": pa.array(
        ...             [[("a", 1), ("b", 2)], [("a", 1)], None],
        ...             type=pa.map_(pa.utf8(), pa.int64()),
        ...         )
        ...     }
        ... )
        >>> t = ibis.memtable(tab)
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ m                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<!string, int64>  │
        ├──────────────────────┤
        │ {'a': 1, 'b': 2}     │
        │ {'a': 1}             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.m.keys()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ MapKeys(m)           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<!string>       │
        ├──────────────────────┤
        │ ['a', 'b']           │
        │ ['a']                │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.MapKeys(self).to_expr()

    def values(self) -> ir.ArrayValue:
        """Extract the values of a map.

        Returns
        -------
        ArrayValue
            The values of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m.values()
        [1, 2]
        """
        return ops.MapValues(self).to_expr()

    def __add__(self, other: MapValue) -> MapValue:
        """Concatenate this map with another.

        Parameters
        ----------
        other
            Map to concatenate with `self`

        Returns
        -------
        MapValue
            `self` concatenated with `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> m1 = ibis.map({"a": 1, "b": 2})
        >>> m2 = ibis.map({"c": 3, "d": 4})
        >>> m1 + m2
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        """
        return ops.MapMerge(self, other).to_expr()

    def __radd__(self, other: MapValue) -> MapValue:
        """Concatenate this map with another.

        Parameters
        ----------
        other
            Map to concatenate with `self`

        Returns
        -------
        MapValue
            `self` concatenated with `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> m1 = ibis.map({"a": 1, "b": 2})
        >>> m2 = ibis.map({"c": 3, "d": 4})
        >>> m1 + m2
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        """
        return ops.MapMerge(self, other).to_expr()


@public
class MapScalar(Scalar, MapValue):
    pass


@public
class MapColumn(Column, MapValue):
    def __getitem__(self, key: ir.Value) -> ir.Column:
        return MapValue.__getitem__(self, key)


@public
@deferrable
def map(
    keys: Iterable[Any] | Mapping[Any, Any] | ArrayColumn,
    values: Iterable[Any] | ArrayColumn | None = None,
) -> MapValue:
    """Create a [map container object](https://docs.python.org/3/glossary.html#term-mapping).

    If the `keys` and `values` are Python literals, then the output will be a
    `MapScalar`. If the `keys` and `values` are expressions (`ArrayColumn`),
    then the the output will be a `MapColumn`.

    Parameters
    ----------
    keys
        Keys of the map or `Mapping`. If `keys` is a `Mapping`, `values` must be `None`.
    values
        Values of the map or `None`. If `None`, the `keys` argument must be a `Mapping`.

    Returns
    -------
    MapValue
        An expression representing either a map column or literal (associative
        array with key/value pairs of fixed types)

    Examples
    --------
    Create a map literal from a dict with the type inferred

    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.map(dict(a=1, b=2))
    {'a': 1, 'b': 2}

    Create a new map column from columns with keys and values

    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t = ibis.memtable({"keys": [["a", "b"], ["b"]], "values": [[1, 2], [3]]})
    >>> t
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ keys                 ┃ values               ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<string>        │ array<int64>         │
    ├──────────────────────┼──────────────────────┤
    │ ['a', 'b']           │ [1, 2]               │
    │ ['b']                │ [3]                  │
    └──────────────────────┴──────────────────────┘
    >>> ibis.map(t.keys, t.values)
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Map(keys, values)    ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ map<string, int64>   │
    ├──────────────────────┤
    │ {'a': 1, 'b': 2}     │
    │ {'b': 3}             │
    └──────────────────────┘
    """
    if values is None:
        keys, values = tuple(keys.keys()), tuple(keys.values())
    return ops.Map(keys, values).to_expr()
