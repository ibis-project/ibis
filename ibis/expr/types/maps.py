from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.generic import Column, Scalar, Value

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class MapValue(Value):
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
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m.get("a")
        MapGet(...)
        >>> m.get("c", 3)
        MapGet(...)
        >>> m.get("d")
        MapGet(...)
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
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m.length()
        MapLength(...)
        """
        return ops.MapLength(self).to_expr()

    def __getitem__(self, key: ir.Value) -> ir.Value:
        """Get the value for a given map `key`.

        !!! info "This operation may have different semantics depending on the backend."

            Some backends return `NULL` when a key is missing, others may fail
            the query.

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
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m["a"]
        MapGet(...)
        >>> m["c"]  # note that this does not fail on construction
        MapGet(...)
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
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m.keys()
        MapKeys(...)
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
        >>> m = ibis.map({"a": 1, "b": 2})
        >>> m.values()
        MapValues(...)
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
        >>> m1 = ibis.map({"a": 1, "b": 2})
        >>> m2 = ibis.map({"c": 3, "d": 4})
        >>> m1 + m2
        MapMerge(...)
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
        >>> m1 = ibis.map({"a": 1, "b": 2})
        >>> m2 = ibis.map({"c": 3, "d": 4})
        >>> m1 + m2
        MapMerge(...)
        """
        return ops.MapMerge(self, other).to_expr()


@public
class MapScalar(Scalar, MapValue):
    pass


@public
class MapColumn(Column, MapValue):
    pass


@public
def map(keys, values=None) -> MapValue:
    """Create a map literal from a [`dict`][dict], other mapping or two sequences.

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
    >>> t = ibis.map(dict(a=1, b=2))
    """
    if values is None:
        keys, values = list(keys.keys()), list(keys.values())
    return ops.Map(keys, values).to_expr()
