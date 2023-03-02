from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from public import public

import ibis.expr.operations as ops
from ibis.expr.types.generic import Column, Scalar, Value, literal
from ibis.expr.types.typing import V

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir

import ibis.common.exceptions as com


@public
class ArrayValue(Value):
    def length(self) -> ir.IntegerValue:
        """Compute the length of an array.

        Returns
        -------
        IntegerValue
            The integer length of each element of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a.length()
        ┏━━━━━━━━━━━━━━━━┓
        ┃ ArrayLength(a) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ int64          │
        ├────────────────┤
        │              2 │
        │              1 │
        │              ∅ │
        └────────────────┘
        """
        return ops.ArrayLength(self).to_expr()

    def __getitem__(self, index: int | ir.IntegerValue | slice) -> ir.Value:
        """Extract one or more elements of `self`.

        Parameters
        ----------
        index
            Index into `array`

        Returns
        -------
        Value
            - If `index` is an [`int`][int] or
              [`IntegerValue`][ibis.expr.types.IntegerValue] then the return
              type is the element type of `self`.
            - If `index` is a [`slice`][slice] then the return type is the same
              type as the input.

        Examples
        --------
        Extract a single element

        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3], None]})
        >>> t.a[0]
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayIndex(a, 0) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ int8             │
        ├──────────────────┤
        │                7 │
        │                3 │
        │                ∅ │
        └──────────────────┘

        Extract a range of elements

        >>> t = ibis.memtable({"a": [[7, 42, 72], [3] * 5, None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 42, ... +1]      │
        │ [3, 3, ... +3]       │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a[1:2]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArraySlice(a, 1, 2)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [42]                 │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        """
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if step is not None and step != 1:
                raise NotImplementedError('step can only be 1')

            op = ops.ArraySlice(self, start if start is not None else 0, stop)
        else:
            op = ops.ArrayIndex(self, index)
        return op.to_expr()

    def __add__(self, other: ArrayValue) -> ArrayValue:
        """Concatenate this array with another.

        Parameters
        ----------
        other
            Array to concat with `self`

        Returns
        -------
        ArrayValue
            `self` concatenated with `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3] , None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a + t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat(a, a)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a + [4]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat(a, (4,)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 4]               │
        │ [3, 4]               │
        │ [4]                  │
        └──────────────────────┘
        """
        return ops.ArrayConcat(self, other).to_expr()

    def __radd__(self, other: ArrayValue) -> ArrayValue:
        """Concatenate this array with another.

        Parameters
        ----------
        other
            Array to concat with `self`

        Returns
        -------
        ArrayValue
            `self` concatenated with `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3] , None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        >>> [4] + t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat((4,), a) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [4, 7]               │
        │ [4, 3]               │
        │ [4]                  │
        └──────────────────────┘
        """
        return ops.ArrayConcat(other, self).to_expr()

    def __mul__(self, n: int | ir.IntegerValue) -> ArrayValue:
        """Repeat this array `n` times.

        Parameters
        ----------
        n
            Number of times to repeat `self`.

        Returns
        -------
        ArrayValue
            `self` repeated `n` times

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3] , None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a * 2
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘
        """
        return ops.ArrayRepeat(self, n).to_expr()

    def __rmul__(self, n: int | ir.IntegerValue) -> ArrayValue:
        """Repeat this array `n` times.

        Parameters
        ----------
        n
            Number of times to repeat `self`.

        Returns
        -------
        ArrayValue
            `self` repeated `n` times

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3] , None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ ∅                    │
        └──────────────────────┘
        >>> 2 * t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘
        """
        return ops.ArrayRepeat(self, n).to_expr()

    def unnest(self) -> ir.Value:
        """Flatten an array into a column.

        !!! note "This operation changes the cardinality of the result"

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3, 3] , None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3, 3]               │
        │ ∅                    │
        └──────────────────────┘
        >>> t.a.unnest()
        ┏━━━━━━┓
        ┃ a    ┃
        ┡━━━━━━┩
        │ int8 │
        ├──────┤
        │    7 │
        │   42 │
        │    3 │
        │    3 │
        └──────┘

        Returns
        -------
        ir.Value
            Unnested array
        """
        expr = ops.Unnest(self).to_expr()
        try:
            return expr.name(self.get_name())
        except com.ExpressionError:
            return expr

    def join(self, sep: str | ir.StringValue) -> ir.StringValue:
        """Join the elements of this array expression with `sep`.

        Parameters
        ----------
        sep
            Separator to use for joining array elements

        Returns
        -------
        StringValue
            Elements of `self` joined with `sep`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [["a", "b", "c"], None, [], ["b", None]]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>        │
        ├──────────────────────┤
        │ ['a', 'b', ... +1]   │
        │ ∅                    │
        │ []                   │
        │ ['b', None]          │
        └──────────────────────┘
        >>> t.arr.join("|")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayStringJoin('|', arr) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                    │
        ├───────────────────────────┤
        │ a|b|c                     │
        │ ∅                         │
        │ ∅                         │
        │ b                         │
        └───────────────────────────┘

        See Also
        --------
        [`StringValue.join`][ibis.expr.types.strings.StringValue.join]
        """
        return ops.ArrayStringJoin(sep, self).to_expr()

    def map(self, func: Callable[[ir.Value], ir.Value]) -> ir.ArrayValue:
        """Apply a callable `func` to each element of this array expression.

        Parameters
        ----------
        func
            Function to apply to each element of this array

        Returns
        -------
        ArrayValue
            `func` applied to every element of this array expression.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[1, None, 2], [4], []]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘
        >>> t.a.map(lambda x: (x + 100).cast("float"))
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a)           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<float64>        │
        ├───────────────────────┤
        │ [101.0, None, ... +1] │
        │ [104.0]               │
        │ []                    │
        └───────────────────────┘
        """
        return ops.ArrayMap(self, func=func).to_expr()

    def filter(self, predicate: Callable[[ir.Value], ir.BooleanValue]) -> ir.ArrayValue:
        """Filter array elements using `predicate`.

        Parameters
        ----------
        predicate
            Function to use to filter array elements

        Returns
        -------
        ArrayValue
            Array elements filtered using `predicate`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[1, None, 2], [4], []]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘
        >>> t.a.filter(lambda x: x > 1)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a)       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int8>          │
        ├──────────────────────┤
        │ [2]                  │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘
        """
        return ops.ArrayFilter(self, func=predicate).to_expr()


@public
class ArrayScalar(Scalar, ArrayValue):
    pass


@public
class ArrayColumn(Column, ArrayValue):
    pass


@public
def array(
    values: Iterable[V],
    type: str | dt.DataType | None = None,
) -> ArrayValue:
    """Create an array expression.

    If the input expressions are all column expressions, then the output will
    be an `ArrayColumn`. The input columns will be concatenated row-wise to
    produce each array in the output array column. Each array will have length
    _n_, where _n_ is the number of input columns. All input columns should be
    of the same datatype.

    If the input expressions are Python literals, then the output will be a
    single `ArrayScalar` of length _n_, where _n_ is the number of input
    values. This is equivalent to

    ```python
    values = [1, 2, 3]
    ibis.literal(values)
    ```

    Parameters
    ----------
    values
        An iterable of Ibis expressions or a list of Python literals
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`.

    Returns
    -------
    ArrayValue
        An array column (if the inputs are column expressions), or an array
        scalar (if the inputs are Python literals)

    Examples
    --------
    Create an array column from column expressions
    >>> import ibis
    >>> t = ibis.table([('a', 'int64'), ('b', 'int64')], name='t')
    >>> result = ibis.array([t.a, t.b])

    Create an array scalar from Python literals
    >>> import ibis
    >>> result = ibis.array([1.0, 2.0, 3.0])
    """
    if all(isinstance(value, Column) for value in values):
        return ops.ArrayColumn(values).to_expr()
    elif any(isinstance(value, Column) for value in values):
        raise com.IbisTypeError(
            'To create an array column using `array`, all input values must '
            'be column expressions.'
        )
    else:
        try:
            return literal(list(values), type=type)
        except com.IbisTypeError as e:
            raise com.IbisTypeError(
                'Could not create an array scalar from the values provided '
                'to `array`. Ensure that all input values have the same '
                'Python type, or can be casted to a single Python type.'
            ) from e
