from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable
import inspect
from public import public

import ibis.expr.operations as ops
from ibis.expr.types.generic import Column, Scalar, Value, literal
from ibis.expr.types.typing import V
from ibis.common.deferred import deferrable

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
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.length()
        ┏━━━━━━━━━━━━━━━━┓
        ┃ ArrayLength(a) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ int64          │
        ├────────────────┤
        │              2 │
        │              1 │
        │           NULL │
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
            - If `index` is an [](`int`) or
              [`IntegerValue`](./expression-numeric.qmd#ibis.expr.types.IntegerValue)
              then the return type is the element type of `self`.
            - If `index` is a [](`slice`) then the return type is the same
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
        │ int64            │
        ├──────────────────┤
        │                7 │
        │                3 │
        │             NULL │
        └──────────────────┘

        Extract a range of elements

        >>> t = ibis.memtable({"a": [[7, 42, 72], [3] * 5, None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42, ... +1]      │
        │ [3, 3, ... +3]       │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a[1:2]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArraySlice(a, 1, 2)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [42]                 │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        """
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if step is not None and step != 1:
                raise NotImplementedError("step can only be 1")

            op = ops.ArraySlice(self, start if start is not None else 0, stop)
        else:
            op = ops.ArrayIndex(self, index)
        return op.to_expr()

    def concat(self, other: ArrayValue, *args: ArrayValue) -> ArrayValue:
        """Concatenate this array with one or more arrays.

        Parameters
        ----------
        other
            Other array to concat with `self`
        args
            Other arrays to concat with `self`

        Returns
        -------
        ArrayValue
            `self` concatenated with `other` and `args`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.concat(t.a)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat()        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.concat(ibis.literal([4], type="array<int64>"))
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat()        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 4]               │
        │ [3, 4]               │
        │ [4]                  │
        └──────────────────────┘

        `concat` is also available using the `+` operator

        >>> [1] + t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat()        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, 7]               │
        │ [1, 3]               │
        │ [1]                  │
        └──────────────────────┘
        >>> t.a + [1]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat()        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 1]               │
        │ [3, 1]               │
        │ [1]                  │
        └──────────────────────┘
        """
        return ops.ArrayConcat((self, other, *args)).to_expr()

    def __add__(self, other: ArrayValue) -> ArrayValue:
        return self.concat(other)

    def __radd__(self, other: ArrayValue) -> ArrayValue:
        return ops.ArrayConcat((other, self)).to_expr()

    def repeat(self, n: int | ir.IntegerValue) -> ArrayValue:
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
        >>> t = ibis.memtable({"a": [[7], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.repeat(2)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘

        `repeat` is also available using the `*` operator

        >>> 2 * t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘
        """
        return ops.ArrayRepeat(self, n).to_expr()

    __mul__ = __rmul__ = repeat

    def unnest(self) -> ir.Value:
        """Flatten an array into a column.

        ::: {.callout-note}
        ## This operation changes the cardinality of the result
        :::

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3, 3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3, 3]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.unnest()
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     7 │
        │    42 │
        │     3 │
        │     3 │
        └───────┘

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
        │ NULL                 │
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
        │ NULL                      │
        │ NULL                      │
        │ b                         │
        └───────────────────────────┘

        See Also
        --------
        [`StringValue.join`](./expression-strings.qmd#ibis.expr.types.strings.StringValue.join)
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
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘
        >>> t.a.map(lambda x: (x + 100).cast("float"))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Cast(Add(x, 100), float64)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<float64>                          │
        ├─────────────────────────────────────────┤
        │ [101.0, None, ... +1]                   │
        │ [104.0]                                 │
        │ []                                      │
        └─────────────────────────────────────────┘

        `.map()` also supports more complex callables like `functools.partial`
        and lambdas with closures

        >>> from functools import partial
        >>> def add(x, y):
        ...     return x + y
        ...
        >>> add2 = partial(add, y=2)
        >>> t.a.map(add2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Add(x, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [3, None, ... +1]      │
        │ [6]                    │
        │ []                     │
        └────────────────────────┘
        >>> y = 2
        >>> t.a.map(lambda x: x + y)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Add(x, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [3, None, ... +1]      │
        │ [6]                    │
        │ []                     │
        └────────────────────────┘
        """
        param = next(iter(inspect.signature(func).parameters.keys()))
        parameter = ops.Argument(
            name=param, shape=self.op().shape, dtype=self.type().value_type
        ).to_expr()
        return ops.ArrayMap(self, param=param, body=func(parameter)).to_expr()

    def filter(
        self, predicate: Callable[[ir.Value], bool | ir.BooleanValue]
    ) -> ir.ArrayValue:
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
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘
        >>> t.a.filter(lambda x: x > 1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘

        `.filter()` also supports more complex callables like `functools.partial`
        and lambdas with closures

        >>> from functools import partial
        >>> def gt(x, y):
        ...     return x > y
        ...
        >>> gt1 = partial(gt, y=1)
        >>> t.a.filter(gt1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘
        >>> y = 1
        >>> t.a.filter(lambda x: x > y)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘
        """
        param = next(iter(inspect.signature(predicate).parameters.keys()))
        parameter = ops.Argument(
            name=param, shape=self.op().shape, dtype=self.type().value_type
        ).to_expr()
        return ops.ArrayFilter(self, param=param, body=predicate(parameter)).to_expr()

    def contains(self, other: ir.Value) -> ir.BooleanValue:
        """Return whether the array contains `other`.

        Parameters
        ----------
        other
            Ibis expression to check for existence of in `self`

        Returns
        -------
        BooleanValue
            Whether `other` is contained in `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1]                  │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.contains(42)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(arr, 42) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                │
        ├────────────────────────┤
        │ False                  │
        │ False                  │
        │ True                   │
        │ NULL                   │
        └────────────────────────┘
        >>> t.arr.contains(None)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(arr, None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                  │
        ├──────────────────────────┤
        │ NULL                     │
        │ NULL                     │
        │ NULL                     │
        │ NULL                     │
        └──────────────────────────┘
        """
        return ops.ArrayContains(self, other).to_expr()

    def index(self, other: ir.Value) -> ir.IntegerValue:
        """Return the position of `other` in an array.

        Parameters
        ----------
        other
            Ibis expression to existence of in `self`

        Returns
        -------
        BooleanValue
            The position of `other` in `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1]                  │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.index(42)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, 42) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                  │
        ├────────────────────────┤
        │                     -1 │
        │                     -1 │
        │                      0 │
        │                   NULL │
        └────────────────────────┘
        >>> t.arr.index(800)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, 800) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                   │
        ├─────────────────────────┤
        │                      -1 │
        │                      -1 │
        │                      -1 │
        │                    NULL │
        └─────────────────────────┘
        >>> t.arr.index(None)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                    │
        ├──────────────────────────┤
        │                     NULL │
        │                     NULL │
        │                     NULL │
        │                     NULL │
        └──────────────────────────┘
        """
        return ops.ArrayPosition(self, other).to_expr()

    def remove(self, other: ir.Value) -> ir.ArrayValue:
        """Remove `other` from `self`.

        Parameters
        ----------
        other
            Element to remove from `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[3, 2], [], [42, 2], [2, 2], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 2]               │
        │ []                   │
        │ [42, 2]              │
        │ [2, 2]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.remove(2)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRemove(arr, 2)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3]                  │
        │ []                   │
        │ [42]                 │
        │ []                   │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArrayRemove(self, other).to_expr()

    def unique(self) -> ir.ArrayValue:
        """Return the unique values in an array.

        ::: {.callout-note}
        ## Element ordering in array may not be retained.
        :::

        Returns
        -------
        ArrayValue
            Unique values in an array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 3, 3], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, 3, ... +1]       │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.unique()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayDistinct(arr)   ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 1]               │
        │ []                   │
        │ [42]                 │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArrayDistinct(self).to_expr()

    def sort(self) -> ir.ArrayValue:
        """Sort the elements in an array.

        Returns
        -------
        ArrayValue
            Sorted values in an array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[3, 2], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 2]               │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.sort()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArraySort(arr)       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [2, 3]               │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArraySort(self).to_expr()

    def union(self, other: ir.ArrayValue) -> ir.ArrayValue:
        """Union two arrays.

        Parameters
        ----------
        other
            Another array to union with `self`

        Returns
        -------
        ArrayValue
            Unioned arrays

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"arr1": [[3, 2], [], None], "arr2": [[1, 3], [None], [5]]}
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr1                 ┃ arr2                 ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<int64>         │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ [1, 3]               │
        │ []                   │ [None]               │
        │ NULL                 │ [5]                  │
        └──────────────────────┴──────────────────────┘
        >>> t.arr1.union(t.arr2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayUnion(arr1, arr2) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [1, 2, ... +1]         │
        │ [None]                 │
        │ [5]                    │
        └────────────────────────┘
        >>> t.arr1.union(t.arr2).contains(3)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(ArrayUnion(arr1, arr2), 3) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                  │
        ├──────────────────────────────────────────┤
        │ True                                     │
        │ False                                    │
        │ False                                    │
        └──────────────────────────────────────────┘
        """
        return ops.ArrayUnion(self, other).to_expr()

    def intersect(self, other: ArrayValue) -> ArrayValue:
        """Intersect two arrays.

        Parameters
        ----------
        other
            Another array to intersect with `self`

        Returns
        -------
        ArrayValue
            Intersected arrays

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"arr1": [[3, 2], [], None], "arr2": [[1, 3], [None], [5]]}
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr1                 ┃ arr2                 ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<int64>         │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ [1, 3]               │
        │ []                   │ [None]               │
        │ NULL                 │ [5]                  │
        └──────────────────────┴──────────────────────┘
        >>> t.arr1.intersect(t.arr2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayIntersect(arr1, arr2) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>               │
        ├────────────────────────────┤
        │ [3]                        │
        │ []                         │
        │ NULL                       │
        └────────────────────────────┘
        """
        return ops.ArrayIntersect(self, other).to_expr()

    def zip(self, other: ArrayValue, *others: ArrayValue) -> ArrayValue:
        """Zip two or more arrays together.

        Parameters
        ----------
        other
            Another array to zip with `self`
        others
            Additional arrays to zip with `self`

        Returns
        -------
        Array
            Array of structs where each struct field is an element of each input
            array.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"numbers": [[3, 2], [], None], "strings": [["a", "c"], None, ["e"]]}
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ numbers              ┃ strings              ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<string>        │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ ['a', 'c']           │
        │ []                   │ NULL                 │
        │ NULL                 │ ['e']                │
        └──────────────────────┴──────────────────────┘
        >>> expr = t.numbers.zip(t.strings)
        >>> expr
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayZip()                           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<struct<f1: int64, f2: string>> │
        ├──────────────────────────────────────┤
        │ [{...}, {...}]                       │
        │ []                                   │
        │ [{...}]                              │
        └──────────────────────────────────────┘
        >>> expr.unnest()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayZip()                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ struct<f1: int64, f2: string> │
        ├───────────────────────────────┤
        │ {'f1': 3, 'f2': 'a'}          │
        │ {'f1': 2, 'f2': 'c'}          │
        │ {'f1': None, 'f2': 'e'}       │
        └───────────────────────────────┘
        """

        return ops.ArrayZip((self, other, *others)).to_expr()


@public
class ArrayScalar(Scalar, ArrayValue):
    pass


@public
class ArrayColumn(Column, ArrayValue):
    def __getitem__(self, index: int | ir.IntegerValue | slice) -> ir.Column:
        return ArrayValue.__getitem__(self, index)


@public
@deferrable
def array(values: Iterable[V], type: str | dt.DataType | None = None) -> ArrayValue:
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
    >>> ibis.options.interactive = True
    >>> t = ibis.memtable({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> ibis.array([t.a, t.b])
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ ArrayColumn()        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<int64>         │
    ├──────────────────────┤
    │ [1, 4]               │
    │ [2, 5]               │
    │ [3, 6]               │
    └──────────────────────┘

    Create an array scalar from Python literals

    >>> ibis.array([1.0, 2.0, 3.0])
    [1.0, 2.0, 3.0]

    Mixing scalar and column expressions is allowed

    >>> ibis.array([t.a, 42])
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ ArrayColumn()        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<int64>         │
    ├──────────────────────┤
    │ [1, 42]              │
    │ [2, 42]              │
    │ [3, 42]              │
    └──────────────────────┘
    """
    if any(isinstance(value, Value) for value in values):
        return ops.ArrayColumn(values).to_expr()
    else:
        try:
            return literal(list(values), type=type)
        except com.IbisTypeError as e:
            raise com.IbisTypeError(
                "Could not create an array scalar from the values provided "
                "to `array`. Ensure that all input values have the same "
                "Python type, or can be casted to a single Python type."
            ) from e
