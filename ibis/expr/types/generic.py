from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.common.grounds import Singleton
from ibis.expr.types.core import Expr, _binop, _FixedTextJupyterMixin

if TYPE_CHECKING:
    import pandas as pd

    import ibis.expr.types as ir


@public
class Value(Expr):
    """Base class for a data generating expression having a known type."""

    def name(self, name):
        """Rename an expression to `name`.

        Parameters
        ----------
        name
            The new name of the expression

        Returns
        -------
        Value
            `self` with name `name`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table(dict(a="int64"), name="t")
        >>> t.a.name("b")
        r0 := UnboundTable: t
          a int64
        b: r0.a
        """
        # TODO(kszucs): shouldn't do simplification here, but rather later
        # when simplifying the whole operation tree
        # the expression's name is idendical to the new one
        if self.has_name() and self.get_name() == name:
            return self

        if isinstance(self.op(), ops.Alias):
            # only keep a single alias operation
            op = ops.Alias(arg=self.op().arg, name=name)
        else:
            op = ops.Alias(arg=self, name=name)

        return op.to_expr()

    # TODO(kszucs): should rename to dtype
    def type(self) -> dt.DataType:
        """Return the [DataType] of this expression."""
        return self.op().output_dtype

    def hash(self, how: str = "fnv") -> ir.IntegerValue:
        """Compute an integer hash value.

        Parameters
        ----------
        how
            Hash algorithm to use

        Returns
        -------
        IntegerValue
            The hash value of `self`
        """
        return ops.Hash(self, how).to_expr()

    def cast(self, target_type: dt.DataType) -> Value:
        """Cast expression to indicated data type.

        Parameters
        ----------
        target_type
            Type to cast to

        Returns
        -------
        Value
            Casted expression
        """
        op = ops.Cast(self, to=target_type)

        if op.to == self.type():
            # noop case if passed type is the same
            return self

        if op.to.is_geospatial():
            from_geotype = self.type().geotype or 'geometry'
            to_geotype = op.to.geotype
            if from_geotype == to_geotype:
                return self

        return op.to_expr()

    def coalesce(self, *args: Value) -> Value:
        """Return the first non-null value from `args`.

        Parameters
        ----------
        args
            Arguments from which to choose the first non-null value

        Returns
        -------
        Value
            Coalesced expression

        Examples
        --------
        >>> import ibis
        >>> ibis.coalesce(None, 4, 5).name("x")
        x: Coalesce(...)
        """
        return ops.Coalesce((self, *args)).to_expr()

    def greatest(self, *args: ir.Value) -> ir.Value:
        """Compute the largest value among the supplied arguments.

        Parameters
        ----------
        args
            Arguments to choose from

        Returns
        -------
        Value
            Maximum of the passed arguments
        """
        return ops.Greatest((self, *args)).to_expr()

    def least(self, *args: ir.Value) -> ir.Value:
        """Compute the smallest value among the supplied arguments.

        Parameters
        ----------
        args
            Arguments to choose from

        Returns
        -------
        Value
            Minimum of the passed arguments
        """
        return ops.Least((self, *args)).to_expr()

    def typeof(self) -> ir.StringValue:
        """Return the data type of the expression.

        The values of the returned strings are necessarily backend dependent.

        Returns
        -------
        StringValue
            A string indicating the type of the value
        """
        return ops.TypeOf(self).to_expr()

    def fillna(self, fill_value: Scalar) -> Value:
        """Replace any null values with the indicated fill value.

        Parameters
        ----------
        fill_value
            Value with which to replace `NA` values in `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.sex
        ┏━━━━━━━━┓
        ┃ sex    ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ male   │
        │ female │
        │ female │
        │ NULL   │
        │ female │
        │ male   │
        │ female │
        │ male   │
        │ NULL   │
        │ NULL   │
        │ …      │
        └────────┘
        >>> t.sex.fillna("unrecorded").name("sex")
        ┏━━━━━━━━━━━━┓
        ┃ sex        ┃
        ┡━━━━━━━━━━━━┩
        │ string     │
        ├────────────┤
        │ male       │
        │ female     │
        │ female     │
        │ unrecorded │
        │ female     │
        │ male       │
        │ female     │
        │ male       │
        │ unrecorded │
        │ unrecorded │
        │ …          │
        └────────────┘

        Returns
        -------
        Value
            `self` filled with `fill_value` where it is `NA`
        """
        return ops.IfNull(self, fill_value).to_expr()

    def nullif(self, null_if_expr: Value) -> Value:
        """Set values to null if they equal the values `null_if_expr`.

        Commonly use to avoid divide-by-zero problems by replacing zero with
        `NULL` in the divisor.

        Parameters
        ----------
        null_if_expr
            Expression indicating what values should be NULL

        Returns
        -------
        Value
            Value expression
        """
        return ops.NullIf(self, null_if_expr).to_expr()

    def between(
        self,
        lower: Value,
        upper: Value,
    ) -> ir.BooleanValue:
        """Check if this expression is between `lower` and `upper`, inclusive.

        Parameters
        ----------
        lower
            Lower bound
        upper
            Upper bound

        Returns
        -------
        BooleanValue
            Expression indicating membership in the provided range
        """
        return ops.Between(self, lower, upper).to_expr()

    def isin(self, values: Value | Sequence[Value]) -> ir.BooleanValue:
        """Check whether this expression's values are in `values`.

        Parameters
        ----------
        values
            Values or expression to check for membership

        Returns
        -------
        BooleanValue
            Expression indicating membership

        Examples
        --------
        Check whether a column's values are contained in a sequence

        >>> import ibis
        >>> table = ibis.table(dict(string_col='string'), name="t")
        >>> table.string_col.isin(['foo', 'bar', 'baz'])
        r0 := UnboundTable: t
          string_col string
        Contains(string_col): Contains(...)

        Check whether a column's values are contained in another table's column

        >>> table2 = ibis.table(dict(other_string_col='string'), name="t2")
        >>> table.string_col.isin(table2.other_string_col)
        r0 := UnboundTable: t
          string_col string
        r1 := UnboundTable: t2
          other_string_col string
        Contains(string_col, other_string_col): Contains(...)
        """
        return ops.Contains(self, values).to_expr()

    def notin(self, values: Value | Sequence[Value]) -> ir.BooleanValue:
        """Check whether this expression's values are not in `values`.

        Parameters
        ----------
        values
            Values or expression to check for lack of membership

        Returns
        -------
        BooleanValue
            Whether `self`'s values are not contained in `values`
        """
        return ops.NotContains(self, values).to_expr()

    def substitute(
        self,
        value: Value | dict,
        replacement: Value | None = None,
        else_: Value | None = None,
    ):
        """Replace values given in `values` with `replacement`.

        This is similar to the pandas `replace` method.

        Parameters
        ----------
        value
            Expression or dict.
        replacement
            If an expression is passed to value, this must be
            passed.
        else_
            If an original value does not match `value`, then `else_` is used.
            The default of `None` means leave the original value unchanged.

        Returns
        -------
        Value
            Replaced values
        """
        expr = self.case()
        if isinstance(value, dict):
            for k, v in sorted(value.items()):
                expr = expr.when(k, v)
        else:
            expr = expr.when(value, replacement)

        return expr.else_(else_ if else_ is not None else self).end()

    def over(
        self,
        window=None,
        *,
        rows=None,
        range=None,
        group_by=None,
        order_by=None,
    ) -> Value:
        """Construct a window expression.

        Parameters
        ----------
        window
            Window specification
        rows
            Whether to use the `ROWS` window clause
        range
            Whether to use the `RANGE` window clause
        group_by
            Grouping key
        order_by
            Ordering key

        Returns
        -------
        Value
            A window function expression
        """
        import ibis.expr.analysis as an
        import ibis.expr.builders as bl
        import ibis.expr.deferred as de

        if window is None:
            window = ibis.window(
                rows=rows,
                range=range,
                group_by=group_by,
                order_by=order_by,
            )

        def bind(table):
            frame = window.bind(table)
            return ops.WindowFunction(self, frame).to_expr()

        op = self.op()
        if isinstance(op, ops.Alias):
            return op.arg.to_expr().over(window).name(op.name)
        elif isinstance(op, ops.WindowFunction):
            return op.func.to_expr().over(window)
        elif isinstance(window, bl.WindowBuilder):
            if table := an.find_first_base_table(self.op()):
                return bind(table)
            else:
                return de.Deferred(bind)
        else:
            return ops.WindowFunction(self, window).to_expr()

    def isnull(self) -> ir.BooleanValue:
        """Return whether this expression is NULL."""
        return ops.IsNull(self).to_expr()

    def notnull(self) -> ir.BooleanValue:
        """Return whether this expression is not NULL."""
        return ops.NotNull(self).to_expr()

    def case(self):
        """Create a SimpleCaseBuilder to chain multiple if-else statements.

        Add new search expressions with the `.when()` method. These must be
        comparable with this column expression. Conclude by calling `.end()`

        Returns
        -------
        SimpleCaseBuilder
            A case builder

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([('string_col', 'string')], name='t')
        >>> expr = t.string_col
        >>> case_expr = (expr.case()
        ...              .when('a', 'an a')
        ...              .when('b', 'a b')
        ...              .else_('null or (not a and not b)')
        ...              .end())
        >>> case_expr
        r0 := UnboundTable: t
          string_col string
        SimpleCase(...)
        """
        import ibis.expr.builders as bl

        return bl.SimpleCaseBuilder(self.op())

    def cases(
        self,
        case_result_pairs: Iterable[tuple[ir.BooleanValue, Value]],
        default: Value | None = None,
    ) -> Value:
        """Create a case expression in one shot.

        Parameters
        ----------
        case_result_pairs
            Conditional-result pairs
        default
            Value to return if none of the case conditions are true

        Returns
        -------
        Value
            Value expression
        """
        builder = self.case()
        for case, result in case_result_pairs:
            builder = builder.when(case, result)
        return builder.else_(default).end()

    def collect(self, where: ir.BooleanValue | None = None) -> ir.ArrayScalar:
        """Aggregate this expression's elements into an array.

        This function is called `array_agg`, `list_agg`, or `list` in other systems.

        Parameters
        ----------
        where
            Filter to apply before aggregation

        Returns
        -------
        ArrayScalar
            Collected array

        Examples
        --------
        Basic collect usage

        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"key": list("aaabb"), "value": [1, 2, 3, 4, 5]})
        >>> t
        ┏━━━━━━━━┳━━━━━━━┓
        ┃ key    ┃ value ┃
        ┡━━━━━━━━╇━━━━━━━┩
        │ string │ int64 │
        ├────────┼───────┤
        │ a      │     1 │
        │ a      │     2 │
        │ a      │     3 │
        │ b      │     4 │
        │ b      │     5 │
        └────────┴───────┘
        >>> t.value.collect()
        [1, 2, 3, 4, 5]
        >>> type(t.value.collect())
        <class 'ibis.expr.types.arrays.ArrayScalar'>

        Collect elements per group

        >>> t.group_by("key").agg(v=lambda t: t.value.collect())
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ key    ┃ v                    ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ string │ array<int64>         │
        ├────────┼──────────────────────┤
        │ a      │ [1, 2, ... +1]       │
        │ b      │ [4, 5]               │
        └────────┴──────────────────────┘

        Collect elements per group using a filter

        >>> t.group_by("key").agg(v=lambda t: t.value.collect(where=t.value > 1))
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ key    ┃ v                    ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ string │ array<int64>         │
        ├────────┼──────────────────────┤
        │ a      │ [2, 3]               │
        │ b      │ [4, 5]               │
        └────────┴──────────────────────┘
        """
        return ops.ArrayCollect(self, where=where).to_expr()

    def identical_to(self, other: Value) -> ir.BooleanValue:
        """Return whether this expression is identical to other.

        Corresponds to `IS NOT DISTINCT FROM` in SQL.

        Parameters
        ----------
        other
            Expression to compare to

        Returns
        -------
        BooleanValue
            Whether this expression is not distinct from `other`
        """
        try:
            return ops.IdenticalTo(self, other).to_expr()
        except (com.IbisTypeError, NotImplementedError):
            return NotImplemented

    def group_concat(
        self,
        sep: str = ",",
        where: ir.BooleanValue | None = None,
    ) -> ir.StringScalar:
        """Concatenate values using the indicated separator to produce a string.

        Parameters
        ----------
        sep
            Separator will be used to join strings
        where
            Filter expression

        Returns
        -------
        StringScalar
            Concatenated string expression
        """
        return ops.GroupConcat(self, sep=sep, where=where).to_expr()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.Equals, self, other)

    def __ne__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.NotEquals, self, other)

    def __ge__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.GreaterEqual, self, other)

    def __gt__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.Greater, self, other)

    def __le__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.LessEqual, self, other)

    def __lt__(self, other: Value) -> ir.BooleanValue:
        return _binop(ops.Less, self, other)

    def asc(self) -> ir.Value:
        """Sort an expression ascending."""
        return ops.SortKey(self, ascending=True).to_expr()

    def desc(self) -> ir.Value:
        """Sort an expression descending."""
        return ops.SortKey(self, ascending=False).to_expr()

    def as_table(self) -> ir.Table:
        """Promote the expression to a table.

        Returns
        -------
        Table
            A table expression

        Examples
        --------
        >>> t = ibis.table(dict(a="str"), name="t")
        >>> expr = t.a.length().name("len").as_table()
        >>> expected = t.select(len=t.a.length())
        >>> expr.equals(expected)
        True
        """
        from ibis.expr.analysis import find_immediate_parent_tables

        roots = find_immediate_parent_tables(self.op())
        if len(roots) > 1:
            raise com.RelationError(
                f'Cannot convert {type(self)} expression '
                'involving multiple base table references '
                'to a projection'
            )
        table = roots[0].to_expr()
        return table.select(self)

    def to_pandas(self, **kwargs) -> pd.Series:
        """Convert a column expression to a pandas Series or scalar object.

        Parameters
        ----------
        kwargs
            Same as keyword arguments to [`execute`][ibis.expr.types.core.Expr.execute]
        """
        return self.execute(**kwargs)


@public
class Scalar(Value):
    def __rich_console__(self, console, options):
        from rich.text import Text

        if not ibis.options.interactive:
            return console.render(Text(self._repr()), options=options)
        return console.render(repr(self.execute()), options=options)

    def as_table(self) -> ir.Table:
        """Promote the scalar expression to a table.

        Returns
        -------
        Table
            A table expression

        Examples
        --------
        Promote an aggregation to a table

        >>> import ibis
        >>> import ibis.expr.types as ir
        >>> t = ibis.table(dict(a="str"), name="t")
        >>> expr = t.a.length().sum().name("len").as_table()
        >>> isinstance(expr, ir.Table)
        True

        Promote a literal value to a table

        >>> import ibis.expr.types as ir
        >>> lit = ibis.literal(1).name("a").as_table()
        >>> isinstance(lit, ir.Table)
        True
        """
        from ibis.expr.analysis import (
            find_first_base_table,
            is_scalar_reduction,
            reduction_to_aggregation,
        )

        op = self.op()
        if is_scalar_reduction(op):
            return reduction_to_aggregation(op)

        table = find_first_base_table(op)
        if table is not None:
            agg = ops.Aggregation(table=table, metrics=(op,))
        else:
            agg = ops.DummyTable(values=(op,))
        return agg.to_expr()

    def _repr_html_(self) -> str | None:
        return None


@public
class Column(Value, _FixedTextJupyterMixin):
    # Higher than numpy & dask objects
    __array_priority__ = 20

    __array_ufunc__ = None

    def __array__(self, dtype=None):
        return self.execute().__array__(dtype)

    def __rich_console__(self, console, options):
        named = self.name(self.op().name)
        projection = named.as_table()
        return console.render(projection, options=options)

    def approx_nunique(
        self,
        where: ir.BooleanValue | None = None,
    ) -> ir.IntegerScalar:
        """Return the approximate number of distinct elements in `self`.

        !!! info "The result may or may not be exact"

            Whether the result is an approximation depends on the backend.

            !!! warning "Do not depend on the results being exact"

        Parameters
        ----------
        where
            Filter in values when `where` is `True`

        Returns
        -------
        Scalar
            An approximate count of the distinct elements of `self`
        """
        return ops.ApproxCountDistinct(self, where).to_expr()

    def approx_median(
        self,
        where: ir.BooleanValue | None = None,
    ) -> Scalar:
        """Return an approximate of the median of `self`.

        !!! info "The result may or may not be exact"

            Whether the result is an approximation depends on the backend.

            !!! warning "Do not depend on the results being exact"

        Parameters
        ----------
        where
            Filter in values when `where` is `True`

        Returns
        -------
        Scalar
            An approximation of the median of `self`
        """
        return ops.ApproxMedian(self, where).to_expr()

    def mode(self, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the mode of a column."""
        return ops.Mode(self, where).to_expr()

    def max(self, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the maximum of a column."""
        return ops.Max(self, where).to_expr()

    def min(self, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the minimum of a column."""
        return ops.Min(self, where).to_expr()

    def argmax(self, key: ir.Value, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the value of `self` that maximizes `key`."""
        return ops.ArgMax(self, key=key, where=where).to_expr()

    def argmin(self, key: ir.Value, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the value of `self` that minimizes `key`."""
        return ops.ArgMin(self, key=key, where=where).to_expr()

    def nunique(self, where: ir.BooleanValue | None = None) -> ir.IntegerScalar:
        """Compute the number of distinct rows in an expression.

        Parameters
        ----------
        where
            Filter expression

        Returns
        -------
        IntegerScalar
            Number of distinct elements in an expression
        """
        return ops.CountDistinct(self, where).to_expr()

    def topk(
        self,
        k: int,
        by: ir.Value | None = None,
    ) -> ir.TopK:
        """Return a "top k" expression.

        Parameters
        ----------
        k
            Return this number of rows
        by
            An expression. Defaults to `count`.

        Returns
        -------
        TableExpr
            A top-k expression
        """

        from ibis.expr.analysis import find_first_base_table

        arg_table = find_first_base_table(self.op()).to_expr()

        if by is None:
            by = self.count().name("count")

        if callable(by):
            by = by(arg_table)
            by_table = arg_table
        elif isinstance(by, Value):
            by_table = find_first_base_table(by.op()).to_expr()
        else:
            raise com.IbisTypeError(f"Invalid `by` argument with type {type(by)}")

        assert by.op().name != self.op().name

        if not arg_table.equals(by_table):
            raise com.IbisError('Cross-table TopK; must provide a parent joined table')

        return (
            arg_table.aggregate(by, by=[self])
            .order_by(ibis.desc(by.get_name()))
            .limit(k)
        )

    @util.deprecated(
        instead="Reach out at https://github.com/ibis-project/ibis if you'd like this API to remain.",
        as_of="5.0",
        removed_in="6.0",
    )
    def summary(
        self,
        exact_nunique: bool = False,
        prefix: str = "",
        suffix: str = "",
    ) -> list[ir.NumericScalar]:
        """Compute a set of summary metrics.

        Parameters
        ----------
        exact_nunique
            Compute the exact number of distinct values. Typically slower if
            `True`.
        prefix
            String prefix for metric names
        suffix
            String suffix for metric names

        Returns
        -------
        list[NumericScalar]
            Metrics list
        """
        if exact_nunique:
            unique_metric = self.nunique()
        else:
            unique_metric = self.approx_nunique()
        unique_metric = unique_metric.name("uniques")

        metrics = [
            self.count().name("count"),
            self.isnull().sum().name('nulls'),
            unique_metric,
        ]
        metrics = [m.name(f"{prefix}{m.get_name()}{suffix}") for m in metrics]

        return metrics

    def arbitrary(
        self,
        where: ir.BooleanValue | None = None,
        how: Literal["first", "last", "heavy"] = "first",
    ) -> Scalar:
        """Select an arbitrary value in a column.

        Parameters
        ----------
        where
            A filter expression
        how
            The method to use for selecting the element.

            * `"first"`: Select the first non-`NULL` element
            * `"last"`: Select the last non-`NULL` element
            * `"heavy"`: Select a frequently occurring value using the heavy
              hitters algorithm. `"heavy"` is only supported by Clickhouse
              backend.

        Returns
        -------
        Scalar
            An expression
        """
        return ops.Arbitrary(self, how=how, where=where).to_expr()

    def count(self, where: ir.BooleanValue | None = None) -> ir.IntegerScalar:
        """Compute the number of rows in an expression.

        Parameters
        ----------
        where
            Filter expression

        Returns
        -------
        IntegerScalar
            Number of elements in an expression
        """
        return ops.Count(self, where).to_expr()

    def value_counts(self) -> ir.Table:
        """Compute a frequency table.

        Returns
        -------
        Table
            Frequency table expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"chars": char} for char in "aabcddd")
        >>> t
        ┏━━━━━━━━┓
        ┃ chars  ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ a      │
        │ a      │
        │ b      │
        │ c      │
        │ d      │
        │ d      │
        │ d      │
        └────────┘
        >>> t.chars.value_counts()
        ┏━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ chars  ┃ chars_count ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━┩
        │ string │ int64       │
        ├────────┼─────────────┤
        │ a      │           2 │
        │ b      │           1 │
        │ c      │           1 │
        │ d      │           3 │
        └────────┴─────────────┘
        """
        from ibis.expr.analysis import find_first_base_table

        name = self.get_name()
        return (
            find_first_base_table(self.op())
            .to_expr()
            .select(self)
            .group_by(name)
            .agg(**{f"{name}_count": lambda t: t.count()})
        )

    def first(self) -> Column:
        """Return the first value of a column.

        Equivalent to SQL's `FIRST_VALUE` window function.
        """
        return ops.FirstValue(self).to_expr()

    def last(self) -> Column:
        """Return the last value of a column.

        Equivalent to SQL's `LAST_VALUE` window function.
        """
        return ops.LastValue(self).to_expr()

    def rank(self) -> ir.IntegerColumn:
        """Compute position of first element within each equal-value group in sorted order.

        Equivalent to SQL's `RANK()` window function.

        Examples
        --------
        values   ranks
        1        0
        1        0
        2        2
        2        2
        2        2
        3        5

        Returns
        -------
        Int64Column
            The min rank
        """
        return ops.MinRank(self).to_expr()

    def dense_rank(self) -> ir.IntegerColumn:
        """Position of first element within each group of equal values.

        Values are returned in sorted order and duplicate values are ignored.

        Equivalent to SQL's `DENSE_RANK()`.

        Examples
        --------
        values   ranks
        1        0
        1        0
        2        1
        2        1
        2        1
        3        2

        Returns
        -------
        IntegerColumn
            The rank
        """
        return ops.DenseRank(self).to_expr()

    def percent_rank(self) -> Column:
        """Return the relative rank of the values in the column."""
        return ops.PercentRank(self).to_expr()

    def cume_dist(self) -> Column:
        """Return the cumulative distribution over a window."""
        return ops.CumeDist(self).to_expr()

    def cummin(self) -> Column:
        """Return the cumulative min over a window."""
        return ops.CumulativeMin(self).to_expr()

    def cummax(self) -> Column:
        """Return the cumulative max over a window."""
        return ops.CumulativeMax(self).to_expr()

    def lag(
        self,
        offset: int | ir.IntegerValue | None = None,
        default: Value | None = None,
    ) -> Column:
        """Return the row located at `offset` rows **before** the current row.

        Parameters
        ----------
        offset
            Index of row to select
        default
            Value used if no row exists at `offset`
        """
        return ops.Lag(self, offset, default).to_expr()

    def lead(
        self,
        offset: int | ir.IntegerValue | None = None,
        default: Value | None = None,
    ) -> Column:
        """Return the row located at `offset` rows **after** the current row.

        Parameters
        ----------
        offset
            Index of row to select
        default
            Value used if no row exists at `offset`
        """
        return ops.Lead(self, offset, default).to_expr()

    def ntile(self, buckets: int | ir.IntegerValue) -> ir.IntegerColumn:
        """Return the integer number of a partitioning of the column values.

        Parameters
        ----------
        buckets
            Number of buckets to partition into
        """
        return ops.NTile(self, buckets).to_expr()

    def nth(self, n: int | ir.IntegerValue) -> Column:
        """Return the `n`th value (0-indexed) over a window.

        `.nth(0)` is equivalent to `.first()`. Negative will result in `NULL`.
        If the value of `n` is greater than the number of rows in the window,
        `NULL` will be returned.

        Parameters
        ----------
        n
            Desired rank value

        Returns
        -------
        Column
            The nth value over a window
        """
        return ops.NthValue(self, n).to_expr()


@public
class UnknownValue(Value):
    pass


@public
class UnknownScalar(Scalar):
    pass


@public
class UnknownColumn(Column):
    pass


@public
class NullValue(Value):
    pass


@public
class NullScalar(Scalar, NullValue, Singleton):
    pass


@public
class NullColumn(Column, NullValue):
    pass


@public
def null():
    """Create a NULL/NA scalar."""
    return ops.NullLiteral().to_expr()


@public
def literal(value: Any, type: dt.DataType | str | None = None) -> Scalar:
    """Create a scalar expression from a Python value.

    !!! tip "Use specific functions for arrays, structs and maps"

        Ibis supports literal construction of arrays using the following
        functions:

        1. [`ibis.array`][ibis.array]
        1. [`ibis.struct`][ibis.struct]
        1. [`ibis.map`][ibis.map]

        Constructing these types using `literal` will be deprecated in a future
        release.

    Parameters
    ----------
    value
        A Python value
    type
        An instance of [`DataType`][ibis.expr.datatypes.DataType] or a string
        indicating the ibis type of `value`. This parameter can be used
        in cases where ibis's type inference isn't sufficient for discovering
        the type of `value`.

    Returns
    -------
    Scalar
        An expression representing a literal value

    Examples
    --------
    Construct an integer literal

    >>> import ibis
    >>> x = ibis.literal(42)
    >>> x.type()
    Int8(nullable=True)

    Construct a `float64` literal from an `int`

    >>> y = ibis.literal(42, type='double')
    >>> y.type()
    Float64(nullable=True)

    Ibis checks for invalid types

    >>> ibis.literal('foobar', type='int64')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Value 'foobar' cannot be safely coerced to int64
    """
    import ibis.expr.rules as rlz

    if isinstance(value, Expr):
        value = value.op()

    return rlz.literal(type, value).to_expr()


public(
    ValueExpr=Value,
    ScalarExpr=Scalar,
    ColumnExpr=Column,
    AnyValue=Value,
    AnyScalar=Scalar,
    AnyColumn=Column,
)
