from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

if TYPE_CHECKING:
    import ibis.expr.types as ir
    import ibis.expr.window as win

from public import public

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis.expr import datatypes as dt
from ibis.expr.types.core import Expr, _binop


@public
class Value(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

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
        >>> t = ibis.table(dict(a="int64"))
        >>> t.a.name("b")
        r0 := UnboundTable[unbound_table_...]
          a int64
        b: r0.a
        """
        import ibis.expr.operations as ops

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
    def type(self):
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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

        op = ops.Cast(self, to=target_type)

        if op.to.equals(self.type()):
            # noop case if passed type is the same
            return self

        if isinstance(op.to, (dt.Geography, dt.Geometry)):
            from_geotype = self.type().geotype or 'geometry'
            to_geotype = op.to.geotype
            if from_geotype == to_geotype:
                return self

        result = op.to_expr()
        if not self.has_name():
            return result
        return result.name(f'cast({self.get_name()}, {op.to})')

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
        >>> ibis.coalesce(None, 4, 5)
        Coalesce([ValueList(values=[None, 4, 5])])
        """
        import ibis.expr.operations as ops

        return ops.Coalesce([self, *args]).to_expr()

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
        import ibis.expr.operations as ops

        return ops.Greatest([self, *args]).to_expr()

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
        import ibis.expr.operations as ops

        return ops.Least([self, *args]).to_expr()

    def typeof(self) -> ir.StringValue:
        """Return the data type of the expression.

        The values of the returned strings are necessarily backend dependent.

        Returns
        -------
        StringValue
            A string indicating the type of the value
        """
        import ibis.expr.operations as ops

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
        >>> table = ibis.table(dict(col='int64', other_col='int64'))
        >>> result = table.col.fillna(5)
        r0 := UnboundTable: unbound_table_0
          col       int64
          other_col int64
        IfNull(r0.col, ifnull_expr=5)
        >>> table.col.fillna(table.other_col * 3)
        r0 := UnboundTable: unbound_table_0
          col       int64
          other_col int64
        IfNull(r0.col, ifnull_expr=r0.other_col * 3)

        Returns
        -------
        Value
            `self` filled with `fill_value` where it is `NA`
        """
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return ops.Between(self, rlz.any(lower), rlz.any(upper)).to_expr()

    def isin(
        self,
        values: Value | Sequence[Value],
    ) -> ir.BooleanValue:
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
        >>> table = ibis.table(dict(string_col='string'))
        >>> table.string_col.isin(['foo', 'bar', 'baz'])
        r0 := UnboundTable: unbound_table_1
          string_col string
        Contains(value=r0.string_col, options=[ValueList(values=['foo', 'bar', 'baz'])])

        Check whether a column's values are contained in another table's column

        >>> table2 = ibis.table(dict(other_string_col='string'))
        >>> table.string_col.isin(table2.other_string_col)
        r0 := UnboundTable: unbound_table_3
          other_string_col string
        r1 := UnboundTable: unbound_table_1
          string_col string
        Contains(value=r1.string_col, options=r0.other_string_col)
        """  # noqa: E501
        import ibis.expr.operations as ops

        return ops.Contains(self, values).to_expr()

    def notin(
        self,
        values: Value | Sequence[Value],
    ) -> ir.BooleanValue:
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
        import ibis.expr.operations as ops

        return ops.NotContains(self, values).to_expr()

    def substitute(
        self,
        value: Value,
        replacement: Value | None = None,
        else_: Value | None = None,
    ):
        """Replace one or more values in a value expression.

        Parameters
        ----------
        value
            Expression or mapping
        replacement
            Expression. If an expression is passed to value, this must be
            passed.
        else_
            Expression

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

    def over(self, window: win.Window) -> Value:
        """Construct a window expression.

        Parameters
        ----------
        window
            Window specification

        Returns
        -------
        Value
            A window function expression
        """
        import ibis.expr.operations as ops

        prior_op = self.op()

        # TODO(kszucs): fix this ugly hack
        if isinstance(prior_op, ops.Alias):
            return prior_op.arg.over(window).name(prior_op.name)

        if isinstance(prior_op, ops.Window):
            op = prior_op.over(window)
        else:
            op = ops.Window(self, window)

        result = op.to_expr()

        if self.has_name():
            return result.name(self.get_name())
        return result

    def isnull(self) -> ir.BooleanValue:
        """Return whether this expression is NULL."""
        import ibis.expr.operations as ops

        return ops.IsNull(self).to_expr()

    def notnull(self) -> ir.BooleanValue:
        """Return whether this expression is not NULL."""
        import ibis.expr.operations as ops

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
        r0 := UnboundTable[t]
          string_col string
        SimpleCase(base=r0.string_col, cases=[ValueList(values=['a', 'b'])], results=[ValueList(values=['an a', 'a b'])], default='null or (not a and not b)')
        """  # noqa: E501
        import ibis.expr.builders as bl

        return bl.SimpleCaseBuilder(self)

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

    def collect(self) -> ir.ArrayValue:
        """Return an array of the elements of this expression."""
        import ibis.expr.operations as ops

        return ops.ArrayCollect(self).to_expr()

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
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        try:
            return ops.IdenticalTo(self, rlz.any(other)).to_expr()
        except (com.IbisTypeError, NotImplementedError):
            return NotImplemented

    def group_concat(
        self,
        sep: str = ",",
        where: ir.BooleanValue | None = None,
    ) -> ir.StringScalar:
        """Concatenate values using the indicated separator to produce a
        string.

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
        import ibis.expr.operations as ops

        return ops.GroupConcat(self, sep=sep, where=where).to_expr()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.Equals, self, rlz.any(other))

    def __ne__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.NotEquals, self, rlz.any(other))

    def __ge__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.GreaterEqual, self, rlz.any(other))

    def __gt__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.Greater, self, rlz.any(other))

    def __le__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.LessEqual, self, rlz.any(other))

    def __lt__(self, other: Value) -> ir.BooleanValue:
        import ibis.expr.operations as ops
        import ibis.expr.rules as rlz

        return _binop(ops.Less, self, rlz.any(other))


@public
class Scalar(Value):
    def to_projection(self) -> ir.Table:
        """Promote this scalar expression to a projection."""
        from ibis.expr.types.relations import Table

        roots = self.op().root_tables()
        if len(roots) > 1:
            raise com.RelationError(
                'Cannot convert scalar expression '
                'involving multiple base table references '
                'to a projection'
            )

        table = Table(roots[0])
        return table.projection([self])

    def _repr_html_(self) -> str | None:
        return None


@public
class Column(Value):
    @util.deprecated(version="4.0.0", instead="")
    def parent(self):  # pragma: no cover
        return self._arg

    def to_projection(self) -> ir.Table:
        """Promote this column expression to a projection."""
        from ibis.expr.types.relations import Table

        roots = self.op().root_tables()
        if len(roots) > 1:
            raise com.RelationError(
                'Cannot convert array expression involving multiple base '
                'table references to a projection'
            )

        table = Table(roots[0])
        return table.projection([self])

    def _repr_html_(self) -> str | None:
        if not ibis.options.interactive:
            return None

        return self.execute().to_frame()._repr_html_()

    def bottomk(self, k: int, by: Value | None = None) -> ir.TopK:
        raise NotImplementedError("bottomk is not implemented")

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
        import ibis.expr.operations as ops

        return (
            ops.ApproxCountDistinct(self, where)
            .to_expr()
            .name("approx_nunique")
        )

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
        import ibis.expr.operations as ops

        return ops.ApproxMedian(self, where).to_expr().name("approx_median")

    def max(self, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the maximum of a column."""
        import ibis.expr.operations as ops

        return ops.Max(self, where).to_expr().name("max")

    def min(self, where: ir.BooleanValue | None = None) -> Scalar:
        """Return the minimum of a column."""
        import ibis.expr.operations as ops

        return ops.Min(self, where).to_expr().name("min")

    def argmax(
        self, key: ir.Value, where: ir.BooleanValue | None = None
    ) -> Scalar:
        """Return the value of `self` that maximizes `key`."""
        import ibis.expr.operations as ops

        return ops.ArgMax(self, key=key, where=where).to_expr()

    def argmin(
        self, key: ir.Value, where: ir.BooleanValue | None = None
    ) -> Scalar:
        """Return the value of `self` that minimizes `key`."""
        import ibis.expr.operations as ops

        return ops.ArgMin(self, key=key, where=where).to_expr()

    def nunique(
        self, where: ir.BooleanValue | None = None
    ) -> ir.IntegerScalar:
        import ibis.expr.operations as ops

        return ops.CountDistinct(self, where).to_expr().name("nunique")

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
            An expression. Defaults to the count

        Returns
        -------
        TopK
            A top-k expression
        """
        import ibis.expr.operations as ops

        op = ops.TopK(self, k, by=by if by is not None else self.count())
        return op.to_expr()

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
            unique_metric = self.nunique().name('uniques')
        else:
            unique_metric = self.approx_nunique().name('uniques')

        metrics = [
            self.count(),
            self.isnull().sum().name('nulls'),
            unique_metric,
        ]
        metrics = [m.name(f"{prefix}{m.get_name()}{suffix}") for m in metrics]

        return metrics

    def arbitrary(
        self,
        where: ir.BooleanValue | None = None,
        how: Literal["first", "last", "heavy"] | None = None,
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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

        return ops.Count(self, where).to_expr().name("count")

    def value_counts(self, metric_name: str = "count") -> ir.Table:
        """Compute a frequency table.

        Returns
        -------
        Table
            Frequency table expression
        """
        from ibis.expr.analysis import find_first_base_table

        base = find_first_base_table(self)
        metric = base.count().name(metric_name)

        if not self.has_name():
            expr = self.name("unnamed")
        else:
            expr = self

        return base.group_by(expr).aggregate(metric)

    def first(self) -> Column:
        import ibis.expr.operations as ops

        return ops.FirstValue(self).to_expr()

    def last(self) -> Column:
        import ibis.expr.operations as ops

        return ops.LastValue(self).to_expr()

    def rank(self) -> Column:
        import ibis.expr.operations as ops

        return ops.MinRank(self).to_expr()

    def dense_rank(self) -> Column:
        import ibis.expr.operations as ops

        return ops.DenseRank(self).to_expr()

    def percent_rank(self) -> Column:
        import ibis.expr.operations as ops

        return ops.PercentRank(self).to_expr()

    def cume_dist(self) -> Column:
        import ibis.expr.operations as ops

        return ops.CumeDist(self).to_expr()

    def cummin(self) -> Column:
        import ibis.expr.operations as ops

        return ops.CumulativeMin(self).to_expr()

    def cummax(self) -> Column:
        import ibis.expr.operations as ops

        return ops.CumulativeMax(self).to_expr()

    def lag(
        self,
        offset: int | ir.IntegerValue | None = None,
        default: Value | None = None,
    ) -> Column:
        import ibis.expr.operations as ops

        return ops.Lag(self, offset, default).to_expr()

    def lead(
        self,
        offset: int | ir.IntegerValue | None = None,
        default: Value | None = None,
    ) -> Column:
        import ibis.expr.operations as ops

        return ops.Lead(self, offset, default).to_expr()

    def ntile(self, buckets: int | ir.IntegerValue) -> ir.IntegerColumn:
        import ibis.expr.operations as ops

        return ops.NTile(self, buckets).to_expr()

    def nth(self, n: int | ir.IntegerValue) -> Column:
        """Return the `n`th value over a window.

        Parameters
        ----------
        n
            Desired rank value

        Returns
        -------
        Column
            The nth value over a window
        """
        import ibis.expr.operations as ops

        return ops.NthValue(self, n).to_expr()


@public
class NullValue(Value):
    pass  # noqa: E701,E302


@public
class NullScalar(Scalar, NullValue):
    pass  # noqa: E701,E302


@public
class NullColumn(Column, NullValue):
    pass  # noqa: E701,E302


@public
class ValueList(Value, Sequence[Value]):
    @property
    def values(self):
        return self.op().values

    def __getitem__(self, key):
        return self.values[key]

    def __bool__(self):
        return bool(self.values)

    __nonzero__ = __bool__

    def __len__(self):
        return len(self.values)


_NULL = None


@public
def null():
    """Create a NULL/NA scalar"""
    import ibis.expr.operations as ops

    global _NULL
    if _NULL is None:
        _NULL = ops.NullLiteral().to_expr()

    return _NULL


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
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops

    if hasattr(value, 'op') and isinstance(value.op(), ops.Literal):
        return value

    try:
        inferred_dtype = dt.infer(value)
    except com.InputTypeError:
        has_inferred = False
    else:
        has_inferred = True

    if type is None:
        has_explicit = False
    else:
        has_explicit = True
        explicit_dtype = dt.dtype(type)

    if has_explicit and has_inferred:
        try:
            # ensure type correctness: check that the inferred dtype is
            # implicitly castable to the explicitly given dtype and value
            dtype = inferred_dtype.cast(explicit_dtype, value=value)
        except com.IbisTypeError:
            raise TypeError(
                f'Value {value!r} cannot be safely coerced to {type}'
            )
    elif has_explicit:
        dtype = explicit_dtype
    elif has_inferred:
        dtype = inferred_dtype
    else:
        raise TypeError(
            'The datatype of value {!r} cannot be inferred, try '
            'passing it explicitly with the `type` keyword.'.format(value)
        )

    if dtype is dt.null:
        return null().cast(dtype)
    else:
        value = dt._normalize(dtype, value)
        return ops.Literal(value, dtype=dtype).to_expr()


public(
    ValueExpr=Value,
    ScalarExpr=Scalar,
    ColumnExpr=Column,
    AnyValue=Value,
    AnyScalar=Scalar,
    AnyColumn=Column,
    ListExpr=ValueList,
)
