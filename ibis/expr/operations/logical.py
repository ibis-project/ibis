from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis.expr.types as ir

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Binary, Unary, Value
from ibis.expr.operations.generic import _Negatable


@public
class LogicalBinary(Binary):
    left = rlz.boolean
    right = rlz.boolean

    output_dtype = dt.boolean


@public
class Not(Unary):
    arg = rlz.boolean

    output_dtype = dt.boolean


@public
class And(LogicalBinary):
    pass


@public
class Or(LogicalBinary):
    pass


@public
class Xor(LogicalBinary):
    pass


@public
class Comparison(Binary):
    left = rlz.any
    right = rlz.any

    output_dtype = dt.boolean

    def __init__(self, left, right):
        """Construct a comparison operation between `left` and `right`.

        Casting rules for type promotions (for resolving the output type) may
        depend on the target backend.

        TODO: how are overflows handled? Can we provide anything useful in
        Ibis to help the user avoid them?
        """
        if not rlz.comparable(left, right):
            raise TypeError(
                f'Arguments {rlz._arg_type_error_format(left)} and '
                f'{rlz._arg_type_error_format(right)} are not comparable'
            )
        super().__init__(left=left, right=right)


@public
class Equals(Comparison):
    pass


@public
class NotEquals(Comparison):
    pass


@public
class GreaterEqual(Comparison):
    pass


@public
class Greater(Comparison):
    pass


@public
class LessEqual(Comparison):
    pass


@public
class Less(Comparison):
    pass


@public
class IdenticalTo(Comparison):
    pass


@public
class Between(Value):
    arg = rlz.any
    lower_bound = rlz.any
    upper_bound = rlz.any

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("args")

    def __init__(self, arg, lower_bound, upper_bound):
        if not rlz.comparable(arg, lower_bound):
            raise TypeError(
                f'Arguments {rlz._arg_type_error_format(arg)} and '
                f'{rlz._arg_type_error_format(lower_bound)} are not comparable'
            )
        if not rlz.comparable(arg, upper_bound):
            raise TypeError(
                f'Arguments {rlz._arg_type_error_format(arg)} and '
                f'{rlz._arg_type_error_format(upper_bound)} are not comparable'
            )
        super().__init__(arg=arg, lower_bound=lower_bound, upper_bound=upper_bound)


# TODO(kszucs): decompose it into at least two operations
@public
class Contains(Value):
    value = rlz.any
    options = rlz.one_of(
        [
            rlz.tuple_of(rlz.any),
            rlz.column(rlz.any),
            rlz.array,
        ]
    )

    output_dtype = dt.boolean

    @attribute.default
    def output_shape(self):
        if isinstance(self.options, tuple):
            args = [self.value, *self.options]
        else:
            args = self.args
        return rlz.highest_precedence_shape(args)


@public
class NotContains(Contains):
    pass


@public
class Where(Value):
    """Ternary case expression, equivalent to.

    bool_expr.case().when(True, true_expr).else_(false_or_null_expr)

    Many backends implement this as a built-in function.
    """

    bool_expr = rlz.boolean
    true_expr = rlz.any
    false_null_expr = rlz.any

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return rlz.highest_precedence_dtype([self.true_expr, self.false_null_expr])


@public
class ExistsSubquery(Value, _Negatable):
    foreign_table = rlz.table
    predicates = rlz.tuple_of(rlz.boolean)

    output_dtype = dt.boolean
    output_shape = rlz.Shape.COLUMNAR

    def negate(self) -> NotExistsSubquery:
        return NotExistsSubquery(*self.args)


@public
class NotExistsSubquery(Value, _Negatable):
    foreign_table = rlz.table
    predicates = rlz.tuple_of(rlz.boolean)

    output_dtype = dt.boolean
    output_shape = rlz.Shape.COLUMNAR

    def negate(self) -> ExistsSubquery:
        return ExistsSubquery(*self.args)


class _UnresolvedSubquery(Value, _Negatable):
    """An exists subquery whose outer leaf table is unknown.

    Notes
    -----
    Consider the following ibis expressions

    >>> t = ibis.table(dict(a="string"))  # doctest: +SKIP
    >>> s = ibis.table(dict(a="string"))  # doctest: +SKIP
    >>> cond = (t.a == s.a).any()  # doctest: +SKIP

    Without knowing the table to use as the outer query there are two ways to
    turn this expression into a SQL `EXISTS` predicate depending on which of
    `t` or `s` is filtered on.

    Filtering from `t`:

    ```sql
    SELECT *
    FROM t
    WHERE EXISTS (SELECT 1 WHERE t.a = s.a)
    ```

    Filtering from `s`:

    ```sql
    SELECT *
    FROM s
    WHERE EXISTS (SELECT 1 WHERE t.a = s.a)
    ```

    Notably the subquery `(SELECT 1 WHERE t.a = s.a)` cannot stand on its own.

    The purpose of `_UnresolvedSubquery` is to capture enough information about
    an exists predicate such that it can be resolved when predicates are
    resolved against the outer leaf table when `Selection`s are constructed.
    """

    tables = rlz.tuple_of(rlz.table)
    predicates = rlz.tuple_of(rlz.boolean)

    output_dtype = dt.boolean
    output_shape = rlz.Shape.COLUMNAR

    @abc.abstractmethod
    def _resolve(
        self, table: ir.Table
    ) -> type[ExistsSubquery] | type[NotExistsSubquery]:  # pragma: no cover
        ...


@public
class UnresolvedExistsSubquery(_UnresolvedSubquery):
    def negate(self) -> UnresolvedNotExistsSubquery:
        return UnresolvedNotExistsSubquery(*self.args)

    def _resolve(self, table) -> ExistsSubquery:
        from ibis.expr.operations.relations import TableNode

        assert isinstance(table, TableNode)

        (foreign_table,) = (t for t in self.tables if t != table)
        return ExistsSubquery(foreign_table, self.predicates).to_expr()


@public
class UnresolvedNotExistsSubquery(_UnresolvedSubquery):
    def negate(self) -> UnresolvedExistsSubquery:
        return UnresolvedExistsSubquery(*self.args)

    def _resolve(self, table: ir.Table) -> NotExistsSubquery:
        from ibis.expr.operations.relations import TableNode

        assert isinstance(table, TableNode)

        (foreign_table,) = (t for t in self.tables if t != table)
        return NotExistsSubquery(foreign_table, self.predicates).to_expr()
