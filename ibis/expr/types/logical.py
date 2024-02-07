from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis
import ibis.expr.operations as ops
from ibis.expr.types.core import _binop
from ibis.expr.types.numeric import NumericColumn, NumericScalar, NumericValue

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class BooleanValue(NumericValue):
    def ifelse(self, true_expr: ir.Value, false_expr: ir.Value) -> ir.Value:
        """Construct a ternary conditional expression.

        Parameters
        ----------
        true_expr
            Expression to return if `self` evaluates to `True`
        false_expr
            Expression to return if `self` evaluates to `False` or `NULL`

        Returns
        -------
        Value
            The value of `true_expr` if `arg` is `True` else `false_expr`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"is_person": [True, False, True, None]})
        >>> t.is_person.ifelse("yes", "no")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ IfElse(is_person, 'yes', 'no') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                         │
        ├────────────────────────────────┤
        │ yes                            │
        │ no                             │
        │ yes                            │
        │ no                             │
        └────────────────────────────────┘
        """
        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        return ops.IfElse(self, true_expr, false_expr).to_expr()

    def __and__(self, other: BooleanValue) -> BooleanValue:
        """Construct a binary AND conditional expression with `self` and `other`.

        Parameters
        ----------
        self
            Left operand
        other
            Right operand

        Returns
        -------
        BooleanValue
            A Boolean expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1], [], [42, 42], None]})
        >>> t.arr.contains(42) & (t.arr.contains(1))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ And(ArrayContains(arr, 42), ArrayContains(arr, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                            │
        ├────────────────────────────────────────────────────┤
        │ False                                              │
        │ False                                              │
        │ False                                              │
        │ NULL                                               │
        └────────────────────────────────────────────────────┘

        >>> t.arr.contains(42) & (t.arr.contains(42))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ And(ArrayContains(arr, 42), ArrayContains(arr, 42)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                             │
        ├─────────────────────────────────────────────────────┤
        │ False                                               │
        │ False                                               │
        │ True                                                │
        │ NULL                                                │
        └─────────────────────────────────────────────────────┘
        """
        return _binop(ops.And, self, other)

    __rand__ = __and__

    def __or__(self, other: BooleanValue) -> BooleanValue:
        """Construct a binary OR conditional expression with `self` and `other`.

        Parameters
        ----------
        self
            Left operand
        other
            Right operand

        Returns
        -------
        BooleanValue
            A Boolean expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, None]})
        >>> (t.arr > 1) | (t.arr > 2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Or(Greater(arr, 1), Greater(arr, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                              │
        ├──────────────────────────────────────┤
        │ False                                │
        │ True                                 │
        │ True                                 │
        │ NULL                                 │
        └──────────────────────────────────────┘
        """
        return _binop(ops.Or, self, other)

    __ror__ = __or__

    def __xor__(self, other: BooleanValue) -> BooleanValue:
        """Construct a binary XOR conditional expression with `self` and `other`.

        Parameters
        ----------
        self
            Left operand
        other
            Right operand

        Returns
        -------
        BooleanValue
            A Boolean expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, None]})
        >>> t.arr == 2
        ┏━━━━━━━━━━━━━━━━┓
        ┃ Equals(arr, 2) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ boolean        │
        ├────────────────┤
        │ False          │
        │ True           │
        │ False          │
        │ NULL           │
        └────────────────┘

        >>> (t.arr > 2)
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ Greater(arr, 2) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ boolean         │
        ├─────────────────┤
        │ False           │
        │ False           │
        │ True            │
        │ NULL            │
        └─────────────────┘

        >>> (t.arr == 2) ^ (t.arr > 2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Xor(Equals(arr, 2), Greater(arr, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                              │
        ├──────────────────────────────────────┤
        │ False                                │
        │ True                                 │
        │ True                                 │
        │ NULL                                 │
        └──────────────────────────────────────┘
        """

        return _binop(ops.Xor, self, other)

    __rxor__ = __xor__

    def __invert__(self) -> BooleanValue:
        """Construct a unary NOT conditional expression with `self`.

        Parameters
        ----------
        self
            Operand

        Returns
        -------
        BooleanValue
            A Boolean expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [True, False, False, None]})
        >>> ~t.arr
        ┏━━━━━━━━━━┓
        ┃ Not(arr) ┃
        ┡━━━━━━━━━━┩
        │ boolean  │
        ├──────────┤
        │ False    │
        │ True     │
        │ True     │
        │ NULL     │
        └──────────┘
        """
        return self.negate()

    @staticmethod
    def __negate_op__():
        return ops.Not


@public
class BooleanScalar(NumericScalar, BooleanValue):
    pass


@public
class BooleanColumn(NumericColumn, BooleanValue):
    def any(self, where: BooleanValue | None = None) -> BooleanValue:
        """Return whether at least one element is `True`.

        If the expression does not reference any foreign tables, the result
        will be a scalar reduction, otherwise it will be a deferred expression
        constructing an exists subquery when passed to a table method.

        Parameters
        ----------
        where
            Optional filter for the aggregation

        Returns
        -------
        BooleanValue
            Whether at least one element is `True`.

        Notes
        -----
        Consider the following ibis expressions

        ```python
        import ibis

        t = ibis.table(dict(a="string"))
        s = ibis.table(dict(a="string"))

        cond = (t.a == s.a).any()
        ```

        Without knowing the table to use as the outer query there are two ways to
        turn this expression into a SQL `EXISTS` predicate, depending on which of
        `t` or `s` is filtered on.

        Filtering from `t`:

        ```sql
        SELECT *
        FROM t
        WHERE EXISTS (SELECT 1 FROM s WHERE t.a = s.a)
        ```

        Filtering from `s`:

        ```sql
        SELECT *
        FROM s
        WHERE EXISTS (SELECT 1 FROM t WHERE t.a = s.a)
        ```

        Notably the correlated subquery cannot stand on its own.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, None]})
        >>> (t.arr > 2).any()
        True
        >>> (t.arr > 4).any()
        False
        >>> m = ibis.memtable({"arr": [True, True, True, False]})
        >>> (t.arr == None).any(where=t.arr != None)
        False
        """
        from ibis.common.deferred import Call, Deferred, _

        parents = self.op().relations

        def resolve_exists_subquery(outer):
            """An exists subquery whose outer leaf table is unknown."""
            (inner,) = (t for t in parents if t != outer.op())
            relation = ops.Filter(inner, [self])
            return ops.ExistsSubquery(relation).to_expr()

        if len(parents) == 2:
            return Deferred(Call(resolve_exists_subquery, _))
        elif len(parents) == 1:
            op = ops.Any(self, where=self._bind_reduction_filter(where))
        else:
            raise NotImplementedError(
                f'Cannot compute "any" for expression of type {type(self)} '
                f"with multiple foreign tables"
            )

        return op.to_expr()

    def notany(self, where: BooleanValue | None = None) -> BooleanValue:
        """Return whether no elements are `True`.

        Parameters
        ----------
        where
            Optional filter for the aggregation

        Returns
        -------
        BooleanValue
            Whether no elements are `True`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, 4]})
        >>> (t.arr > 1).notany()
        False
        >>> (t.arr > 4).notany()
        True
        >>> m = ibis.memtable({"arr": [True, True, True, False]})
        >>> (t.arr == None).notany(where=t.arr != None)
        True
        """
        return ~self.any(where=where)

    def all(self, where: BooleanValue | None = None) -> BooleanScalar:
        """Return whether all elements are `True`.

        Parameters
        ----------
        where
            Optional filter for the aggregation

        Returns
        -------
        BooleanValue
            Whether all elements are `True`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, 4]})
        >>> (t.arr >= 1).all()
        True
        >>> (t.arr > 2).all()
        False
        >>> (t.arr == 2).all(where=t.arr == 2)
        True
        >>> (t.arr == 2).all(where=t.arr >= 2)
        False

        """
        return ops.All(self, where=self._bind_reduction_filter(where)).to_expr()

    def notall(self, where: BooleanValue | None = None) -> BooleanScalar:
        """Return whether not all elements are `True`.

        Parameters
        ----------
        where
            Optional filter for the aggregation

        Returns
        -------
        BooleanValue
            Whether not all elements are `True`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, 4]})
        >>> (t.arr >= 1).notall()
        False
        >>> (t.arr > 2).notall()
        True
        >>> (t.arr == 2).notall(where=t.arr == 2)
        False
        >>> (t.arr == 2).notall(where=t.arr >= 2)
        True
        """
        return ~self.all(where=where)

    def cumany(self, *, where=None, group_by=None, order_by=None) -> BooleanColumn:
        """Accumulate the `any` aggregate.

        Returns
        -------
        BooleanColumn
            A boolean column with the cumulative `any` aggregate.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, 4]})
        >>> ((t.arr > 1) | (t.arr >= 1)).cumany()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Any(Or(Greater(arr, 1), GreaterEqual(arr, 1))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                        │
        ├────────────────────────────────────────────────┤
        │ True                                           │
        │ True                                           │
        │ True                                           │
        │ True                                           │
        └────────────────────────────────────────────────┘
        >>> ((t.arr > 1) & (t.arr >= 1)).cumany()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Any(And(Greater(arr, 1), GreaterEqual(arr, 1))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                         │
        ├─────────────────────────────────────────────────┤
        │ False                                           │
        │ True                                            │
        │ True                                            │
        │ True                                            │
        └─────────────────────────────────────────────────┘
        """
        return self.any(where=where).over(
            ibis.cumulative_window(group_by=group_by, order_by=order_by)
        )

    def cumall(self, *, where=None, group_by=None, order_by=None) -> BooleanColumn:
        """Accumulate the `all` aggregate.

        Returns
        -------
        BooleanColumn
            A boolean column with the cumulative `all` aggregate.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [1, 2, 3, 4]})
        >>> ((t.arr > 1) & (t.arr >= 1)).cumall()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ All(And(Greater(arr, 1), GreaterEqual(arr, 1))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                         │
        ├─────────────────────────────────────────────────┤
        │ False                                           │
        │ False                                           │
        │ False                                           │
        │ False                                           │
        └─────────────────────────────────────────────────┘
        >>> ((t.arr > 0) & (t.arr >= 1)).cumall()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ All(And(Greater(arr, 0), GreaterEqual(arr, 1))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                         │
        ├─────────────────────────────────────────────────┤
        │ True                                            │
        │ True                                            │
        │ True                                            │
        │ True                                            │
        └─────────────────────────────────────────────────┘
        """
        return self.all(where=where).over(
            ibis.cumulative_window(group_by=group_by, order_by=order_by)
        )
