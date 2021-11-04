import collections
import itertools
from typing import List

from cached_property import cached_property
from public import public

from ... import util
from ...common import exceptions as com
from .. import rules as rlz
from .. import schema as sch
from .. import types as ir
from ..signature import Argument as Arg
from .core import Node, all_equal, distinct_roots
from .sortkeys import _maybe_convert_sort_keys

_table_names = (f'unbound_table_{i:d}' for i in itertools.count())


@public
def genname():
    return next(_table_names)


@public
class TableNode(Node):
    def get_type(self, name):
        return self.schema[name]

    def output_type(self):
        return ir.TableExpr

    def aggregate(self, this, metrics, by=None, having=None):
        return Aggregation(this, metrics, by=by, having=having)

    def sort_by(self, expr, sort_exprs):
        return Selection(
            expr,
            [],
            sort_keys=_maybe_convert_sort_keys(
                [self.to_expr(), expr],
                sort_exprs,
            ),
        )

    def is_ancestor(self, other):
        import ibis.expr.lineage as lin

        if isinstance(other, ir.Expr):
            other = other.op()

        if self.equals(other):
            return True

        fn = lambda e: (lin.proceed, e.op())  # noqa: E731
        expr = self.to_expr()
        for child in lin.traverse(fn, expr):
            if child.equals(other):
                return True
        return False


@public
class PhysicalTable(TableNode, sch.HasSchema):
    def blocks(self):
        return True


@public
class UnboundTable(PhysicalTable):
    schema = Arg(sch.Schema)
    name = Arg(str, default=genname)


@public
class DatabaseTable(PhysicalTable):
    name = Arg(str)
    schema = Arg(sch.Schema)
    source = Arg(rlz.client)

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


@public
class SQLQueryResult(TableNode, sch.HasSchema):
    """A table sourced from the result set of a select query"""

    query = Arg(str)
    schema = Arg(sch.Schema)
    source = Arg(rlz.client)

    def blocks(self):
        return True


def _make_distinct_join_predicates(left, right, predicates):
    # see GH #667

    # If left and right table have a common parent expression (e.g. they
    # have different filters), must add a self-reference and make the
    # appropriate substitution in the join predicates

    if left.equals(right):
        right = right.view()

    predicates = _clean_join_predicates(left, right, predicates)
    return left, right, predicates


def _clean_join_predicates(left, right, predicates):
    import ibis.expr.analysis as L

    result = []

    if not isinstance(predicates, (list, tuple)):
        predicates = [predicates]

    for pred in predicates:
        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise com.ExpressionError('Join key tuple must be ' 'length 2')
            lk, rk = pred
            lk = left._ensure_expr(lk)
            rk = right._ensure_expr(rk)
            pred = lk == rk
        elif isinstance(pred, str):
            pred = left[pred] == right[pred]
        elif not isinstance(pred, ir.Expr):
            raise NotImplementedError

        if not isinstance(pred, ir.BooleanColumn):
            raise com.ExpressionError('Join predicate must be comparison')

        preds = L.flatten_predicate(pred)
        result.extend(preds)

    _validate_join_predicates(left, right, result)
    return result


def _validate_join_predicates(left, right, predicates):
    from ibis.expr.analysis import fully_originate_from

    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    for predicate in predicates:
        if not fully_originate_from(predicate, [left, right]):
            raise com.RelationError(
                'The expression {!r} does not fully '
                'originate from dependencies of the table '
                'expression.'.format(predicate)
            )


@public
class Join(TableNode):
    left = Arg(rlz.table)
    right = Arg(rlz.table)
    predicates = Arg(rlz.list_of(rlz.boolean), default=[])

    def __init__(self, left, right, predicates):
        left, right, predicates = _make_distinct_join_predicates(
            left, right, predicates
        )
        super().__init__(left, right, predicates)

    def _get_schema(self):
        # For joins retaining both table schemas, merge them together here
        left = self.left
        right = self.right

        if not left._is_materialized():
            left = left.materialize()

        if not right._is_materialized():
            right = right.materialize()

        sleft = left.schema()
        sright = right.schema()

        overlap = set(sleft.names) & set(sright.names)
        if overlap:
            raise com.RelationError(
                'Joined tables have overlapping names: %s' % str(list(overlap))
            )

        return sleft.append(sright)

    def has_schema(self):
        return False

    def root_tables(self):
        if util.all_of([self.left.op(), self.right.op()], (Join, Selection)):
            # Unraveling is not possible
            return [self.left.op(), self.right.op()]
        else:
            return distinct_roots(self.left, self.right)


@public
class InnerJoin(Join):
    pass


@public
class LeftJoin(Join):
    pass


@public
class RightJoin(Join):
    pass


@public
class OuterJoin(Join):
    pass


@public
class AnyInnerJoin(Join):
    pass


@public
class AnyLeftJoin(Join):
    pass


@public
class LeftSemiJoin(Join):
    def _get_schema(self):
        return self.left.schema()


@public
class LeftAntiJoin(Join):
    def _get_schema(self):
        return self.left.schema()


@public
class CrossJoin(Join):
    pass


@public
class MaterializedJoin(TableNode, sch.HasSchema):
    join = Arg(rlz.table)

    def _validate(self):
        assert isinstance(self.join.op(), Join)
        # check whether the underlying schema has overlapping columns or not
        assert self.schema

    @cached_property
    def schema(self):
        return self.join.op()._get_schema()

    def root_tables(self):
        return self.join.op().root_tables()

    def blocks(self):
        return True


@public
class AsOfJoin(Join):
    left = Arg(rlz.table)
    right = Arg(rlz.table)
    predicates = Arg(rlz.list_of(rlz.boolean))
    by = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.function_of("table"),
                    rlz.column_from("table"),
                    rlz.any,
                )
            )
        ),
        default=[],
    )
    tolerance = Arg(rlz.interval, default=None)

    def __init__(self, left, right, predicates, by, tolerance):
        super().__init__(left, right, predicates)
        self.by = _clean_join_predicates(self.left, self.right, by)
        self.tolerance = tolerance

        self._validate_args(['by', 'tolerance'])

    def _validate_args(self, args: List[str]):
        # this should be removed altogether
        for arg in args:
            argument = self.__signature__.parameters[arg]
            value = argument.validate(self, getattr(self, arg))
            setattr(self, arg, value)


@public
class SetOp(TableNode, sch.HasSchema):
    left = Arg(rlz.table)
    right = Arg(rlz.table)

    def _validate(self):
        if not self.left.schema().equals(self.right.schema()):
            raise com.RelationError(
                'Table schemas must be equal for set operations'
            )

    @cached_property
    def schema(self):
        return self.left.schema()

    def blocks(self):
        return True


@public
class Union(SetOp):
    distinct = Arg(rlz.instance_of(bool), default=False)


@public
class Intersection(SetOp):
    pass


@public
class Difference(SetOp):
    pass


@public
class Limit(TableNode):
    table = Arg(rlz.table)
    n = Arg(rlz.instance_of(int))
    offset = Arg(rlz.instance_of(int))

    def blocks(self):
        return True

    @property
    def schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        return [self]


@public
class SelfReference(TableNode, sch.HasSchema):
    table = Arg(rlz.table)

    @cached_property
    def schema(self):
        return self.table.schema()

    def root_tables(self):
        # The dependencies of this operation are not walked, which makes the
        # table expression holding this relationally distinct from other
        # expressions, so things like self-joins are possible
        return [self]

    def blocks(self):
        return True


@public
class Selection(TableNode, sch.HasSchema):
    table = Arg(rlz.table)
    selections = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.table,
                    rlz.column_from("table"),
                    rlz.function_of("table"),
                    rlz.any,
                    rlz.named_literal,
                )
            )
        ),
        default=[],
    )
    predicates = Arg(rlz.list_of(rlz.boolean), default=[])
    sort_keys = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.column_from("table"),
                    rlz.function_of("table"),
                    rlz.sort_key(from_="table"),
                    rlz.pair(
                        rlz.one_of(
                            (
                                rlz.column_from("table"),
                                rlz.function_of("table"),
                                rlz.any,
                            )
                        ),
                        rlz.map_to(
                            {
                                True: True,
                                False: False,
                                "desc": False,
                                "descending": False,
                                "asc": True,
                                "ascending": True,
                                1: True,
                                0: False,
                            }
                        ),
                    ),
                )
            )
        ),
        default=[],
    )

    def _validate(self):
        from ibis.expr.analysis import FilterValidator

        # Need to validate that the column expressions are compatible with the
        # input table; this means they must either be scalar expressions or
        # array expressions originating from the same root table expression
        dependent_exprs = self.selections + self.sort_keys
        self.table._assert_valid(dependent_exprs)

        # Validate predicates
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

        # Validate no overlapping columns in schema
        assert self.schema

    @cached_property
    def schema(self):
        # Resolve schema and initialize
        if not self.selections:
            return self.table.schema()

        types = []
        names = []

        for projection in self.selections:
            if isinstance(projection, ir.DestructColumn):
                # If this is a destruct, then we destructure
                # the result and assign to multiple columns
                struct_type = projection.type()
                for name in struct_type.names:
                    names.append(name)
                    types.append(struct_type[name])
            elif isinstance(projection, ir.ValueExpr):
                names.append(projection.get_name())
                types.append(projection.type())
            elif isinstance(projection, ir.TableExpr):
                schema = projection.schema()
                names.extend(schema.names)
                types.extend(schema.types)

        return sch.Schema(names, types)

    def blocks(self):
        return bool(self.selections)

    def substitute_table(self, table_expr):
        return Selection(table_expr, self.selections)

    def root_tables(self):
        return [self]

    def can_add_filters(self, wrapped_expr, predicates):
        pass

    @staticmethod
    def empty_or_equal(lefts, rights):
        return not lefts or not rights or all_equal(lefts, rights)

    def compatible_with(self, other):
        # self and other are equivalent except for predicates, selections, or
        # sort keys any of which is allowed to be empty. If both are not empty
        # then they must be equal
        if self.equals(other):
            return True

        if not isinstance(other, type(self)):
            return False

        return self.table.equals(other.table) and (
            self.empty_or_equal(self.predicates, other.predicates)
            and self.empty_or_equal(self.selections, other.selections)
            and self.empty_or_equal(self.sort_keys, other.sort_keys)
        )

    # Operator combination / fusion logic

    def aggregate(self, this, metrics, by=None, having=None):
        if len(self.selections) > 0:
            return Aggregation(this, metrics, by=by, having=having)
        else:
            helper = AggregateSelection(this, metrics, by, having)
            return helper.get_result()

    def sort_by(self, expr, sort_exprs):
        resolved_keys = _maybe_convert_sort_keys(
            [self.table, expr], sort_exprs
        )
        if not self.blocks():
            if self.table._is_valid(resolved_keys):
                return Selection(
                    self.table,
                    self.selections,
                    predicates=self.predicates,
                    sort_keys=self.sort_keys + resolved_keys,
                )

        return Selection(expr, [], sort_keys=resolved_keys)


@public
class AggregateSelection:
    # sort keys cannot be discarded because of order-dependent
    # aggregate functions like GROUP_CONCAT

    def __init__(self, parent, metrics, by, having):
        self.parent = parent
        self.op = parent.op()
        self.metrics = metrics
        self.by = by
        self.having = having

    def get_result(self):
        if self.op.blocks():
            return self._plain_subquery()
        else:
            return self._attempt_pushdown()

    def _plain_subquery(self):
        return Aggregation(
            self.parent, self.metrics, by=self.by, having=self.having
        )

    def _attempt_pushdown(self):
        metrics_valid, lowered_metrics = self._pushdown_exprs(self.metrics)
        by_valid, lowered_by = self._pushdown_exprs(self.by)
        having_valid, lowered_having = self._pushdown_exprs(self.having)

        if metrics_valid and by_valid and having_valid:
            return Aggregation(
                self.op.table,
                lowered_metrics,
                by=lowered_by,
                having=lowered_having,
                predicates=self.op.predicates,
                sort_keys=self.op.sort_keys,
            )
        else:
            return self._plain_subquery()

    def _pushdown_exprs(self, exprs):
        import ibis.expr.analysis as L

        # exit early if there's nothing to push down
        if not exprs:
            return True, []

        resolved = self.op.table._resolve(exprs)
        subbed_exprs = []

        valid = False
        if resolved:
            for x in util.promote_list(resolved):
                subbed = L.sub_for(x, [(self.parent, self.op.table)])
                subbed_exprs.append(subbed)
            valid = self.op.table._is_valid(subbed_exprs)
        else:
            valid = False

        return valid, subbed_exprs


@public
class Aggregation(TableNode, sch.HasSchema):

    """
    metrics : per-group scalar aggregates
    by : group expressions
    having : post-aggregation predicate

    TODO: not putting this in the aggregate operation yet
    where : pre-aggregation predicate
    """

    table = Arg(rlz.table)
    metrics = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.function_of(
                        "table",
                        output_rule=rlz.one_of(
                            (rlz.reduction, rlz.scalar(rlz.any))
                        ),
                    ),
                    rlz.reduction,
                    rlz.scalar(rlz.any),
                    rlz.list_of(rlz.scalar(rlz.any)),
                    rlz.named_literal,
                )
            ),
            flatten=True,
        ),
        default=[],
    )
    by = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.function_of("table"),
                    rlz.column_from("table"),
                    rlz.column(rlz.any),
                )
            )
        ),
        default=[],
    )
    having = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.function_of(
                        "table", output_rule=rlz.scalar(rlz.boolean)
                    ),
                    rlz.scalar(rlz.boolean),
                )
            ),
        ),
        default=[],
    )
    predicates = Arg(rlz.list_of(rlz.boolean), default=[])
    sort_keys = Arg(
        rlz.list_of(
            rlz.one_of(
                (
                    rlz.column_from("table"),
                    rlz.function_of("table"),
                    rlz.sort_key(from_="table"),
                    rlz.pair(
                        rlz.one_of(
                            (
                                rlz.column_from("table"),
                                rlz.function_of("table"),
                                rlz.any,
                            )
                        ),
                        rlz.map_to(
                            {
                                True: True,
                                False: False,
                                "desc": False,
                                "descending": False,
                                "asc": True,
                                "ascending": True,
                                1: True,
                                0: False,
                            }
                        ),
                    ),
                )
            )
        ),
        default=[],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.by:
            self.sort_keys.clear()

    def _validate(self):
        from ibis.expr.analysis import FilterValidator

        # All non-scalar refs originate from the input table
        all_exprs = self.metrics + self.by + self.having + self.sort_keys
        self.table._assert_valid(all_exprs)

        # Validate predicates
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

        # Validate schema has no overlapping columns
        assert self.schema

    def blocks(self):
        return True

    def substitute_table(self, table_expr):
        return Aggregation(
            table_expr, self.metrics, by=self.by, having=self.having
        )

    @cached_property
    def schema(self):
        names = []
        types = []

        for e in self.by + self.metrics:
            if isinstance(e, ir.DestructValue):
                # If this is a destruct, then we destructure
                # the result and assign to multiple columns
                struct_type = e.type()
                for name in struct_type.names:
                    names.append(name)
                    types.append(struct_type[name])
            else:
                names.append(e.get_name())
                types.append(e.type())

        return sch.Schema(names, types)

    def sort_by(self, expr, sort_exprs):
        resolved_keys = _maybe_convert_sort_keys(
            [self.table, expr], sort_exprs
        )
        if self.table._is_valid(resolved_keys):
            return Aggregation(
                self.table,
                self.metrics,
                by=self.by,
                having=self.having,
                predicates=self.predicates,
                sort_keys=self.sort_keys + resolved_keys,
            )

        return Selection(expr, [], sort_keys=resolved_keys)


@public
class Distinct(TableNode, sch.HasSchema):
    """
    Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """

    table = Arg(rlz.table)

    def _validate(self):
        # check whether schema has overlapping columns or not
        assert self.schema

    @cached_property
    def schema(self):
        return self.table.schema()

    def blocks(self):
        return True


@public
class ExistsSubquery(Node):
    foreign_table = Arg(rlz.table)
    predicates = Arg(rlz.list_of(rlz.boolean))

    def output_type(self):
        return ir.ExistsExpr


@public
class NotExistsSubquery(Node):
    foreign_table = Arg(rlz.table)
    predicates = Arg(rlz.list_of(rlz.boolean))

    def output_type(self):
        return ir.ExistsExpr


@public
class FillNa(TableNode, sch.HasSchema):
    """Fill null values in the table."""

    table = Arg(rlz.table)
    replacements = Arg(
        rlz.one_of(
            (
                rlz.numeric,
                rlz.string,
                rlz.instance_of(collections.abc.Mapping),
            )
        )
    )

    @cached_property
    def schema(self):
        return self.table.schema()


@public
class DropNa(TableNode, sch.HasSchema):
    """Drop null values in the table."""

    table = Arg(rlz.table)
    how = Arg(rlz.isin({'any', 'all'}))
    subset = Arg(rlz.list_of(rlz.column_from("table")), default=[])

    @cached_property
    def schema(self):
        return self.table.schema()
