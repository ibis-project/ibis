from __future__ import annotations

import collections
import itertools
from abc import abstractmethod
from functools import cached_property

from public import public

from ibis import util
from ibis.common import exceptions as com
from ibis.expr import rules as rlz
from ibis.expr import schema as sch
from ibis.expr import types as ir
from ibis.expr.operations.core import Node
from ibis.expr.operations.logical import ExistsSubquery, NotExistsSubquery
from ibis.expr.operations.sortkeys import _maybe_convert_sort_keys

_table_names = (f'unbound_table_{i:d}' for i in itertools.count())


@public
def genname():
    return next(_table_names)


@public
class TableNode(Node):
    output_type = ir.Table

    @util.deprecated(
        instead="Use table.schema()[name] instead", version="4.0.0"
    )
    def get_type(self, name):  # pragma: no cover
        return self.schema[name]

    def aggregate(self, this, metrics, by=None, having=None):
        return Aggregation(this, metrics, by=by, having=having)

    def sort_by(self, expr, sort_exprs):
        return Selection(
            expr,
            [],
            sort_keys=_maybe_convert_sort_keys(
                [expr],
                sort_exprs,
            ),
        )

    @util.deprecated(version="4.0.0", instead="")
    def is_ancestor(self, other):  # pragma: no cover
        from ibis.expr.analysis import _is_ancestor

        return _is_ancestor(self, other)

    def root_tables(self):
        return [self]


@public
class PhysicalTable(TableNode, sch.HasSchema):
    def blocks(self):
        return True


@public
class UnboundTable(PhysicalTable):
    schema = rlz.instance_of(sch.Schema)
    name = rlz.optional(rlz.instance_of(str), default=genname)

    def has_resolved_name(self):
        return True

    def resolve_name(self):
        return self.name


@public
class DatabaseTable(PhysicalTable):
    name = rlz.instance_of(str)
    schema = rlz.instance_of(sch.Schema)
    source = rlz.client

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


@public
class SQLQueryResult(TableNode, sch.HasSchema):
    """A table sourced from the result set of a select query"""

    query = rlz.instance_of(str)
    schema = rlz.instance_of(sch.Schema)
    source = rlz.client

    def blocks(self):
        return True


@public
class InMemoryTable(PhysicalTable):
    name = rlz.instance_of(str)
    schema = rlz.instance_of(sch.Schema)

    @property
    @abstractmethod
    def data(self) -> util.ToFrame:
        """Return the data of an in-memory table."""

    def has_resolved_name(self):
        return True

    def resolve_name(self):
        return self.name


def _make_distinct_join_predicates(left, right, predicates):
    import ibis.expr.analysis as L

    if left.equals(right):
        # GH #667: If left and right table have a common parent expression,
        # e.g. they have different filters, we need to add a self-reference and
        # make the appropriate substitution in the join predicates
        right = right.view()
    elif isinstance(right.op(), Join):
        # for joins with joins on the right side we turn the right side into a
        # view, otherwise the join tree is incorrectly flattened and tables on
        # the right are incorrectly scoped
        old = right
        new = right = right.view()
        predicates = [
            L.sub_for(pred, [(old, new)])
            if isinstance(pred, ir.Expr)
            else pred
            for pred in predicates
        ]

    predicates = _clean_join_predicates(left, right, predicates)
    return left, right, predicates


def _clean_join_predicates(left, right, predicates):
    import ibis.expr.analysis as L

    result = []

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
    return tuple(result)


def _validate_join_predicates(left, right, predicates):
    from ibis.expr.analysis import shares_all_roots

    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    for predicate in predicates:
        if not shares_all_roots(predicate, [left, right]):
            raise com.RelationError(
                'The expression {!r} does not fully '
                'originate from dependencies of the table '
                'expression.'.format(predicate)
            )


@public
class Join(TableNode):
    left = rlz.table
    right = rlz.table
    # TODO(kszucs): convert to proper predicate rules
    predicates = rlz.optional(lambda x, this: x, default=())

    def __init__(self, left, right, predicates, **kwargs):
        left, right, predicates = _make_distinct_join_predicates(
            left, right, util.promote_list(predicates)
        )
        super().__init__(
            left=left, right=right, predicates=predicates, **kwargs
        )

    @property
    def schema(self):
        # For joins retaining both table schemas, merge them together here
        return self.left.schema().append(self.right.schema())

    @util.deprecated(version="4.0", instead="")
    def has_schema(self):
        return not set(self.left.columns) & set(self.right.columns)


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
    @property
    def schema(self):
        return self.left.schema()


@public
class LeftAntiJoin(Join):
    @property
    def schema(self):
        return self.left.schema()


@public
class CrossJoin(Join):
    pass


@public
class AsOfJoin(Join):
    # TODO(kszucs): convert to proper predicate rules
    by = rlz.optional(lambda x, this: x, default=())
    tolerance = rlz.optional(rlz.interval)

    def __init__(self, left, right, by, predicates, **kwargs):
        by = _clean_join_predicates(left, right, util.promote_list(by))
        super().__init__(
            left=left, right=right, by=by, predicates=predicates, **kwargs
        )


@public
class SetOp(TableNode, sch.HasSchema):
    left = rlz.table
    right = rlz.table
    distinct = rlz.optional(rlz.instance_of(bool), default=False)

    def __init__(self, left, right, distinct: bool, **kwargs):
        if not left.schema().equals(right.schema()):
            raise com.RelationError(
                'Table schemas must be equal for set operations'
            )
        super().__init__(left=left, right=right, distinct=distinct, **kwargs)

    @property
    def schema(self):
        return self.left.schema()

    def blocks(self):
        return True


@public
class Union(SetOp):
    pass


@public
class Intersection(SetOp):
    pass


@public
class Difference(SetOp):
    pass


@public
class Limit(TableNode):
    table = rlz.table
    n = rlz.instance_of(int)
    offset = rlz.instance_of(int)

    def blocks(self):
        return True

    @property
    def schema(self):
        return self.table.schema()

    @util.deprecated(version="4.0", instead="")
    def has_schema(self):
        return self.table.op().has_schema()


@public
class SelfReference(TableNode, sch.HasSchema):
    table = rlz.table

    @property
    def schema(self):
        return self.table.schema()

    def blocks(self):
        return True


@public
class Selection(TableNode, sch.HasSchema):
    table = rlz.table
    selections = rlz.optional(
        rlz.tuple_of(
            rlz.one_of(
                (
                    rlz.table,
                    rlz.column_from("table"),
                    rlz.function_of("table"),
                    rlz.any,
                )
            )
        ),
        default=(),
    )
    predicates = rlz.optional(rlz.tuple_of(rlz.boolean), default=())
    sort_keys = rlz.optional(
        rlz.tuple_of(
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
        default=(),
    )

    def __init__(self, table, selections, predicates, sort_keys, **kwargs):
        from ibis.expr.analysis import shares_all_roots, shares_some_roots

        if not shares_all_roots(selections + sort_keys, table):
            raise com.RelationError(
                "Selection expressions don't fully originate from "
                "dependencies of the table expression."
            )

        for predicate in predicates:
            if not shares_some_roots(predicate, table):
                raise com.RelationError(
                    "Predicate doesn't share any roots with table"
                )

        super().__init__(
            table=table,
            selections=selections,
            predicates=predicates,
            sort_keys=sort_keys,
            **kwargs,
        )

        # Validate no overlapping columns in schema
        assert self.schema

    @cached_property
    def _projection(self):
        return self.__class__(table=self.table, selections=self.selections)

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
            elif isinstance(projection, ir.Value):
                names.append(projection.get_name())
                types.append(projection.type())
            elif isinstance(projection, ir.Table):
                schema = projection.schema()
                names.extend(schema.names)
                types.extend(schema.types)

        return sch.Schema(names, types)

    def blocks(self):
        return bool(self.selections)

    @util.deprecated(instead="instantiate Selection directly", version="4.0.0")
    def substitute_table(self, table_expr):  # pragma: no cover
        return Selection(table_expr, self.selections)

    @util.deprecated(instead="", version="4.0.0")
    def can_add_filters(self, wrapped_expr, predicates):  # pragma: no cover
        pass

    @util.deprecated(instead="", version="4.0.0")
    def empty_or_equal(self, other) -> bool:  # pragma: no cover
        for field in "selections", "sort_keys", "predicates":
            selfs = getattr(self, field)
            others = getattr(other, field)
            valid = (
                not selfs
                or not others
                or (a.equals(b) for a, b in zip(selfs, others))
            )
            if not valid:
                return False
        return True

    @util.deprecated(instead="", version="4.0.0")
    def compatible_with(self, other):  # pragma: no cover
        # self and other are equivalent except for predicates, selections, or
        # sort keys any of which is allowed to be empty. If both are not empty
        # then they must be equal
        if self.equals(other):
            return True

        if not isinstance(other, type(self)):
            return False

        return self.table.equals(other.table) and self.empty_or_equal(other)

    def aggregate(self, this, metrics, by=None, having=None):
        if len(self.selections) > 0:
            return Aggregation(this, metrics, by=by, having=having)
        else:
            helper = AggregateSelection(this, metrics, by, having)
            return helper.get_result()

    def sort_by(self, expr, sort_exprs):
        from ibis.expr.analysis import shares_all_roots

        resolved_keys = _maybe_convert_sort_keys(
            [self.table, expr], sort_exprs
        )
        if not self.blocks():
            if shares_all_roots(resolved_keys, self.table):
                return Selection(
                    self.table,
                    self.selections,
                    predicates=self.predicates,
                    sort_keys=self.sort_keys + tuple(resolved_keys),
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
        from ibis.expr.analysis import shares_all_roots, sub_for

        subbed_exprs = []
        for expr in util.promote_list(exprs):
            expr = self.op.table._ensure_expr(expr)
            subbed = sub_for(expr, [(self.parent, self.op.table)])
            subbed_exprs.append(subbed)

        if subbed_exprs:
            valid = shares_all_roots(subbed_exprs, self.op.table)
        else:
            valid = True

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

    table = rlz.table
    metrics = rlz.optional(
        rlz.tuple_of(
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
                    rlz.tuple_of(rlz.scalar(rlz.any)),
                )
            ),
            flatten=True,
        ),
        default=(),
    )
    by = rlz.optional(
        rlz.tuple_of(
            rlz.one_of(
                (
                    rlz.function_of("table"),
                    rlz.column_from("table"),
                    rlz.column(rlz.any),
                )
            )
        ),
        default=(),
    )
    having = rlz.optional(
        rlz.tuple_of(
            rlz.one_of(
                (
                    rlz.function_of(
                        "table", output_rule=rlz.scalar(rlz.boolean)
                    ),
                    rlz.scalar(rlz.boolean),
                )
            ),
        ),
        default=(),
    )
    predicates = rlz.optional(rlz.tuple_of(rlz.boolean), default=())
    sort_keys = rlz.optional(
        rlz.tuple_of(
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
        default=(),
    )

    def __init__(self, table, metrics, by, having, predicates, sort_keys):
        from ibis.expr.analysis import shares_all_roots, shares_some_roots

        # All non-scalar refs originate from the input table
        if not shares_all_roots(metrics + by + having + sort_keys, table):
            raise com.RelationError(
                "Selection expressions don't fully originate from "
                "dependencies of the table expression."
            )

        # invariant due to Aggregation and AggregateSelection requiring a valid
        # Selection
        assert all(
            shares_some_roots(predicate, table) for predicate in predicates
        )

        if not by:
            sort_keys = tuple()

        super().__init__(
            table=table,
            metrics=metrics,
            by=by,
            having=having,
            predicates=predicates,
            sort_keys=sort_keys,
        )
        # Validate schema has no overlapping columns
        assert self.schema

    def blocks(self):
        return True

    @util.deprecated(
        instead="instantiate Aggregation directly", version="4.0.0"
    )
    def substitute_table(self, table_expr):  # pragma: no cover
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
        from ibis.expr.analysis import shares_all_roots

        resolved_keys = _maybe_convert_sort_keys(
            [self.table, expr], sort_exprs
        )
        if shares_all_roots(resolved_keys, self.table):
            return Aggregation(
                self.table,
                self.metrics,
                by=self.by,
                having=self.having,
                predicates=self.predicates,
                sort_keys=self.sort_keys + tuple(resolved_keys),
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

    table = rlz.table

    def __init__(self, table):
        # check whether schema has overlapping columns or not
        assert table.schema()
        super().__init__(table=table)

    @property
    def schema(self):
        return self.table.schema()

    def blocks(self):
        return True


@public
class FillNa(TableNode, sch.HasSchema):
    """Fill null values in the table."""

    table = rlz.table
    replacements = rlz.one_of(
        (
            rlz.numeric,
            rlz.string,
            rlz.instance_of(collections.abc.Mapping),
        )
    )

    def __init__(self, table, replacements, **kwargs):
        super().__init__(
            table=table,
            replacements=(
                replacements
                if not isinstance(replacements, collections.abc.Mapping)
                else util.frozendict(replacements)
            ),
            **kwargs,
        )

    @property
    def schema(self):
        return self.table.schema()


@public
class DropNa(TableNode, sch.HasSchema):
    """Drop null values in the table."""

    table = rlz.table
    how = rlz.isin({'any', 'all'})
    subset = rlz.optional(rlz.tuple_of(rlz.column_from("table")), default=None)

    @property
    def schema(self):
        return self.table.schema()


@public
class View(PhysicalTable):
    """A view created from an expression."""

    child = rlz.table
    name = rlz.instance_of(str)

    @property
    def schema(self):
        return self.child.schema()


@public
class SQLStringView(PhysicalTable):
    """A view created from a SQL string."""

    child = rlz.table
    name = rlz.instance_of(str)
    query = rlz.instance_of(str)

    @cached_property
    def schema(self):
        backend = self.child._find_backend()
        return backend._get_schema_using_query(self.query)


def _dedup_join_columns(
    expr: ir.Table,
    *,
    left: ir.Table,
    right: ir.Table,
    suffixes: tuple[str, str],
):
    right_columns = frozenset(right.columns)
    overlap = frozenset(
        column for column in left.columns if column in right_columns
    )

    if not overlap:
        return expr

    left_suffix, right_suffix = suffixes

    left_projections = [
        left[column].name(f"{column}{left_suffix}")
        if column in overlap
        else left[column]
        for column in left.columns
    ]

    right_projections = [
        right[column].name(f"{column}{right_suffix}")
        if column in overlap
        else right[column]
        for column in right.columns
    ]
    return expr.projection(left_projections + right_projections)


public(ExistsSubquery=ExistsSubquery, NotExistsSubquery=NotExistsSubquery)
