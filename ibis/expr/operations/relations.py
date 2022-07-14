from __future__ import annotations

import collections
import itertools
from abc import abstractmethod

from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.common.validators import immutable_property
from ibis.expr.operations.core import Node, Value
from ibis.expr.operations.logical import ExistsSubquery, NotExistsSubquery
from ibis.expr.operations.sortkeys import _maybe_convert_sort_keys

_table_names = (f'unbound_table_{i:d}' for i in itertools.count())


@public
def genname():
    return next(_table_names)


@public
class TableNode(Node):
    def sort_by(self, expr, sort_exprs):
        return Selection(
            expr,
            [],
            sort_keys=_maybe_convert_sort_keys(
                [expr],
                sort_exprs,
            ),
        )

    @property
    @abstractmethod
    def schema(self) -> sch.Schema:
        """Return a schema."""

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.Table(self)


@public
class PhysicalTable(TableNode):
    pass


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
class SQLQueryResult(TableNode):
    """A table sourced from the result set of a select query"""

    query = rlz.instance_of(str)
    schema = rlz.instance_of(sch.Schema)
    source = rlz.client


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


# TODO(kszucs): desperately need to clean this up, the majority of this
# functionality should be handled by input rules for the Join class
def _clean_join_predicates(left, right, predicates):
    import ibis.expr.analysis as L
    import ibis.expr.types as ir
    from ibis.expr.analysis import shares_all_roots

    result = []

    for pred in predicates:
        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise com.ExpressionError('Join key tuple must be ' 'length 2')
            lk, rk = pred
            lk = left.to_expr()._ensure_expr(lk)
            rk = right.to_expr()._ensure_expr(rk)
            pred = lk == rk
        elif isinstance(pred, str):
            pred = left.to_expr()[pred] == right.to_expr()[pred]
        elif isinstance(pred, Value):
            pred = pred.to_expr()
        elif not isinstance(pred, ir.Expr):
            raise NotImplementedError

        if not isinstance(pred, ir.BooleanColumn):
            raise com.ExpressionError('Join predicate must be comparison')

        preds = L.flatten_predicate(pred.op())
        result.extend(preds)

    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    for predicate in result:
        if not shares_all_roots(predicate, [left, right]):
            raise com.RelationError(
                'The expression {!r} does not fully '
                'originate from dependencies of the table '
                'expression.'.format(predicate)
            )

    assert all(isinstance(pred, ops.Node) for pred in result)

    return tuple(result)


@public
class Join(TableNode):
    left = rlz.table
    right = rlz.table
    predicates = rlz.optional(lambda x, this: x, default=())

    def __init__(self, left, right, predicates, **kwargs):
        # TODO(kszucs): predicates should be already a list of operations, need
        # to update the validation rule for the Join classes which is a noop
        # currently
        import ibis.expr.analysis as L
        import ibis.expr.operations as ops

        # TODO(kszucs): need to factor this out to appropiate join predicate
        # rules
        predicates = [
            pred.op() if isinstance(pred, ir.Expr) else pred
            for pred in util.promote_list(predicates)
        ]

        if left.equals(right):
            # GH #667: If left and right table have a common parent expression,
            # e.g. they have different filters, we need to add a self-reference
            # and make the appropriate substitution in the join predicates
            right = ops.SelfReference(right)
        elif isinstance(right, Join):
            # for joins with joins on the right side we turn the right side
            # into a view, otherwise the join tree is incorrectly flattened
            # and tables on the right are incorrectly scoped
            old = right
            new = right = ops.SelfReference(right)
            predicates = [
                L.sub_for(pred, {old: new})
                if isinstance(pred, ops.Node)
                else pred
                for pred in predicates
            ]

        predicates = _clean_join_predicates(left, right, predicates)

        super().__init__(
            left=left, right=right, predicates=predicates, **kwargs
        )

    @property
    def schema(self):
        # For joins retaining both table schemas, merge them together here
        return self.left.schema.append(self.right.schema)


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
        return self.left.schema


@public
class LeftAntiJoin(Join):
    @property
    def schema(self):
        return self.left.schema


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
class SetOp(TableNode):
    left = rlz.table
    right = rlz.table
    distinct = rlz.optional(rlz.instance_of(bool), default=False)

    def __init__(self, left, right, **kwargs):
        if not left.schema == right.schema:
            raise com.RelationError(
                'Table schemas must be equal for set operations'
            )
        super().__init__(left=left, right=right, **kwargs)

    @property
    def schema(self):
        return self.left.schema


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

    @property
    def schema(self):
        return self.table.schema


@public
class SelfReference(TableNode):
    table = rlz.table

    @property
    def schema(self):
        return self.table.schema


class Projection(TableNode):
    table = rlz.table
    selections = rlz.tuple_of(
        rlz.one_of(
            (
                rlz.table,
                rlz.column_from("table"),
                rlz.function_of("table"),
                rlz.any,
            )
        )
    )

    @immutable_property
    def schema(self):
        # Resolve schema and initialize

        if not self.selections:
            return self.table.schema

        types = []
        names = []

        for projection in self.selections:
            if isinstance(projection, Value):
                names.append(projection.resolve_name())
                types.append(projection.output_dtype)
            elif isinstance(projection, TableNode):
                schema = projection.schema
                names.extend(schema.names)
                types.extend(schema.types)

        return sch.Schema(names, types)


@public
class Selection(Projection):
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

    def sort_by(self, expr, sort_exprs):
        from ibis.expr.analysis import shares_all_roots

        resolved_keys = _maybe_convert_sort_keys(
            [self.table, expr], sort_exprs
        )
        if not self.selections:
            if shares_all_roots(resolved_keys, self.table):
                return Selection(
                    self.table,
                    self.selections,
                    predicates=self.predicates,
                    sort_keys=self.sort_keys + tuple(resolved_keys),
                )

        return Selection(expr, [], sort_keys=resolved_keys)


@public
class Aggregation(TableNode):

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

    @immutable_property
    def schema(self):
        names = []
        types = []

        for e in self.by + self.metrics:
            names.append(e.resolve_name())
            types.append(e.output_dtype)

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
class Distinct(TableNode):
    """
    Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """

    table = rlz.table

    @property
    def schema(self):
        return self.table.schema


@public
class FillNa(TableNode):
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
        return self.table.schema


@public
class DropNa(TableNode):
    """Drop null values in the table."""

    table = rlz.table
    how = rlz.isin({'any', 'all'})
    subset = rlz.optional(rlz.tuple_of(rlz.column_from("table")))

    @property
    def schema(self):
        return self.table.schema


@public
class View(PhysicalTable):
    """A view created from an expression."""

    child = rlz.table
    name = rlz.instance_of(str)

    @property
    def schema(self):
        return self.child.schema


@public
class SQLStringView(PhysicalTable):
    """A view created from a SQL string."""

    child = rlz.table
    name = rlz.instance_of(str)
    query = rlz.instance_of(str)

    @immutable_property
    def schema(self):
        # TODO(kszucs): avoid converting to expression
        backend = self.child.to_expr()._find_backend()
        return backend._get_schema_using_query(self.query)


def _dedup_join_columns(expr, suffixes: tuple[str, str]):
    op = expr.op()
    left = op.left.to_expr()
    right = op.right.to_expr()

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
