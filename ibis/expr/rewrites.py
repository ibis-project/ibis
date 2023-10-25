"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.deferred import var
from ibis.common.exceptions import UnsupportedOperationError
from ibis.common.patterns import Eq, In, NoMatch, pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)


@replace(p.FillNa)
def rewrite_fillna(_):
    """Rewrite FillNa expressions to use more common operations."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.table.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.table

    selections = []
    for name in _.table.schema.names:
        col = ops.TableColumn(_.table, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Alias(ops.Coalesce((col, value)), name)
        selections.append(col)

    return ops.Selection(_.table, selections, (), ())


@replace(p.DropNa)
def rewrite_dropna(_):
    """Rewrite DropNa expressions to use more common operations."""
    if _.subset is None:
        columns = [ops.TableColumn(_.table, name) for name in _.table.schema.names]
    else:
        columns = _.subset

    if columns:
        preds = [
            functools.reduce(
                ops.And if _.how == "any" else ops.Or,
                [ops.NotNull(c) for c in columns],
            )
        ]
    elif _.how == "all":
        preds = [ops.Literal(False, dtype=dt.bool)]
    else:
        return _.table

    return ops.Selection(_.table, (), preds, ())


@replace(p.Sample)
def rewrite_sample(_):
    """Rewrite Sample as `t.filter(random() <= fraction)`.

    Errors as unsupported if a `seed` is specified.
    """

    if _.seed is not None:
        raise UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )

    return ops.Selection(
        _.table,
        (),
        (ops.LessEqual(ops.RandomScalar(), _.fraction),),
        (),
    )


x = var("x")
y = var("y")
t = var("t")
r = var("r")
parent = var("parent")
fields = var("fields")
parent_fields = var("parent_fields")
peeled_fields = var("peeled_fields")

# reprojection of the same fields from the parent selection
rewrite_redundant_selection = (
    p.Selection(x, selections=[p.Selection(selections=Eq(x.selections))]) >> x
)

# the following rewrite rule is responsible to fuse the following two non-overlapping
# selections into a single one:
#
# INPUT:
#
# r0 := UnboundTable: t
#   col int32
#
# r1 := Selection[r0]
#   selections:
#     r0
#     col1: r0.col + 1
#
# Selection[r1]
#   selections:
#     r1
#     col2: r0.col + 2
#
# OUTPUT:
#
# r0 := UnboundTable: t
#   col int32

# Selection[r0]
#   selections:
#     r0
#     col1: r0.col + 1
#     col2: r0.col + 2


def can_prune_parent_projection(selection, context):
    parent = context["parent"]
    fields = context["fields"]
    parent_fields = context["parent_fields"]

    columns, conflicts = {}, {}
    for value in parent_fields:
        if isinstance(value, ops.Relation):
            # TODO(kszucs): this makes the rewrite rule really sloggy
            for name in value.schema:
                columns[name] = ops.TableColumn(value, name)
        elif isinstance(value, ops.TableColumn):
            columns[value.name] = value
        elif value.find((ops.WindowFunction, ops.ExistsSubquery), filter=ops.Value):
            # the parent has analytic projections like window functions so we
            # can't push down filters to that level
            return NoMatch
        else:
            # otherwise collect the names of newly projected value expressions
            # which are not just plain column references
            conflicts[value.name] = value

    peeled_fields = []
    reprojected_columns = p.TableColumn(parent, In(columns)) >> (
        lambda _: columns[_.name]
    )
    conflicting_columns = p.TableColumn(parent, In(conflicts))

    for field in fields:
        if field == parent:
            if parent_fields:
                peeled_fields.extend(parent_fields)
            else:
                # order_by() and filter() creates selections objects with empty
                # selections field, so we need to add the parent table to the
                # selections explicitly, should add selections=[self] there instead
                peeled_fields.append(parent.table)
        elif field.match(conflicting_columns, filter=ops.Value):
            return NoMatch
        else:
            field = field.replace(reprojected_columns, filter=ops.Value)
            peeled_fields.append(field)

    context["peeled_fields"] = peeled_fields
    return selection


# TODO(kszucs): merge this rewrite rule pushdown_selection_filters() and
# simplify_aggregation() analysis functions since their logic is almost
# identical
@replace(
    p.Selection(parent @ p.Selection(selections=parent_fields), selections=fields)
    & can_prune_parent_projection
)
def prune_subsequent_projection(_, parent, fields, parent_fields, peeled_fields):
    # needed to support the ibis/tests/sql/test_select_sql.py::test_fuse_projections
    # test case which wouldn't work with Eq(parent) since it calls
    # filtered_table.select(table.field) referencing a different but semantically
    # equivalent table object r2 instead of r1
    #
    # r0 := UnboundTable: tbl
    #   foo   int32
    #   bar   int64
    #   value float64
    #
    # r1 := Selection[r0]
    #   predicates:
    #     r0.value > 0
    #   selections:
    #     r0
    #     baz: r0.foo + r0.bar
    #
    # r2 := Selection[r0]
    #   selections:
    #     r0
    #     baz: r0.foo + r0.bar
    #
    # Selection[r1]
    #   selections:
    #     r2
    #     qux: r2.foo * 2
    traverse_only = p.Value | p.Selection
    pattern = p.Projection(parent.table, parent.selections) >> parent.table
    selections = [
        field.replace(pattern, filter=traverse_only) for field in peeled_fields
    ]
    return parent.copy(selections=selections)
