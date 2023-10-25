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


parent = var("parent")
fields = var("fields")
parent_fields = var("parent_fields")


# reprojection of the same fields from the parent selection
rewrite_redundant_selection = (
    p.Selection(parent, selections=[p.Selection(selections=Eq(parent.selections))])
    >> parent
)


def can_prune_parent_projection(selection, context):
    parent = context["parent"]
    fields = context["fields"]
    parent_fields = context["parent_fields"]

    conflicts = {}
    for value in parent_fields:
        if isinstance(value, (ops.Relation, ops.TableColumn)):
            continue
        elif value.find((ops.WindowFunction, ops.ExistsSubquery), filter=ops.Value):
            # the parent has analytic projections like window functions so we
            # can't push down filters to that level
            return NoMatch
        else:
            # otherwise collect the names of newly projected value expressions
            # which are not just plain column references
            conflicts[value.name] = value

    conflicting_columns = p.TableColumn(parent, In(conflicts))
    for field in fields:
        if field.match(conflicting_columns, filter=ops.Value):
            # the field references a newly projected value expression from the
            # parent selection so we can't eliminate the parent projection
            return NoMatch

    return selection


# TODO(kszucs): merge this rewrite rule pushdown_selection_filters() and
# simplify_aggregation() analysis functions since their logic is almost
# identical
@replace(
    p.Selection(parent @ p.Selection(selections=parent_fields), selections=fields)
    & can_prune_parent_projection
)
def prune_subsequent_projection(_, parent, fields, parent_fields, **kwargs):
    # create a mapping of column names to projected value expressions from the parent
    column_lookup = {}
    parent_fields = parent_fields or [parent.table]
    for field in parent_fields:
        if isinstance(field, ops.Relation):
            for name in field.schema:
                column_lookup[name] = ops.TableColumn(field, name)
        else:
            column_lookup[field.name] = field

    # Eq() is an optimization to avoid deeper pattern matching
    substitute_parent = p.Selection(parent.table, Eq(parent.selections)) >> parent.table
    # replace parent columns with the corresponding value from the lookup table
    lookup_from_parent = p.TableColumn(parent) >> (lambda _: column_lookup[_.name])

    selections = []
    for field in fields:
        if field == parent:
            if parent_fields:
                # the parent selection is referenced directly, so we need to
                # include all of its fields in the new selection
                selections.extend(parent_fields)
            else:
                # order_by() and filter() creates selections objects with empty
                # selections field, so we need to add the parent table to the
                # selections explicitly, should add selections=[self] there instead
                selections.append(parent.table)
        elif isinstance(field, ops.TableColumn):
            # faster branch for simple column references
            if field.table == parent:
                field = column_lookup[field.name]
            selections.append(field)
        else:
            # need to replace columns referencing the parent table with the
            # corresponding projected value expressions from the parent selection
            field = field.replace(lookup_from_parent, filter=ops.Value)
            # replace any leftover references to the parent selection which may
            # be in e.g. WindowFrame operations
            field = field.replace(substitute_parent, filter=p.Value | p.Selection)
            selections.append(field)

    return parent.copy(selections=selections)
