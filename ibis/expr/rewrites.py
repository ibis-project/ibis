"""Some common rewrite functions to be shared between backends."""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Mapping

import toolz

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import ExpressionError, IbisInputError
from ibis.common.graph import Node as Traversable
from ibis.common.grounds import Concrete
from ibis.common.patterns import Check, pattern, replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.util import Namespace, promote_list

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


x = var("x")
y = var("y")
name = var("name")


class DerefMap(Concrete, Traversable):
    """Trace and replace fields from earlier relations in the hierarchy.

    In order to provide a nice user experience, we need to allow expressions
    from earlier relations in the hierarchy. Consider the following example:

    t = ibis.table([('a', 'int64'), ('b', 'string')], name='t')
    t1 = t.select([t.a, t.b])
    t2 = t1.filter(t.a > 0)  # note that not t1.a is referenced here
    t3 = t2.select(t.a)  # note that not t2.a is referenced here

    However the relational operations in the IR are strictly enforcing that
    the expressions are referencing the immediate parent only. So we need to
    track fields upwards the hierarchy to replace `t.a` with `t1.a` and `t2.a`
    in the example above. This is called dereferencing.

    Whether we can treat or not a field of a relation semantically equivalent
    with a field of an earlier relation in the hierarchy depends on the
    `.values` mapping of the relation. Leaf relations, like `t` in the example
    above, have an empty `.values` mapping, so we cannot dereference fields
    from them. On the other hand a projection, like `t1` in the example above,
    has a `.values` mapping like `{'a': t.a, 'b': t.b}`, so we can deduce that
    `t1.a` is semantically equivalent with `t.a` and so on.
    """

    """The relations we want the values to point to."""
    rels: VarTuple[ops.Relation]

    """Substitution mapping from values of earlier relations to the fields of `rels`."""
    subs: FrozenDict[ops.Value, ops.Field]

    """Ambiguous field references."""
    ambigs: FrozenDict[ops.Value, VarTuple[ops.Value]]

    @classmethod
    def from_targets(cls, rels, extra=None):
        """Create a dereference map from a list of target relations.

        Usually a single relation is passed except for joins where multiple
        relations are involved.

        Parameters
        ----------
        rels : list of ops.Relation
            The target relations to dereference to.
        extra : dict, optional
            Extra substitutions to be added to the dereference map.

        Returns
        -------
        DerefMap
        """
        rels = promote_list(rels)
        mapping = defaultdict(dict)
        for rel in rels:
            for field in rel.fields.values():
                for value, distance in cls.backtrack(field):
                    mapping[value][field] = distance

        subs, ambigs = {}, {}
        for from_, to in mapping.items():
            mindist = min(to.values())
            minkeys = [k for k, v in to.items() if v == mindist]
            # if all the closest fields are from the same relation, then we
            # can safely substitute them and we pick the first one arbitrarily
            if all(minkeys[0].relations == k.relations for k in minkeys):
                subs[from_] = minkeys[0]
            else:
                ambigs[from_] = minkeys

        if extra is not None:
            subs.update(extra)

        return cls(rels, subs, ambigs)

    @classmethod
    def backtrack(cls, value):
        """Backtrack the field in the relation hierarchy.

        The field is traced back until no modification is made, so only follow
        ops.Field nodes not arbitrary values.

        Parameters
        ----------
        value : ops.Value
            The value to backtrack.

        Yields
        ------
        tuple[ops.Field, int]
            The value node and the distance from the original value.
        """
        distance = 0
        # track down the field in the hierarchy until no modification
        # is made so only follow ops.Field nodes not arbitrary values;
        while isinstance(value, ops.Field):
            yield value, distance
            value = value.rel.values.get(value.name)
            distance += 1
        if value is not None and not value.find(ops.Impure, filter=ops.Value):
            yield value, distance

    def dereference(self, value):
        """Dereference a value to the target relations.

        Also check for ambiguous field references. If a field reference is found
        which is marked as ambiguous, then raise an error.

        Parameters
        ----------
        value : ops.Value
            The value to dereference.

        Returns
        -------
        ops.Value
            The dereferenced value.
        """
        ambigs = value.find(lambda x: x in self.ambigs, filter=ops.Value)
        if ambigs:
            raise IbisInputError(
                f"Ambiguous field reference {ambigs!r} in expression {value!r}"
            )
        return value.replace(self.subs, filter=ops.Value)


@replace(p.Field(p.JoinChain))
def peel_join_field(_):
    return _.rel.values[_.name]


@replace(p.ScalarParameter)
def replace_parameter(_, params, **kwargs):
    """Replace scalar parameters with their values."""
    return ops.Literal(value=params[_], dtype=_.dtype)


@replace(p.FillNa)
def rewrite_fillna(_):
    """Rewrite FillNa expressions to use more common operations."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.parent.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.parent

    selections = []
    for name in _.parent.schema.names:
        col = ops.Field(_.parent, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Alias(ops.Coalesce((col, value)), name)
        selections.append(col)

    return ops.Project(_.parent, selections)


@replace(p.DropNa)
def rewrite_dropna(_):
    """Rewrite DropNa expressions to use more common operations."""
    if _.subset is None:
        columns = [ops.Field(_.parent, name) for name in _.parent.schema.names]
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
        return _.parent

    return ops.Filter(_.parent, tuple(preds))


@replace(p.StringSlice)
def rewrite_stringslice(_, **kwargs):
    """Rewrite StringSlice in terms of Substring."""
    if _.end is None:
        return ops.Substring(_.arg, start=_.start)
    if _.start is None:
        return ops.Substring(_.arg, start=0, length=_.end)
    if (
        isinstance(_.start, ops.Literal)
        and isinstance(_.start.value, int)
        and isinstance(_.end, ops.Literal)
        and isinstance(_.end.value, int)
    ):
        # optimization for constant values
        length = _.end.value - _.start.value
    else:
        length = ops.Subtract(_.end, _.start)
    return ops.Substring(_.arg, start=_.start, length=length)


@replace(p.Analytic)
def project_wrap_analytic(_, rel):
    # Wrap analytic functions in a window function
    return ops.WindowFunction(_)


@replace(p.Reduction)
def project_wrap_reduction(_, rel):
    # Query all the tables that the reduction depends on
    if _.relations == {rel}:
        # The reduction is fully originating from the `rel`, so turn
        # it into a window function of `rel`
        return ops.WindowFunction(_)
    else:
        # 1. The reduction doesn't depend on any table, constructed from
        #    scalar values, so turn it into a scalar subquery.
        # 2. The reduction is originating from `rel` and other tables,
        #    so this is a correlated scalar subquery.
        # 3. The reduction is originating entirely from other tables,
        #    so this is an uncorrelated scalar subquery.
        return ops.ScalarSubquery(_.to_expr().as_table())


def rewrite_project_input(value, relation):
    # we need to detect reductions which are either turned into window functions
    # or scalar subqueries depending on whether they are originating from the
    # relation
    return value.replace(
        project_wrap_analytic | project_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context={"rel": relation},
    )


ReductionLike = p.Reduction | p.Field(p.Aggregate(groups={}))


@replace(ReductionLike)
def filter_wrap_reduction(_):
    # Wrap reductions or fields referencing an aggregation without a group by -
    # which are scalar fields - in a scalar subquery. In the latter case we
    # use the reduction value from the aggregation.
    if isinstance(_, ops.Field):
        value = _.rel.values[_.name]
    else:
        value = _
    return ops.ScalarSubquery(value.to_expr().as_table())


def rewrite_filter_input(value):
    return value.replace(filter_wrap_reduction, filter=p.Value & ~p.WindowFunction)


@replace(p.Analytic | p.Reduction)
def window_wrap_reduction(_, window):
    # Wrap analytic and reduction functions in a window function. Used in the
    # value.over() API.
    return ops.WindowFunction(
        _,
        how=window.how,
        start=window.start,
        end=window.end,
        group_by=window.groupings,
        order_by=window.orderings,
    )


@replace(p.WindowFunction)
def window_merge_frames(_, window):
    # Merge window frames, used in the value.over() and groupby.select() APIs.
    if _.how != window.how:
        raise ExpressionError(
            f"Unable to merge {_.how} window with {window.how} window"
        )
    elif _.start and window.start and _.start != window.start:
        raise ExpressionError(
            "Unable to merge windows with conflicting `start` boundary"
        )
    elif _.end and window.end and _.end != window.end:
        raise ExpressionError("Unable to merge windows with conflicting `end` boundary")

    start = _.start or window.start
    end = _.end or window.end
    group_by = tuple(toolz.unique(_.group_by + window.groupings))

    order_keys = {}
    for sort_key in window.orderings + _.order_by:
        order_keys[sort_key.expr] = sort_key.ascending
    order_by = (ops.SortKey(k, v) for k, v in order_keys.items())

    return _.copy(start=start, end=end, group_by=group_by, order_by=order_by)


def rewrite_window_input(value, window):
    context = {"window": window}
    # if self is a reduction or analytic function, wrap it in a window function
    node = value.replace(
        window_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context=context,
    )
    # if self is already a window function, merge the existing window frame
    # with the requested window frame
    return node.replace(window_merge_frames, filter=p.Value, context=context)


@replace(p.Bucket)
def replace_bucket(_):
    cases = []
    results = []

    if _.closed == "left":
        l_cmp = ops.LessEqual
        r_cmp = ops.Less
    else:
        l_cmp = ops.Less
        r_cmp = ops.LessEqual

    user_num_buckets = len(_.buckets) - 1

    bucket_id = 0
    if _.include_under:
        if user_num_buckets > 0:
            cmp = ops.Less if _.close_extreme else r_cmp
        else:
            cmp = ops.LessEqual if _.closed == "right" else ops.Less
        cases.append(cmp(_.arg, _.buckets[0]))
        results.append(bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(_.buckets, _.buckets[1:])):
        if _.close_extreme and (
            (_.closed == "right" and j == 0)
            or (_.closed == "left" and j == (user_num_buckets - 1))
        ):
            cases.append(
                ops.And(ops.LessEqual(lower, _.arg), ops.LessEqual(_.arg, upper))
            )
            results.append(bucket_id)
        else:
            cases.append(ops.And(l_cmp(lower, _.arg), r_cmp(_.arg, upper)))
            results.append(bucket_id)
        bucket_id += 1

    if _.include_over:
        if user_num_buckets > 0:
            cmp = ops.Less if _.close_extreme else l_cmp
        else:
            cmp = ops.Less if _.closed == "right" else ops.LessEqual

        cases.append(cmp(_.buckets[-1], _.arg))
        results.append(bucket_id)
        bucket_id += 1

    return ops.SearchedCase(
        cases=tuple(cases), results=tuple(results), default=ops.NULL
    )


# TODO(kszucs): schema comparison should be updated to not distinguish between
# different column order
@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != ops.Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)
    return ops.Filter(y.parent, y.predicates + preds)


@replace(p.Filter(y @ p.Project))
def reorder_filter_project(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)

    inner = ops.Filter(y.parent, preds)
    rule = p.Field(y.parent, name) >> d.Field(inner, name)
    projs = {k: v.replace(rule, filter=ops.Value) for k, v in y.values.items()}

    return ops.Project(inner, projs)


def simplify(node):
    # TODO(kszucs): add a utility to the graph module to do rewrites in multiple
    # passes after each other
    node = node.replace(reorder_filter_project)
    node = node.replace(reorder_filter_project)
    node = node.replace(subsequent_projects | subsequent_filters)
    node = node.replace(complete_reprojection)
    return node
