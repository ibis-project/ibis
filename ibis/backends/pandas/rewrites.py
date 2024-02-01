from __future__ import annotations

from public import public

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import replace_parameter
from ibis.expr.schema import Schema
from ibis.util import gen_name


class PandasRelation(ops.Relation):
    pass


class PandasValue(ops.Value):
    pass


@public
class PandasRename(PandasRelation):
    parent: ops.Relation
    mapping: FrozenDict[str, str]

    @classmethod
    def from_prefix(cls, parent, prefix):
        mapping = {k: f"{prefix}_{k}" for k in parent.schema}
        return cls(parent, mapping)

    @attribute
    def values(self):
        return FrozenDict(
            {to: ops.Field(self.parent, from_) for from_, to in self.mapping.items()}
        )

    @attribute
    def schema(self):
        return Schema(
            {self.mapping[name]: dtype for name, dtype in self.parent.schema.items()}
        )


@public
class PandasResetIndex(PandasRelation):
    parent: ops.Relation

    @attribute
    def values(self):
        return self.parent.values

    @attribute
    def schema(self):
        return self.parent.schema


@public
class PandasJoin(PandasRelation):
    left: ops.Relation
    right: ops.Relation
    left_on: VarTuple[ops.Value]
    right_on: VarTuple[ops.Value]
    how: str

    @attribute
    def values(self):
        return FrozenDict({**self.left.values, **self.right.values})

    @attribute
    def schema(self):
        return self.left.schema | self.right.schema


@public
class PandasAsofJoin(PandasJoin):
    left_by: VarTuple[ops.Value]
    right_by: VarTuple[ops.Value]
    operator: type


@public
class PandasAggregate(PandasRelation):
    parent: ops.Relation
    groups: FrozenDict[str, ops.Field]
    metrics: FrozenDict[str, ops.Reduction]

    @attribute
    def values(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class PandasLimit(PandasRelation):
    parent: ops.Relation
    n: ops.Relation
    offset: ops.Relation

    @attribute
    def values(self):
        return self.parent.values

    @attribute
    def schema(self):
        return self.parent.schema


@public
class PandasScalarSubquery(PandasValue):
    # variant with no integrity checks
    rel: ops.Relation

    shape = ds.scalar

    @attribute
    def dtype(self):
        return self.rel.schema.types[0]


def is_columnar(node):
    return isinstance(node, ops.Value) and node.shape.is_columnar()


@replace(ops.Project)
def rewrite_project(_, **kwargs):
    unnests = []
    winfuncs = []
    for v in _.values.values():
        unnests.extend(v.find(ops.Unnest, filter=ops.Value))
        winfuncs.extend(v.find(ops.WindowFunction, filter=ops.Value))

    if not winfuncs:
        return PandasResetIndex(_) if unnests else _

    selects = {ops.Field(_.parent, k): k for k in _.parent.schema}
    for node in winfuncs:
        # add computed values from the window function
        values = list(node.func.__args__)
        # add computed values from the window frame
        values += node.frame.group_by
        values += [key.expr for key in node.frame.order_by]
        if node.frame.start is not None:
            values.append(node.frame.start.value)
        if node.frame.end is not None:
            values.append(node.frame.end.value)

        for v in values:
            if is_columnar(v) and v not in selects:
                selects[v] = gen_name("value")

    # STEP 1: construct the pre-projection
    proj = ops.Project(_.parent, {v: k for k, v in selects.items()})
    subs = {node: ops.Field(proj, name) for name, node in proj.values.items()}

    # STEP 2: construct new window function nodes
    metrics = {}
    for node in winfuncs:
        frame = node.frame
        start = None if frame.start is None else frame.start.replace(subs)
        end = None if frame.end is None else frame.end.replace(subs)
        order_by = [key.replace(subs) for key in frame.order_by]
        group_by = [key.replace(subs) for key in frame.group_by]
        frame = frame.__class__(
            proj, start=start, end=end, group_by=group_by, order_by=order_by
        )
        metrics[node] = ops.WindowFunction(node.func.replace(subs), frame)

    # STEP 3: reconstruct the current projection with the window functions
    subs.update(metrics)
    values = {k: v.replace(subs, filter=ops.Value) for k, v in _.values.items()}
    result = ops.Project(proj, values)

    return PandasResetIndex(result)


@replace(ops.Aggregate)
def rewrite_aggregate(_, **kwargs):
    selects = {ops.Field(_.parent, k): k for k in _.parent.schema}
    for v in _.groups.values():
        if v not in selects:
            selects[v] = gen_name("group")

    reductions = {}
    for v in _.metrics.values():
        for reduction in v.find_topmost(ops.Reduction):
            for arg in reduction.__args__:
                if is_columnar(arg) and arg not in selects:
                    selects[arg] = gen_name("value")
            if reduction not in reductions:
                reductions[reduction] = gen_name("reduction")

    # STEP 1: construct the pre-projection
    proj = ops.Project(_.parent, {v: k for k, v in selects.items()})

    # STEP 2: construct the pandas aggregation
    subs = {node: ops.Field(proj, name) for name, node in proj.values.items()}
    groups = {name: ops.Field(proj, selects[node]) for name, node in _.groups.items()}
    metrics = {name: node.replace(subs) for node, name in reductions.items()}
    agg = PandasAggregate(proj, groups, metrics)

    # STEP 3: construct the post-projection
    subs = {node: ops.Field(agg, name) for node, name in reductions.items()}
    values = {name: ops.Field(agg, name) for name, node in _.groups.items()}
    values.update({name: node.replace(subs) for name, node in _.metrics.items()})
    return ops.Project(agg, values)


def split_join_predicates(left, right, predicates, only_equality=True):
    left_on = []
    right_on = []
    for pred in predicates:
        if left not in pred.relations or right not in pred.relations:
            # not a usual join predicate, so apply a trick by placing the
            # predicate to the left side and adding a literal True to the right
            # which the left side must be equal to
            left_on.append(pred)
            right_on.append(ops.Literal(True, dtype=dt.boolean))
        elif isinstance(pred, ops.Binary):
            if only_equality and not isinstance(pred, ops.Equals):
                raise TypeError("Only equality join predicates supported with pandas")
            if left in pred.left.relations and right in pred.right.relations:
                left_on.append(pred.left)
                right_on.append(pred.right)
            elif left in pred.right.relations and right in pred.left.relations:
                left_on.append(pred.right)
                right_on.append(pred.left)
            else:
                raise ValueError("Join predicate does not reference both tables")
        else:
            raise TypeError(f"Unsupported join predicate {pred}")

    return left_on, right_on


@replace(ops.JoinChain)
def rewrite_join(_, **kwargs):
    # TODO(kszucs): JoinTable.index can be used as a prefix
    prefixes = {}
    prefixes[_.first] = prefix = str(len(prefixes))
    left = PandasRename.from_prefix(_.first, prefix)

    for link in _.rest:
        prefixes[link.table] = prefix = str(len(prefixes))
        right = PandasRename.from_prefix(link.table, prefix)

        subs = {v: ops.Field(left, k) for k, v in left.values.items()}
        subs.update({v: ops.Field(right, k) for k, v in right.values.items()})
        preds = [pred.replace(subs, filter=ops.Value) for pred in link.predicates]

        # separate ASOF from the rest of the joins
        if link.how == "asof":
            on, *by = preds
            left_on, right_on = split_join_predicates(
                left, right, [on], only_equality=False
            )
            left_by, right_by = split_join_predicates(left, right, by)
            left = PandasAsofJoin(
                how="asof",
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
                left_by=left_by,
                right_by=right_by,
                operator=type(on),
            )
        else:
            # need to replace the fields in the predicates
            left_on, right_on = split_join_predicates(left, right, preds)
            left = PandasJoin(
                how=link.how,
                left=left,
                right=right,
                left_on=left_on,
                right_on=right_on,
            )

    subs = {v: ops.Field(left, k) for k, v in left.values.items()}
    fields = {k: v.replace(subs, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(left, fields)


@replace(ops.Limit)
def rewrite_limit(_, **kwargs):
    if isinstance(_.n, ops.Value):
        n = _.n.to_expr()
    else:
        n = ibis.literal(_.n)

    if isinstance(_.offset, ops.Value):
        offset = _.offset.to_expr()
    else:
        offset = ibis.literal(_.offset)

    n = n.as_table().op()
    if isinstance(n, ops.Aggregate):
        n = rewrite_aggregate.match(n, context={})

    offset = offset.as_table().op()
    if isinstance(offset, ops.Aggregate):
        offset = rewrite_aggregate.match(offset, context={})

    return PandasLimit(_.parent, n, offset)


@replace(ops.ScalarSubquery)
def rewrite_scalar_subquery(_, **kwargs):
    return PandasScalarSubquery(_.rel)


@replace(ops.UnboundTable)
def bind_unbound_table(_, backend, **kwargs):
    return ops.DatabaseTable(name=_.name, schema=_.schema, source=backend)


def plan(node, backend, params):
    ctx = {"params": params, "backend": backend}
    node = node.replace(rewrite_scalar_subquery)
    node = node.replace(
        rewrite_project
        | rewrite_aggregate
        | rewrite_join
        | rewrite_limit
        | replace_parameter
        | bind_unbound_table,
        context=ctx,
    )
    return node
