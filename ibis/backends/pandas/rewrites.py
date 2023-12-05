from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple  # noqa: TCH001
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
    def fields(self):
        return FrozenDict(
            {to: ops.Field(self.parent, from_) for from_, to in self.mapping.items()}
        )

    @attribute
    def schema(self):
        return Schema(
            {self.mapping[name]: dtype for name, dtype in self.parent.schema.items()}
        )


@public
class PandasJoin(PandasRelation):
    left: ops.Relation
    right: ops.Relation
    left_on: VarTuple[ops.Value]
    right_on: VarTuple[ops.Value]
    how: str

    @attribute
    def fields(self):
        return FrozenDict({**self.left.fields, **self.right.fields})

    @attribute
    def schema(self):
        return self.left.schema | self.right.schema


@public
class PandasReduce(PandasRelation):
    parent: ops.Relation
    metrics: FrozenDict[str, ops.Scalar]

    @attribute
    def fields(self):
        return self.metrics

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


@public
class PandasAggregate(PandasRelation):
    parent: ops.Relation
    groups: FrozenDict[str, ops.Field]
    metrics: FrozenDict[str, ops.Reduction]

    @attribute
    def fields(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


def is_columnar(node):
    return isinstance(node, ops.Value) and node.shape.is_columnar()


@replace(ops.Project)
def rewrite_project(_):
    winfuncs = []
    for v in _.values.values():
        winfuncs.extend(v.find(ops.WindowFunction, ops.Value))

    if not winfuncs:
        return _

    selects = {ops.Field(_.parent, k): k for k in _.parent.schema}
    for node in winfuncs:
        for arg in node.func.__args__:
            if is_columnar(arg) and arg not in selects:
                selects[arg] = gen_name("metric")
        for arg in node.frame.__children__:
            if is_columnar(arg) and arg not in selects:
                if isinstance(arg, ops.Relation):
                    raise TypeError(arg)
                selects[arg] = gen_name("frame")

    # STEP 1: construct the pre-projection
    proj = ops.Project(_.parent, {v: k for k, v in selects.items()})
    subs = {node: ops.Field(proj, name) for name, node in proj.fields.items()}

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
    return ops.Project(proj, values)


@replace(ops.Aggregate)
def rewrite_aggregate(_):
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
    subs = {node: ops.Field(proj, name) for name, node in proj.fields.items()}
    groups = {name: ops.Field(proj, selects[node]) for name, node in _.groups.items()}
    metrics = {name: node.replace(subs) for node, name in reductions.items()}
    if groups:
        agg = PandasAggregate(proj, groups, metrics)
    else:
        agg = PandasReduce(proj, metrics)

    # STEP 3: construct the post-projection
    subs = {node: ops.Field(agg, name) for node, name in reductions.items()}
    values = {name: ops.Field(agg, selects[node]) for name, node in _.groups.items()}
    values.update({name: node.replace(subs) for name, node in _.metrics.items()})
    return ops.Project(agg, values)


def split_join_predicates(left, right, predicates):
    left_on = []
    right_on = []
    for pred in predicates:
        if left not in pred.relations or right not in pred.relations:
            # not a usual join predicate, so apply a trick by placing the
            # predicate to the left side and adding a literal True to the right
            # which the left side must be equal to
            left_on.append(pred)
            right_on.append(ops.Literal(True, dtype=dt.boolean))
        elif isinstance(pred, ops.Equals):
            if left in pred.left.relations and right in pred.right.relations:
                left_on.append(pred.left)
                right_on.append(pred.right)
            elif left in pred.right.relations and right in pred.left.relations:
                left_on.append(pred.right)
                right_on.append(pred.left)
            else:
                raise ValueError("Join predicate does not reference both tables")
        else:
            raise TypeError("Only equality join predicates supported with pandas")

    return left_on, right_on


@replace(ops.JoinChain)
def rewrite_join(_):
    prefixes = {}
    prefixes[_.first] = prefix = str(len(prefixes))
    left = PandasRename.from_prefix(_.first, prefix)

    for link in _.rest:
        prefixes[link.table] = prefix = str(len(prefixes))
        right = PandasRename.from_prefix(link.table, prefix)

        subs = {v: ops.Field(left, k) for k, v in left.fields.items()}
        subs.update({v: ops.Field(right, k) for k, v in right.fields.items()})
        preds = [pred.replace(subs, filter=ops.Value) for pred in link.predicates]

        # need to replace the fields in the predicates
        left_on, right_on = split_join_predicates(left, right, preds)

        left = PandasJoin(
            how=link.how,
            left=left,
            right=right,
            left_on=left_on,
            right_on=right_on,
        )

    subs = {v: ops.Field(left, k) for k, v in left.fields.items()}
    fields = {k: v.replace(subs, filter=ops.Value) for k, v in _.fields.items()}
    return ops.Project(left, fields)
