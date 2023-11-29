from __future__ import annotations

from public import public

import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple
from ibis.expr.schema import Schema
from ibis.util import gen_name


# Not a relation on its own
@public
class GroupBy(ops.Relation):
    parent: ops.Relation
    groups: VarTuple[str]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        return self.parent.schema


@public
class GroupByMetrics(ops.Relation):
    parent: GroupBy
    metrics: FrozenDict[str, ops.Scalar]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        groups = {k: self.parent.schema[k] for k in self.parent.groups}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


@public
class PandasProject(ops.Relation):
    parent: ops.Relation
    values: FrozenDict[str, ops.Value]

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


class PandasAggregate(ops.Relation):
    parent: ops.Relation
    groups: VarTuple[str]
    metrics: FrozenDict[str, ops.Reduction]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        groups = {k: self.parent.schema[k] for k in self.groups}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


def flip(d):
    return {v: k for k, v in d.items()}


@replace(ops.Aggregate)
def aggregate_to_groupby(_):
    if not _.groups:
        return ops.PandasProject(_.parent, _.metrics)

    metrics = {}
    reductions = {}
    projections = {ops.Field(_.parent, k): k for k in _.parent.schema}

    # add all the computed groups to the pre-projection
    for _k, v in _.groups.items():
        if v not in projections:
            projections[v] = gen_name("agg")

    # add all the computed dependencies of any reduction to the pre-projection
    # add all the reductions to the actual pandas aggregation
    for _k, v in _.metrics.items():
        for reduction in v.find_topmost(ops.Reduction):
            for child in reduction.__children__:
                if child not in projections:
                    projections[child] = gen_name("agg")
            reductions[reduction] = gen_name("agg")

    # construct the pre-projection
    step1 = ops.Project(_.parent, flip(projections))

    # construct the pandas aggregation
    subs = {node: ops.Field(step1, name) for node, name in projections.items()}
    groups = [projections[node] for node in _.groups.values()]
    metrics = {name: node.replace(subs) for node, name in reductions.items()}
    step2 = PandasAggregate(step1, groups, metrics)

    # construct the post-projection
    subs = {node: ops.Field(step2, name) for node, name in reductions.items()}
    group_values = {name: ops.Field(step2, projections[node]) for name, node in _.groups.items()}
    metric_values = {name: node.replace(subs) for name, node in _.metrics.items()}
    step3 = ops.Project(step2, {**group_values, **metric_values})

    return step3
