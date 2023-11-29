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


@replace(ops.Aggregate)
def aggregate_to_groupby(_):
    if not _.groups:
        return ops.PandasProject(_.parent, _.metrics)

    values = {}

    for v in _.groups.values():
        if not isinstance(v, ops.Field):
            values[v] = gen_name("agg")

    for v in _.metrics.values():
        for red in v.find_topmost(ops.Reduction):
            for arg in red.args:
                if isinstance(arg, ops.Value) and not isinstance(arg, ops.Field):
                    values[arg] = gen_name("agg")

    fields = {k: ops.Field(_.parent, k) for k in _.parent.schema}
    fields.update({v: k for k, v in values.items()})
    proj = ops.Project(_.parent, fields)

    mapping = {v: k for k, v in proj.fields.items()}
    groups = [v.replace(mapping, filter=ops.Value) for k, v in _.groups.items()]
    groupby = GroupBy(proj, groups)

    # turn these into a different type, e.g. LazyField to not compute it
    mapping = {v: ops.Field(groupby, k) for k, v in proj.fields.items()}
    metrics = {k: v.replace(mapping) for k, v in _.metrics.items()}
    return GroupByMetrics(groupby, metrics)
