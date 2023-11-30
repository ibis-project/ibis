from __future__ import annotations

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.grounds import Concrete
from ibis.common.patterns import replace
from ibis.common.typing import VarTuple
from ibis.expr.schema import Schema
from ibis.util import gen_name

# Not a relation on its own
# @public
# class GroupBy(ops.Relation):
#     parent: ops.Relation
#     groups: VarTuple[str]

#     @attribute
#     def fields(self):
#         return {}

#     @attribute
#     def schema(self):
#         return self.parent.schema


# @public
# class GroupByMetrics(ops.Relation):
#     parent: GroupBy
#     metrics: FrozenDict[str, ops.Scalar]

#     @attribute
#     def fields(self):
#         return {}

#     @attribute
#     def schema(self):
#         groups = {k: self.parent.schema[k] for k in self.parent.groups}
#         metrics = {k: v.dtype for k, v in self.metrics.items()}
#         return Schema.from_tuples([*groups.items(), *metrics.items()])


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


# TODO(kszucs): possibly not needed
@public
class ColumnRef(ops.Value):
    name: str
    dtype: dt.DataType
    shape = ds.columnar


@public
class PandasReduce(ops.Relation):
    parent: ops.Relation
    metrics: FrozenDict[str, ops.Scalar]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples(metrics.items())


@public
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
    # add all the computed groups to the pre-projection
    select_derefs = {ops.Field(_.parent, k): k for k in _.parent.schema}
    for _k, v in _.groups.items():
        if v not in select_derefs:
            select_derefs[v] = gen_name("agg")

    # add all the computed dependencies of any reduction to the pre-projection
    # add all the reductions to the actual pandas aggregation
    reduction_derefs = {}
    for _k, v in _.metrics.items():
        for reduction in v.find_topmost(ops.Reduction):
            for arg in reduction.__children__:
                if arg not in select_derefs:
                    select_derefs[arg] = gen_name("agg")
            if reduction not in reduction_derefs:
                reduction_derefs[reduction] = gen_name("agg")

    # STEP 1: construct the pre-projection
    proj = ops.Project(_.parent, flip(select_derefs))

    # STEP 2: construct the pandas aggregation
    subs = {node: ColumnRef(name, node.dtype) for name, node in proj.fields.items()}
    groups = [select_derefs[node] for node in _.groups.values()]
    metrics = {name: node.replace(subs) for node, name in reduction_derefs.items()}
    if groups:
        agg = PandasAggregate(proj, groups, metrics)
    else:
        agg = PandasReduce(proj, metrics)

    # STEP 3: construct the post-projection
    subs = {node: ops.Field(agg, name) for node, name in reduction_derefs.items()}
    group_values = {
        name: ops.Field(agg, select_derefs[node]) for name, node in _.groups.items()
    }
    metric_values = {name: node.replace(subs) for name, node in _.metrics.items()}
    return ops.Project(agg, {**group_values, **metric_values})
