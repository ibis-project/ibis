from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Deferred
from ibis.common.patterns import InstanceOf  # noqa: TCH001
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Alias, Column, Node, Scalar, SortKey, Value
from ibis.expr.schema import Schema
from ibis.expr.types import Expr, literal
from ibis.expr.types import Value as ValueExpr
from ibis.selectors import Selector

# need a parallel Expression and Operation class hierarchy to decompose ops.Selection
# into proper relational algebra operations


class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def schema(self):
        ...

    def to_expr(self):
        return TableExpr(self)


class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        # TODO(kszucs): validate that each field originates from the parent relation
        super().__init__(parent=parent, values=values)

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


class Join(Relation):
    left: Relation
    right: Relation
    fields: FrozenDict[str, Field]
    predicates: VarTuple[Value[dt.Boolean]]

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    @attribute
    def schema(self):
        return self.parent.schema


class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    @attribute
    def schema(self):
        return self.parent.schema


class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
    metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

    @attribute
    def schema(self):
        # schema is consisting both by and metrics
        return Schema.from_tuples([*self.groups.items(), *self.metrics.items()])


class UnboundTable(Relation):
    name: str
    schema: Schema


class TableExpr(Expr):
    def schema(self):
        return self.op().schema

    def __getattr__(self, key):
        return Field(self, key).to_expr()

    def select(self, *args, **kwargs):
        values = bind(self, (args, kwargs))
        values = unwrap_alias(values)
        # TODO(kszucs): windowization of redictions should happen here
        return Project(self, values).to_expr()

    def where(self, *predicates):
        predicates = bind(self, predicates)
        return Filter(self, predicates).to_expr()

    def order_by(self, *keys):
        keys = bind(self, keys)
        return Sort(self, keys).to_expr()

    def aggregate(self, groups, metrics):
        groups = bind(self, groups)
        metrics = bind(self, metrics)
        groups = unwrap_alias(groups)
        metrics = unwrap_alias(metrics)
        return Aggregate(self, groups, metrics).to_expr()


def bind(table: TableExpr, value: Any) -> ir.Value:
    if isinstance(value, ValueExpr):
        yield value
    elif isinstance(value, TableExpr):
        for name in value.schema().keys():
            yield Field(value, name).to_expr()
    elif isinstance(value, str):
        yield Field(table, value).to_expr()
    elif isinstance(value, Deferred):
        yield value.resolve(table)
    elif isinstance(value, Selector):
        yield from value.expand(table)
    elif isinstance(value, tuple):
        for v in value:
            yield from bind(table, v)
    elif isinstance(value, dict):
        for k, v in value.items():
            for val in bind(table, v):
                yield val.name(k)
    elif callable(value):
        yield value(table)
    else:
        yield literal(value)


def unwrap_alias(values):
    result = {}
    for value in values:
        node = value.op()
        if isinstance(node, Alias):
            result[node.name] = node.arg
        else:
            result[node.name] = node
    return result


# POSSIBLE REWRITES:
# 1. Reprojection of the whole relation: t.select(t) >> t
