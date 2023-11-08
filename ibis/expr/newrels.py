from __future__ import annotations

from abc import abstractmethod

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Node, SortKey, Value
from ibis.expr.schema import Schema  # noqa: TCH001
from ibis.expr.types import Expr

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
    values: FrozenDict[str, Value]

    def __init__(self, parent, fields):
        # validate that each field originates from the parent relation
        pass

    @attribute
    def schema(self):
        return FrozenDict({k: v.dtype for k, v in self.fields.items()})


class Join(Relation):
    left: Relation
    right: Relation
    fields: FrozenDict[str, Field]
    predicates: VarTuple[Value[dt.Boolean]]

    @attribute
    def schema(self):
        return FrozenDict({k: v.dtype for k, v in self.fields.items()})


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


class UnboundTable(Relation):
    name: str
    schema: Schema


class TableExpr(Expr):
    def __getattr__(self, key):
        return Field(self, key)
