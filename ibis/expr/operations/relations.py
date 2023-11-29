from __future__ import annotations

import typing
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.deferred import deferred
from ibis.common.exceptions import IbisTypeError, IntegrityError, RelationError
from ibis.common.grounds import Concrete
from ibis.common.patterns import Between, InstanceOf, pattern
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations.core import Alias, Column, Node, Scalar, Value
from ibis.expr.operations.sortkeys import SortKey  # noqa: TCH001
from ibis.expr.schema import Schema
from ibis.util import Namespace, PseudoHashable, gen_name, indent

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

p = Namespace(pattern, module=__name__)
d = Namespace(deferred, module=__name__)


class TableProxy(Concrete):
    data: PseudoHashable

    def __repr__(self) -> str:
        data_repr = indent(repr(self.data.obj), spaces=2)
        return f"{self.__class__.__name__}:\n{data_repr}"

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:  # pragma: no cover
        """Convert this input to a pandas DataFrame."""

    @abstractmethod
    def to_pyarrow(self, schema: Schema) -> pa.Table:  # pragma: no cover
        """Convert this input to a PyArrow Table."""

    def to_pyarrow_bytes(self, schema: Schema) -> bytes:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        data = self.to_pyarrow(schema=schema)
        out = pa.BufferOutputStream()
        with pa.RecordBatchFileWriter(out, data.schema) as writer:
            writer.write(data)
        return out.getvalue()


class PyArrowTableProxy(TableProxy):
    def to_frame(self):
        return self.data.obj.to_pandas()

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        return self.data.obj


class PandasDataFrameProxy(TableProxy):
    def to_frame(self) -> pd.DataFrame:
        return self.data.obj

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        from ibis.formats.pyarrow import PyArrowSchema

        return pa.Table.from_pandas(
            self.data.obj, schema=PyArrowSchema.from_ibis(schema)
        )


@public
class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        from ibis.expr.types import TableExpr

        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def fields(self):
        # used for dereferencing fields/values to the lowest relation in the
        # chain this should return references to parent relation's fields
        ...

    @property
    @abstractmethod
    def schema(self):
        ...

    def to_expr(self):
        from ibis.expr.types import TableExpr

        return TableExpr(self)

    # cardinality property marking if the relation has a single row like SELECT COUNT(*)


# just to reduce boilerplate
class SimpleRelation(Relation):
    parent: Relation

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    def __init__(self, rel, name):
        if name not in rel.schema:
            columns_formatted = ", ".join(map(repr, rel.schema.names))
            # TODO(kszucs): should raise a more specific error type
            raise IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )
        super().__init__(rel=rel, name=name)

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]

    @attribute
    def relations(self):
        return frozenset({self.rel})


@public
class Subquery(Value):
    rel: Relation
    shape = ds.columnar

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.rel.schema) != 1:
            raise IntegrityError(
                f"Subquery must have exactly one column, got {len(self.rel.schema)}"
            )

    @attribute
    def value(self):
        name = self.rel.schema.names[0]
        return self.rel.fields[name]

    @attribute
    def relations(self):
        return frozenset()


@public
class ScalarSubquery(Subquery):
    def __init__(self, rel):
        from ibis.expr.rewrites import ReductionValue

        super().__init__(rel=rel)
        if not self.value.find(ReductionValue, filter=Value):
            raise IntegrityError(
                f"Subquery {self.value!r} is not scalar, it must be turned into a scalar subquery first"
            )

    @attribute
    def dtype(self):
        return self.value.dtype


@public
class ExistsSubquery(Subquery):
    dtype = dt.boolean


@public
class InSubquery(Subquery):
    needle: Value
    dtype = dt.boolean

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not rlz.comparable(self.value, self.needle):
            raise IntegrityError(
                f"Subquery {self.needle!r} is not comparable to {self.value!r}"
            )

    @attribute
    def relations(self):
        return self.needle.relations


# TODO: implement these
# AnySubquery
# AllSubquery


def _check_integrity(values, allowed_parents):
    for value in values:
        for rel in value.relations:
            if rel not in allowed_parents:
                raise IntegrityError(
                    f"Cannot add {value!r} to projection, they belong to another relation"
                )


def _check_filter_integrity(predicates, allowed_parent):
    from ibis.expr.rewrites import ReductionValue

    for pred in predicates:
        if pred.find(ReductionValue, filter=Value):
            raise IntegrityError(f"Cannot add {pred!r} to filter, it is a reduction")

        if pred.relations and allowed_parent not in pred.relations:
            raise IntegrityError(
                f"Cannot add {pred!r} to filter, they belong to another relation"
            )


@public
class Project(Relation):
    parent: Relation
    # TODO(kszucs): rename values to fields
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), {parent})
        super().__init__(parent=parent, values=values)

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class JoinLink(Node):
    how: Literal[
        "inner",
        "left",
        "right",
        "outer",
        "asof",
        "semi",
        "anti",
        "any_inner",
        "any_left",
        "cross",
    ]
    table: Relation
    predicates: VarTuple[Value[dt.Boolean]]


@public
class JoinChain(Relation):
    first: Relation
    rest: VarTuple[JoinLink]
    fields: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, first, rest, fields):
        allowed_parents = {first}
        for join in rest:
            allowed_parents.add(join.table)
            _check_integrity(join.predicates, allowed_parents)
        _check_integrity(fields.values(), allowed_parents)
        super().__init__(first=first, rest=rest, fields=fields)

    @attribute
    def schema(self):
        return Schema({k: v.dtype.copy(nullable=True) for k, v in self.fields.items()})

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.JoinExpr(self)


@public
class Sort(SimpleRelation):
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, {parent})
        super().__init__(parent=parent, keys=keys)


@public
class Filter(SimpleRelation):
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        # TODO(kszucs): use toolz.unique(predicates) to remove duplicates
        _check_filter_integrity(predicates, parent)
        super().__init__(parent=parent, predicates=predicates)


@public
class Limit(SimpleRelation):
    n: typing.Union[int, Scalar[dt.Integer], None] = None
    offset: typing.Union[int, Scalar[dt.Integer]] = 0


@public
class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
    metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

    def __init__(self, parent, groups, metrics):
        _check_integrity(groups.values(), {parent})
        _check_integrity(metrics.values(), {parent})
        super().__init__(parent=parent, groups=groups, metrics=metrics)

    @attribute
    def fields(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        # schema is consisting both by and metrics, use .from_tuples() to disallow
        # duplicate names in the schema
        groups = {k: v.dtype for k, v in self.groups.items()}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


@public
class Set(Relation):
    # turn it into a variadic relation
    left: Relation
    right: Relation
    distinct: bool = False

    def __init__(self, left, right, **kwargs):
        # convert to dictionary first, to get key-unordered comparison
        # semantics
        if dict(left.schema) != dict(right.schema):
            raise RelationError("Table schemas must be equal for set operations")
        elif left.schema.names != right.schema.names:
            # rewrite so that both sides have the columns in the same order making it
            # easier for the backends to implement set operations
            cols = {name: Field(right, name) for name in left.schema.names}
            right = Project(right, cols)
        super().__init__(left=left, right=right, **kwargs)

    @attribute
    def fields(self):
        return FrozenDict()

    @attribute
    def schema(self):
        return self.left.schema


@public
class Union(Set):
    pass


@public
class Intersection(Set):
    pass


@public
class Difference(Set):
    pass


# TODO(kszucs): call it Source or Table?
@public
class PhysicalTable(Relation):
    name: str
    fields = FrozenDict()


@public
class UnboundTable(PhysicalTable):
    schema: Schema


@public
class Namespace(Concrete):
    database: Optional[str] = None
    schema: Optional[str] = None


@public
class DatabaseTable(PhysicalTable):
    schema: Schema
    source: Any
    namespace: Namespace = Namespace()


@public
class InMemoryTable(PhysicalTable):
    schema: Schema
    data: TableProxy


@public
class SQLQueryResult(Relation):
    """A table sourced from the result set of a select query."""

    query: str
    schema: Schema
    source: Any
    fields = FrozenDict()


@public
class SQLStringView(PhysicalTable):
    """A view created from a SQL string."""

    child: Relation
    query: str

    @attribute
    def schema(self):
        # TODO(kszucs): avoid converting to expression
        backend = self.child.to_expr()._find_backend()
        return backend._get_schema_using_query(self.query)


@public
class View(PhysicalTable):
    """A view created from an expression."""

    child: Relation
    name: str

    @attribute
    def schema(self):
        return self.child.schema


@public
class DummyTable(Relation):
    # TODO(kszucs): verify that it has at least one element: Length(at_least=1)
    values: FrozenDict[str, Value]

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class SelfReference(SimpleRelation):
    @attribute
    def name(self) -> str:
        if (name := getattr(self.parent, "name", None)) is not None:
            return f"{name}_ref"
        return gen_name("self_ref")

    @attribute
    def fields(self):
        return FrozenDict()


@public
class FillNa(SimpleRelation):
    """Fill null values in the table."""

    replacements: typing.Union[Value[dt.Numeric | dt.String], FrozenDict[str, Any]]


@public
class DropNa(SimpleRelation):
    """Drop null values in the table."""

    how: typing.Literal["any", "all"]
    subset: Optional[VarTuple[Column]] = None


@public
class Sample(SimpleRelation):
    """Sample performs random sampling of records in a table."""

    fraction: Annotated[float, Between(0, 1)]
    method: typing.Literal["row", "block"]
    seed: typing.Union[int, None] = None


@public
class Distinct(SimpleRelation):
    """Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """


@public
def table(name, schema):
    return UnboundTable(name, schema).to_expr()


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()


# SQL-like selection, not used internally
@public
class Selection(Relation):
    parent: Relation
    selections: FrozenDict[str, Value]
    predicates: VarTuple[Value[dt.Boolean]]
    sort_keys: VarTuple[SortKey]

    @attribute
    def fields(self):
        return self.selections

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.selections.items()})


# Not a relation on its own
@public
class GroupBy(Relation):
    parent: Relation
    groups: VarTuple[str]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        return self.parent.schema


@public
class GroupByMetrics(Relation):
    parent: GroupBy
    metrics: FrozenDict[str, Scalar]

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        groups = {k: self.parent.schema[k] for k in self.parent.groups}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


@public
class PandasProject(Relation):
    parent: Relation
    values: FrozenDict[str, Value]

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})
