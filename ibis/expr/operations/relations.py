from __future__ import annotations

import abc
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional
from typing import Union as UnionType

from public import public

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.common.annotations import annotated, attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Deferred
from ibis.common.grounds import Concrete, Immutable
from ibis.common.patterns import Between, Coercible, Eq
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Column, Named, Node, Scalar, Value
from ibis.expr.operations.sortkeys import SortKey  # noqa: TCH001
from ibis.expr.schema import Schema

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


_table_names = (f"unbound_table_{i:d}" for i in itertools.count())


@public
def genname():
    return next(_table_names)


@public
class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        import pandas as pd

        import ibis
        import ibis.expr.types as ir

        if isinstance(value, pd.DataFrame):
            return ibis.memtable(value).op()
        elif isinstance(value, ir.Expr):
            return value.op()
        else:
            return value

    def order_by(self, sort_exprs):
        return Selection(self, [], sort_keys=sort_exprs)

    @property
    @abstractmethod
    def schema(self) -> Schema:
        ...

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.Table(self)


TableNode = Relation


@public
class Namespace(Concrete):
    database: Optional[str] = None
    schema: Optional[str] = None


@public
class PhysicalTable(Relation, Named):
    pass


# TODO(kszucs): PhysicalTable should have a source attribute and UnbountTable
# should just extend TableNode
@public
class UnboundTable(PhysicalTable):
    schema: Schema
    name: Optional[str] = None
    namespace: Namespace = Namespace()

    def __init__(self, schema, name, namespace) -> None:
        if name is None:
            name = genname()
        super().__init__(schema=schema, name=name, namespace=namespace)


@public
class DatabaseTable(PhysicalTable):
    name: str
    schema: Schema
    source: Any
    namespace: Namespace = Namespace()


@public
class SQLQueryResult(TableNode):
    """A table sourced from the result set of a select query."""

    query: str
    schema: Schema
    source: Any


# TODO(kszucs): Add a pseudohashable wrapper and use that from InMemoryTable
# subclasses PandasTable, PyArrowTable


class TableProxy(Immutable):
    __slots__ = ("_data", "_hash")
    _data: Any
    _hash: int

    def __init__(self, data) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_hash", hash((type(data), id(data))))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        data_repr = util.indent(repr(self._data), spaces=2)
        return f"{self.__class__.__name__}:\n{data_repr}"

    @abc.abstractmethod
    def to_frame(self) -> pd.DataFrame:  # pragma: no cover
        """Convert this input to a pandas DataFrame."""

    @abc.abstractmethod
    def to_pyarrow(self, schema: Schema) -> pa.Table:  # pragma: no cover
        """Convert this input to a PyArrow Table."""

    def to_pyarrow_bytes(self, schema: Schema) -> bytes:
        import pyarrow as pa

        data = self.to_pyarrow(schema=schema)
        out = pa.BufferOutputStream()
        with pa.RecordBatchFileWriter(out, data.schema) as writer:
            writer.write(data)
        return out.getvalue()

    def __len__(self) -> int:
        return len(self._data)


class PyArrowTableProxy(TableProxy):
    __slots__ = ()

    def to_frame(self):
        return self._data.to_pandas()

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        return self._data


class PandasDataFrameProxy(TableProxy):
    __slots__ = ()

    def to_frame(self) -> pd.DataFrame:
        return self._data

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        import pyarrow as pa

        from ibis.formats.pyarrow import PyArrowSchema

        return pa.Table.from_pandas(self._data, schema=PyArrowSchema.from_ibis(schema))


@public
class InMemoryTable(PhysicalTable):
    name: str
    schema: Schema
    data: TableProxy


# TODO(kszucs): desperately need to clean this up, the majority of this
# functionality should be handled by input rules for the Join class
def _clean_join_predicates(left, right, predicates):
    import ibis.expr.analysis as an
    import ibis.expr.types as ir
    from ibis.expr.analysis import shares_all_roots

    result = []

    for pred in predicates:
        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise com.ExpressionError("Join key tuple must be length 2")
            lk, rk = pred
            lk = left.to_expr()._ensure_expr(lk)
            rk = right.to_expr()._ensure_expr(rk)
            pred = lk == rk
        elif isinstance(pred, str):
            pred = left.to_expr()[pred] == right.to_expr()[pred]
        elif pred is True or pred is False:
            pred = ops.Literal(pred, dtype="bool").to_expr()
        elif isinstance(pred, Value):
            pred = pred.to_expr()
        elif isinstance(pred, Deferred):
            # resolve deferred expressions on the left table
            pred = pred.resolve(left.to_expr())
        elif not isinstance(pred, ir.Expr):
            raise NotImplementedError

        if not isinstance(pred, ir.BooleanValue):
            raise com.ExpressionError("Join predicate must be a boolean expression")

        preds = an.flatten_predicate(pred.op())
        result.extend(preds)

    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    for predicate in result:
        if not shares_all_roots(predicate, [left, right]):
            raise com.RelationError(
                f"The expression {predicate!r} does not fully "
                "originate from dependencies of the table "
                "expression."
            )

    assert all(isinstance(pred, ops.Node) for pred in result)

    return tuple(result)


@public
class Join(Relation):
    left: Relation
    right: Relation
    predicates: Any = ()

    def __init__(self, left, right, predicates, **kwargs):
        # TODO(kszucs): predicates should be already a list of operations, need
        # to update the validation rule for the Join classes which is a noop
        # currently
        import ibis.expr.operations as ops
        import ibis.expr.types as ir

        # TODO(kszucs): need to factor this out to appropriate join predicate
        # rules
        predicates = [
            pred.op() if isinstance(pred, ir.Expr) else pred
            for pred in util.promote_list(predicates)
        ]

        if left.equals(right):
            # GH #667: If left and right table have a common parent expression,
            # e.g. they have different filters, we need to add a self-reference
            # and make the appropriate substitution in the join predicates
            right = ops.SelfReference(right)
        elif isinstance(right, Join):
            # for joins with joins on the right side we turn the right side
            # into a view, otherwise the join tree is incorrectly flattened
            # and tables on the right are incorrectly scoped
            old = right
            new = right = ops.SelfReference(right)
            rule = Eq(old) >> new
            predicates = [pred.replace(rule) for pred in predicates]

        predicates = _clean_join_predicates(left, right, predicates)

        super().__init__(left=left, right=right, predicates=predicates, **kwargs)

    @property
    def schema(self):
        # TODO(kszucs): use `return self.left.schema | self.right.schema` instead which
        # eliminates unnecessary projection over the join, but currently breaks the
        # pandas backend
        left, right = self.left.schema, self.right.schema
        if duplicates := left.keys() & right.keys():
            raise com.IntegrityError(f"Duplicate column name(s): {duplicates}")
        return Schema({**left, **right})


@public
class InnerJoin(Join):
    pass


@public
class LeftJoin(Join):
    pass


@public
class RightJoin(Join):
    pass


@public
class OuterJoin(Join):
    pass


@public
class AnyInnerJoin(Join):
    pass


@public
class AnyLeftJoin(Join):
    pass


@public
class LeftSemiJoin(Join):
    @attribute
    def schema(self):
        return self.left.schema


@public
class LeftAntiJoin(Join):
    @attribute
    def schema(self):
        return self.left.schema


@public
class CrossJoin(Join):
    pass


@public
class AsOfJoin(Join):
    # TODO(kszucs): convert to proper predicate rules
    by: Any = ()
    tolerance: Optional[Value[dt.Interval]] = None

    def __init__(self, left, right, by, predicates, **kwargs):
        by = _clean_join_predicates(left, right, util.promote_list(by))
        super().__init__(left=left, right=right, by=by, predicates=predicates, **kwargs)


@public
class SetOp(Relation):
    left: Relation
    right: Relation
    distinct: bool = False

    def __init__(self, left, right, **kwargs):
        # convert to dictionary first, to get key-unordered comparison
        # semantics
        if dict(left.schema) != dict(right.schema):
            raise com.RelationError("Table schemas must be equal for set operations")
        elif left.schema.names != right.schema.names:
            # rewrite so that both sides have the columns in the same order making it
            # easier for the backends to implement set operations
            cols = [ops.TableColumn(right, name) for name in left.schema.names]
            right = Selection(right, cols)
        super().__init__(left=left, right=right, **kwargs)

    @attribute
    def schema(self):
        return self.left.schema


@public
class Union(SetOp):
    pass


@public
class Intersection(SetOp):
    pass


@public
class Difference(SetOp):
    pass


@public
class Limit(Relation):
    table: Relation
    n: UnionType[int, Scalar[dt.Integer], None] = None
    offset: UnionType[int, Scalar[dt.Integer]] = 0

    @attribute
    def schema(self):
        return self.table.schema


@public
class SelfReference(Relation):
    table: Relation

    @attribute
    def name(self) -> str:
        if (name := getattr(self.table, "name", None)) is not None:
            return f"{name}_ref"
        return util.gen_name("self_ref")

    @attribute
    def schema(self):
        return self.table.schema


class Projection(Relation):
    table: Relation
    selections: VarTuple[Relation | Value]

    @attribute
    def schema(self):
        # Resolve schema and initialize
        if not self.selections:
            return self.table.schema

        types, names = [], []
        for projection in self.selections:
            if isinstance(projection, Value):
                names.append(projection.name)
                types.append(projection.dtype)
            elif isinstance(projection, TableNode):
                schema = projection.schema
                names.extend(schema.names)
                types.extend(schema.types)

        return Schema.from_tuples(zip(names, types))


def _add_alias(op: ops.Value | ops.TableNode):
    """Add a name to a projected column if necessary."""
    if isinstance(op, ops.Value) and not isinstance(op, (ops.Alias, ops.TableColumn)):
        return ops.Alias(op, op.name)
    else:
        return op


@public
class Selection(Projection):
    predicates: VarTuple[Value[dt.Boolean]] = ()
    sort_keys: VarTuple[SortKey] = ()

    def __init__(self, table, selections, predicates, sort_keys, **kwargs):
        from ibis.expr.analysis import shares_all_roots, shares_some_roots

        if not shares_all_roots(selections + sort_keys, table):
            raise com.RelationError(
                "Selection expressions don't fully originate from "
                "dependencies of the table expression."
            )

        for predicate in predicates:
            if isinstance(predicate, ops.Literal):
                if not (dtype := predicate.dtype).is_boolean():
                    raise com.IbisTypeError(f"Invalid predicate dtype: {dtype}")
            elif not shares_some_roots(predicate, table):
                raise com.RelationError("Predicate doesn't share any roots with table")

        super().__init__(
            table=table,
            selections=tuple(map(_add_alias, selections)),
            predicates=predicates,
            sort_keys=sort_keys,
            **kwargs,
        )

    @annotated
    def order_by(self, keys: VarTuple[SortKey]):
        from ibis.expr.analysis import shares_all_roots, sub_immediate_parents

        if not self.selections:
            if shares_all_roots(keys, table := self.table):
                sort_keys = tuple(self.sort_keys) + tuple(
                    sub_immediate_parents(key, table) for key in keys
                )

                return Selection(
                    table,
                    self.selections,
                    predicates=self.predicates,
                    sort_keys=sort_keys,
                )

        return Selection(self, [], sort_keys=keys)

    @attribute
    def _projection(self):
        return Projection(self.table, self.selections)


@public
class DummyTable(Relation):
    # TODO(kszucs): verify that it has at least one element: Length(at_least=1)
    values: VarTuple[Value[dt.Any, ds.Scalar]]

    @attribute
    def schema(self):
        return Schema({op.name: op.dtype for op in self.values})


@public
class Aggregation(Relation):
    table: Relation
    metrics: VarTuple[Scalar] = ()
    by: VarTuple[Column] = ()
    having: VarTuple[Scalar[dt.Boolean]] = ()
    predicates: VarTuple[Value[dt.Boolean]] = ()
    sort_keys: VarTuple[SortKey] = ()

    def __init__(self, table, metrics, by, having, predicates, sort_keys):
        from ibis.expr.analysis import shares_all_roots, shares_some_roots

        # All non-scalar refs originate from the input table
        if not shares_all_roots(metrics + by + having + sort_keys, table):
            raise com.RelationError(
                "Selection expressions don't fully originate from "
                "dependencies of the table expression."
            )

        # invariant due to Aggregation and AggregateSelection requiring a valid
        # Selection
        assert all(shares_some_roots(predicate, table) for predicate in predicates)

        if not by:
            sort_keys = tuple()

        super().__init__(
            table=table,
            metrics=tuple(map(_add_alias, metrics)),
            by=tuple(map(_add_alias, by)),
            having=having,
            predicates=predicates,
            sort_keys=sort_keys,
        )

    @attribute
    def _projection(self):
        return Projection(self.table, self.metrics + self.by)

    @attribute
    def schema(self):
        names, types = [], []
        for value in self.by + self.metrics:
            names.append(value.name)
            types.append(value.dtype)
        return Schema.from_tuples(zip(names, types))

    @annotated
    def order_by(self, keys: VarTuple[SortKey]):
        from ibis.expr.analysis import shares_all_roots, sub_immediate_parents

        if shares_all_roots(keys, table := self.table):
            sort_keys = tuple(self.sort_keys) + tuple(
                sub_immediate_parents(key, table) for key in keys
            )
            return Aggregation(
                table,
                metrics=self.metrics,
                by=self.by,
                having=self.having,
                predicates=self.predicates,
                sort_keys=sort_keys,
            )

        return Selection(self, [], sort_keys=keys)


@public
class Distinct(Relation):
    """Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """

    table: Relation

    @attribute
    def schema(self):
        return self.table.schema


@public
class Sample(Relation):
    """Sample performs random sampling of records in a table."""

    table: Relation
    fraction: Annotated[float, Between(0, 1)]
    method: Literal["row", "block"]
    seed: UnionType[int, None] = None

    @attribute
    def schema(self):
        return self.table.schema


# TODO(kszucs): split it into two operations, one working with a single replacement
# value and the other with a mapping
# TODO(kszucs): the single value case was limited to numeric and string types
@public
class FillNa(Relation):
    """Fill null values in the table."""

    table: Relation
    replacements: UnionType[Value[dt.Numeric | dt.String], FrozenDict[str, Any]]

    @attribute
    def schema(self):
        return self.table.schema


@public
class DropNa(Relation):
    """Drop null values in the table."""

    table: Relation
    how: Literal["any", "all"]
    subset: Optional[VarTuple[Column[dt.Any]]] = None

    @attribute
    def schema(self):
        return self.table.schema


@public
class View(PhysicalTable):
    """A view created from an expression."""

    child: Relation
    name: str

    @attribute
    def schema(self):
        return self.child.schema


@public
class SQLStringView(PhysicalTable):
    """A view created from a SQL string."""

    child: Relation
    name: str
    query: str

    @attribute
    def schema(self):
        # TODO(kszucs): avoid converting to expression
        backend = self.child.to_expr()._find_backend()
        return backend._get_schema_using_query(self.query)


def _dedup_join_columns(expr, lname: str, rname: str):
    from ibis.expr.operations.generic import TableColumn
    from ibis.expr.operations.logical import Equals

    op = expr.op()
    left = op.left.to_expr()
    right = op.right.to_expr()

    right_columns = frozenset(right.columns)
    overlap = frozenset(column for column in left.columns if column in right_columns)
    equal = set()

    if isinstance(op, InnerJoin) and util.all_of(op.predicates, Equals):
        # For inner joins composed exclusively of equality predicates, we can
        # avoid renaming columns with colliding names if their values are
        # guaranteed to be equal due to the predicate. Here we collect a set of
        # colliding column names that are known to have equal values between
        # the left and right tables in the join.
        tables = {op.left, op.right}
        for pred in op.predicates:
            if (
                isinstance(pred.left, TableColumn)
                and isinstance(pred.right, TableColumn)
                and {pred.left.table, pred.right.table} == tables
                and pred.left.name == pred.right.name
            ):
                equal.add(pred.left.name)

    if not overlap:
        return expr

    # Rename columns in the left table that overlap, unless they're known to be
    # equal to a column in the right
    left_projections = [
        left[column].name(lname.format(name=column) if lname else column)
        if column in overlap and column not in equal
        else left[column]
        for column in left.columns
    ]

    # Rename columns in the right table that overlap, dropping any columns that
    # are known to be equal to those in the left table
    right_projections = [
        right[column].name(rname.format(name=column) if rname else column)
        if column in overlap
        else right[column]
        for column in right.columns
        if column not in equal
    ]
    projections = left_projections + right_projections

    # Certain configurations can result in the renamed columns still colliding,
    # here we check for duplicates again, and raise a nicer error message if
    # any exist.
    seen = set()
    collisions = set()
    for column in projections:
        name = column.get_name()
        if name in seen:
            collisions.add(name)
        seen.add(name)
    if collisions:
        raise com.IntegrityError(
            f"Joining with `lname={lname!r}, rname={rname!r}` resulted in multiple "
            f"columns mapping to the following names `{sorted(collisions)}`. Please "
            f"adjust `lname` and/or `rname` accordingly"
        )
    return expr.select(projections)


public(TableNode=Relation)
