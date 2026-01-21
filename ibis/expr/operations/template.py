"""Operations for template strings (t-strings)."""

from __future__ import annotations

from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.optimizer.annotate_types import annotate_types

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.deferred import Deferred
from ibis.common.exceptions import IbisInputError
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr import operations as ops
from ibis.expr.operations.util import find_backend

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ibis.backends.sql.datatypes import SqlglotType
    from ibis.expr.operations.relations import Relation
    from ibis.tstring import PTemplate


Dialect = str


@public
class TemplateSQLValue(ops.Value):
    strings: VarTuple[str]
    values: VarTuple[ops.Value]
    dialect: Dialect
    """The SQL dialect the template was written in.

    eg if t'CAST({val} AS REAL)', you should use 'sqlite',
    since REAL is a sqlite-specific concept.
    """
    dtype: dt.DataType

    @classmethod
    def from_template(
        cls,
        template: PTemplate,
        /,
        *,
        dialect: Dialect | None = None,
        dtype: dt.IntoDtype | None = None,
    ) -> TemplateSQLValue:
        raw_values = [interp.value for interp in template.interpolations]
        resolved_values = ensure_values(raw_values)
        if dialect is None:
            backend = find_backend(resolved_values)
            if backend is None:
                backend = ibis.get_backend()
            # Check for eg polars backends
            from ibis.backends.sql import SQLBackend

            if not isinstance(backend, SQLBackend):
                raise IbisInputError(
                    f"Expected a SQL backend, got {type(backend)}: {backend}"
                )
            dialect = backend.name
        if dtype is None:
            parts = interleave(template.strings, resolved_values)
            sql = sql_from_parts(parts, dialect=dialect)
            dtype = dtype_from_sql(dialect, sql)
        dtype = dt.dtype(dtype)
        return cls(
            strings=template.strings,
            values=resolved_values,
            dialect=dialect,
            dtype=dtype,
        )

    @attribute
    def shape(self):
        if not self.values:
            return ds.scalar
        return rlz.highest_precedence_shape(self.values)

    @attribute
    def relations(self) -> frozenset[Relation]:
        return relations_of_vals(self.values)

    @property
    def sql_for_inference(self) -> str:
        parts = interleave(self.strings, self.values)
        return sql_from_parts(parts, dialect=self.dialect)

    @property
    def type_mapper(self) -> SqlglotType:
        return get_type_mapper(self.dialect)


def relations_of_vals(vals: Iterable[ops.Value]) -> frozenset[Relation]:
    children = (v.relations for v in vals)
    return frozenset().union(*children)


def ensure_values(raw: Iterable[Any]) -> tuple[ops.Value, ...]:
    raw = [_try_to_op_value(x) for x in raw]
    already_values = [v for v in raw if isinstance(v, ops.Value)]
    relations = relations_of_vals(already_values)
    if len(relations) > 1:
        raise IbisInputError(
            f"A SQL value can only depend on a single relation, got {len(relations)}"
        )
    relation = next(iter(relations), None)
    return tuple(ensure_value(r, relation) for r in raw)


def _try_to_op_value(x):
    from ibis.expr import types as ir

    if isinstance(x, ops.Value):
        return x
    if isinstance(x, ir.Value):
        return x.op()
    return x


def ensure_value(raw: Any, relation: Relation | None) -> ops.Value:
    result = _ensure_value(raw, relation)
    if not isinstance(result, ops.Value):
        raise TypeError(
            f"Could not convert object {raw} of type {type(raw)} to Value in context of relation {relation}"
        )
    return result


def _ensure_value(raw: Any, relation: Relation | None) -> ops.Value:
    if isinstance(raw, ops.Value):
        return raw
    if relation is None:
        return ibis.literal(raw).op()
    if isinstance(raw, Deferred):
        return raw.resolve(relation).op()
    if callable(raw):
        called = raw(relation)
        return ensure_value(called, relation)
    return ibis.literal(raw).op()


def interleave(
    strings: Iterable[str], values: Iterable[ops.Value]
) -> tuple[str | ops.Value, ...]:
    FILL = object()

    def iter() -> Iterator[str | ops.Value]:
        for s, v in zip_longest(strings, values, fillvalue=FILL):
            if s is not FILL:
                yield s
            if v is not FILL:
                yield v

    return tuple(iter())


def sql_from_parts(parts: tuple[str | ops.Value, ...], dialect: Dialect) -> str:
    result: list[str] = []
    for part in parts:
        if isinstance(part, str):
            result.append(part)
        else:
            ibis_type: dt.DataType = part.dtype
            null_sqlglot_value = sge.cast(
                sge.null(), get_type_mapper(dialect).from_ibis(ibis_type)
            )
            result.append(null_sqlglot_value.sql(dialect))
    return "".join(result)


def dtype_from_sql(dialect: Dialect, sql: str) -> dt.DataType:
    try:
        parsed = sg.parse_one(sql, dialect=dialect)
    except sg.errors.ParseError as e:
        raise IbisInputError(f"failed to parse {sql}") from e
    annotated = annotate_types(parsed, dialect=dialect)
    sqlglot_type = annotated.type
    type_mapper = get_type_mapper(dialect)
    return type_mapper.to_ibis(sqlglot_type)


def get_type_mapper(dialect: Dialect) -> SqlglotType:
    """Get the type mapper for the given SQL dialect."""
    from ibis.backends.sql.datatypes import TYPE_MAPPERS

    return TYPE_MAPPERS[dialect.lower()]
