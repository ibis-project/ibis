from __future__ import annotations

from functools import reduce, singledispatchmethod

import sqlglot as sg
from public import public

import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import NULL, STAR
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import DuckDBType


@public
class DuckDBCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "duckdb"
    type_mapper = DuckDBType

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.ArrayDistinct)
    def visit_ArrayDistinct(self, op, *, arg, **_):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index, **_):
        return self.f.list_extract(arg, index + self.cast(index >= 0, op.index.dtype))

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times, **_):
        func = sg.exp.Lambda(this=arg, expressions=[sg.to_identifier("_")])
        return self.f.flatten(self.f.list_apply(self.f.range(times), func))

    @visit_node.register(ops.Sample)
    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        sample = sg.exp.TableSample(
            this=parent,
            method="bernoulli" if method == "row" else "system",
            percent=sg.exp.convert(fraction * 100.0),
            seed=None if seed is None else sg.exp.convert(seed),
        )
        return sg.select(STAR).from_(sample)

    @visit_node.register(ops.ArraySlice)
    def visit_ArraySlice(self, op, *, arg, start, stop, **_):
        arg_length = self.f.len(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, self._neg_idx_to_pos(arg, start))

        if stop is None:
            stop = arg_length
        else:
            stop = self._neg_idx_to_pos(arg, stop)

        return self.f.list_slice(arg, start + 1, stop)

    @visit_node.register(ops.ArrayMap)
    def visit_ArrayMap(self, op, *, arg, body, param, **_):
        lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_apply(arg, lamduh)

    @visit_node.register(ops.ArrayFilter)
    def visit_ArrayFilter(self, op, *, arg, body, param, **_):
        lamduh = sg.exp.Lambda(this=body, expressions=[sg.to_identifier(param)])
        return self.f.list_filter(arg, lamduh)

    @visit_node.register(ops.ArrayIntersect)
    def visit_ArrayIntersect(self, op, *, left, right, **_):
        param = sg.to_identifier("x")
        body = self.f.list_contains(right, param)
        lamduh = sg.exp.Lambda(this=body, expressions=[param])
        return self.f.list_filter(left, lamduh)

    @visit_node.register(ops.ArrayRemove)
    def visit_ArrayRemove(self, op, *, arg, other, **_):
        param = sg.to_identifier("x")
        body = param.neq(other)
        lamduh = sg.exp.Lambda(this=body, expressions=[param])
        return self.f.list_filter(arg, lamduh)

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right, **_):
        arg = self.f.list_concat(left, right)
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.list_distinct(arg)
            + self.if_(
                self.f.list_count(arg) < self.f.len(arg),
                self.f.array(NULL),
                self.f.array(),
            ),
        )

    @visit_node.register(ops.ArrayZip)
    def visit_ArrayZip(self, op, *, arg, **_):
        i = sg.to_identifier("i")
        body = sg.exp.Struct.from_arg_list(
            [
                sg.exp.Slice(this=k, expression=v[i])
                for k, v in zip(map(sg.exp.convert, op.dtype.value_type.names), arg)
            ]
        )
        func = sg.exp.Lambda(this=body, expressions=[i])
        return self.f.list_apply(
            self.f.range(
                1,
                # DuckDB Range excludes upper bound
                self.f.greatest(*map(self.f.len, arg)) + 1,
            ),
            func,
        )

    @visit_node.register(ops.MapGet)
    def visit_MapGet(self, op, *, arg, key, default, **_):
        return self.f.ifnull(
            self.f.list_extract(self.f.element_at(arg, key), 1), default
        )

    @visit_node.register(ops.MapContains)
    def visit_MapContains(self, op, *, arg, key, **_):
        return self.f.len(self.f.element_at(arg, key)).neq(0)

    @visit_node.register(ops.ToJSONMap)
    @visit_node.register(ops.ToJSONArray)
    def visit_ToJSONMap(self, op, *, arg, **_):
        return self.f.try_cast(arg, self.type_mapper.from_ibis(op.dtype))

    @visit_node.register(ops.ArrayConcat)
    def visit_ArrayConcat(self, op, *, arg, **_):
        return reduce(self.f.list_concat, arg)


_SIMPLE_OPS = {
    ops.ArrayPosition: "list_indexof",
    ops.ArraySort: "list_sort",
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.EndsWith: "suffix",
    ops.Hash: "hash",
    ops.IntegerRange: "range",
    ops.LPad: "lpad",
    ops.Levenshtein: "levenshtein",
    ops.MapKeys: "map_keys",
    ops.MapLength: "cardinality",
    ops.MapMerge: "map_concat",
    ops.MapValues: "map_values",
    ops.Mode: "mode",
    ops.RPad: "rpad",
    ops.Reverse: "reverse",
    ops.StringAscii: "ascii",
    ops.StringReplace: "replace",
    ops.StringToTimestamp: "strptime",
    ops.TimeFromHMS: "make_time",
    ops.TypeOf: "typeof",
    ops.Unnest: "unnest",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @DuckDBCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @DuckDBCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(DuckDBCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
