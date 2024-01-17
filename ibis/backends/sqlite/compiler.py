from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import SQLiteType


@public
class SQLiteCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "sqlite"
    type_mapper = SQLiteType

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.ArrayStringJoin)
    def visit_Undefined(self, op, **kwargs):
        return super().visit_Undefined(op, **kwargs)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return arg.like(self.f.concat(start, "%"))

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return arg.like(self.f.concat("%", end))

    @visit_node.register(ops.StrRight)
    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars, nchars)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if op.end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if op.start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, substr)
            return sg.case().when(pos > 0, pos + start).else_(0)

        return self.f.instr(arg, substr)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        args = [arg[0]]
        for item in arg[1:]:
            args.extend([sep, item])
        return self.f.concat(*args)

    @visit_node.register(ops.StringContains)
    def visit_Contains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) >= 1

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        if op.key is None:
            return self.f._ibis_extract_full_query(arg)
        return self.f._ibis_extract_query(arg, key)


_SIMPLE_OPS = {
    ops.RegexReplace: "_ibis_sqlite_regex_replace",
    ops.RegexExtract: "_ibis_sqlite_regex_extract",
    ops.RegexSearch: "_ibis_sqlite_regex_search",
    ops.Translate: "_ibis_sqlite_translate",
    ops.Capitalize: "_ibis_sqlite_capitalize",
    ops.Reverse: "_ibis_sqlite_reverse",
    ops.RPad: "_ibis_sqlite_rpad",
    ops.LPad: "_ibis_sqlite_lpad",
    ops.Repeat: "_ibis_sqlite_repeat",
    ops.StringAscii: "_ibis_sqlite_string_ascii",
    ops.ExtractAuthority: "_ibis_extract_authority",
    ops.ExtractFragment: "_ibis_extract_fragment",
    ops.ExtractHost: "_ibis_extract_host",
    ops.ExtractPath: "_ibis_extract_path",
    ops.ExtractProtocol: "_ibis_extract_protocol",
    ops.ExtractUserInfo: "_ibis_extract_user_info",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(SQLiteCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
