from __future__ import annotations

import textwrap
from functools import singledispatch
from typing import TYPE_CHECKING

import ibis.expr.operations as ops
import ibis.expr.operations.relations as rels
from ibis.backends.pandas.rewrites import PandasJoin, PandasRename

if TYPE_CHECKING:
    from pathlib import Path


@singledispatch
def translate(op, **_) -> list[str]:
    """Write the output of a unix shell pipeline to target."""
    raise NotImplementedError(f"operation {type(op)} not implemented")


@translate.register(ops.Field)
def field(op, *, name, **_) -> str:
    return f"${op.rel.schema.names.index(name) + 1:d}"


@translate.register(ops.Literal)
def lit(op, *, value, **_):
    if op.dtype.is_string():
        # extremely production ready
        return f'"{value}"'
    return str(value)


@translate.register(ops.StringLength)
def string_length(op, *, arg, **_) -> str:
    return f"length({arg})"


@translate.register(ops.Lowercase)
def lowercase(op, *, arg, **_) -> str:
    return f"tolower({arg})"


@translate.register(ops.Uppercase)
def uppercase(op, *, arg, **_) -> str:
    return f"toupper({arg})"


opcodes = {
    ops.Equals: "==",
    ops.NotEquals: "!=",
    ops.GreaterEqual: ">=",
    ops.Greater: ">",
    ops.LessEqual: "<=",
    ops.Less: "<",
    ops.And: "&&",
    ops.Or: "||",
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.Power: "^",
}


@translate.register(ops.Binary)
def binary(op, *, left, right, **_):
    operator = opcodes[type(op)]
    return f"{left} {operator} {right}"


@translate.register(ops.BitwiseXor)
def bitwise_xor(op, *, left, right, **_):
    return f"xor({left}, {right})"


@translate.register(ops.BitwiseOr)
def bitwise_or(op, *, left, right, **_):
    return f"or({left}, {right})"


@translate.register(ops.BitwiseAnd)
def bitwise_and(op, *, left, right, **_):
    return f"and({left}, {right})"


@translate.register(ops.BitwiseNot)
def bitwise_not(op, *, arg, **_):
    return f"compl({arg})"


@translate.register(ops.BitwiseLeftShift)
def bitwise_left_shift(op, *, arg, **_):
    return f"lshift({arg})"


@translate.register(ops.BitwiseRightShift)
def bitwise_right_shift(op, *, arg, **_):
    return f"rshift({arg})"


@translate.register(ops.Not)
def not_(op, *, arg, **_):
    return f"!({arg})"


@translate.register(ops.Floor)
def floor(op, *, arg, **_):
    return f"int({arg})"


@translate.register(ops.Cos)
def cos(op, *, arg, **_):
    return f"cos({arg})"


@translate.register(ops.Sin)
def sin(op, *, arg, **_):
    return f"sin({arg})"


@translate.register(ops.Tan)
def tan(op, *, arg, **_):
    return f"sin({arg}) / cos({arg})"


@translate.register(ops.Sqrt)
def sqrt(op, *, arg, **_):
    return f"sqrt({arg})"


@translate.register(ops.Exp)
def exp(op, *, arg, **_):
    return f"exp({arg})"


@translate.register(ops.Ln)
def log(op, *, arg, **_):
    return f"log({arg})"


@translate.register(ops.RandomScalar)
def rand(op, **_):
    return "rand()"


@translate.register(ops.TypeOf)
def typeof(op, *, arg, **_):
    return f"typeof({arg})"


@translate.register(ops.CountStar)
def count_star(op, *, arg, where):
    return "++"


@translate.register(ops.Sum)
def total(op, *, arg, where):
    return f"+={arg}"


@translate.register(ops.Mean)
def mean(op, *, arg, where):
    return f"+={arg}/NR"


@translate.register(ops.SortKey)
def sort_key(op, *, expr, ascending):
    return expr


@translate.register(ops.DatabaseTable)
def table(op, *, name: str, **_) -> list[str]:
    # skip the header, fastest type inference ever
    return ["tail", "--lines", "+2", op.source._tables[name]]


@translate.register(ops.Sort)
def sort(op, *, keys, parent: Path) -> list[str]:
    keyspec = []
    for raw_key, key in zip(op.keys, keys):
        idx = key.removeprefix("$")

        keyspec.append("-k")

        suffix = "r" * raw_key.descending + "n" * raw_key.expr.dtype.is_numeric()
        keyspec.append(f"{idx},{idx}{suffix}")

    return ["sort", "-t", ",", *keyspec, parent]


@translate.register(ops.Project)
def project(op, *, values, parent):
    fieldspec = ' "," '.join(values.values())
    return ["awk", "-F", ",", f"{{ print {fieldspec} }}", parent]


@translate.register(ops.Aggregate)
def agg(op, *, groups, metrics, parent):
    if groups:
        agg_lines = [
            "agg{m:d}[{key}]{metric}".format(
                m=m, metric=metric, key='","'.join(groups.values())
            )
            for m, metric in enumerate(metrics.values())
        ]
        output_line = ' "," '.join(f"agg{m:d}[key]" for m in range(len(metrics)))
        output_line = f'for (key in agg0) print key "," {output_line}'
    else:
        agg_lines = [f"agg{m:d}{metric}" for m, metric in enumerate(metrics.values())]
        output_line = ' "," '.join(f"agg{m:d}" for m in range(len(metrics)))
        output_line = f"print {output_line}"

    program_lines = [
        "{",
        textwrap.indent("\n".join(agg_lines), prefix=" " * 2),
        "}",
        f"END {{ {output_line} }}",
    ]
    program = "\n".join(program_lines)
    return ["awk", "-F", ",", program, parent]


@translate.register(ops.Filter)
def filt(op, *, predicates, parent):
    awk_predicate = " && ".join(f"({predicate})" for predicate in predicates)
    return [
        "awk",
        "-F",
        ",",
        f"{{ if ({awk_predicate}) {{ print }}}}",
        parent,
    ]


@translate.register(ops.JoinTable)
@translate.register(PandasRename)
def no_op(op, *, parent, **_):
    return ["cat", parent]


@translate.register(PandasJoin)
def pandas_join(op, *, left, right, left_on, right_on, how, **_):
    if how != "inner":
        raise NotImplementedError(f"join type {how} not implemented")

    if len(left_on) != 1 or len(right_on) != 1:
        raise NotImplementedError("multi-column join not implemented")

    left_on = left_on[0].removeprefix("$")
    right_on = right_on[0].removeprefix("$")

    left_cols = [f"1.{idx}" for idx in range(1, len(op.left.schema) + 1)]
    right_cols = [f"2.{idx}" for idx in range(1, len(op.right.schema) + 1)]

    return [
        "join",
        "--check-order",
        "-o",
        ",".join(left_cols + right_cols),
        "-t",
        ",",
        "-1",
        str(left_on),
        "-2",
        str(right_on),
        left,
        right,
    ]


class Offset(rels.Simple):
    n: int


@translate.register(Offset)
def offset(op, *, n, parent):
    assert n
    n += 1
    return ["tail", "--lines", f"+{n:d}", parent]


@translate.register(ops.Limit)
def limit(op, *, n, offset, parent):
    assert not offset
    return ["head", "--lines", f"{n:d}", parent]
