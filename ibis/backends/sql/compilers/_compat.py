"""Compatibility layer for SQLGlot expressions."""

from __future__ import annotations

import sqlglot.expressions as sge


def _get_arg_name(expr_cls: type[sge.Expression], *args: str) -> str:
    for arg in args:
        if arg in expr_cls.arg_types:
            return arg
    raise ValueError(f"None of {args} are valid arguments for {expr_cls.__name__}")


# "precompute" common arg names
# sqglot >= 28.0 renamed arg types that mirror python keywords to include a trailing underscore
# to keep the code backward compatible, we take the first valid arg name
WITH_ARG = _get_arg_name(sge.Select, "with", "with_")
EXCEPT_ARG = _get_arg_name(sge.Star, "except", "except_")
