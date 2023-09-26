"""DuckDB ibis expression to sqlglot compiler."""

from __future__ import annotations

import functools

from ibis.backends.base.sqlglot.compiler.core import translate as _translate
from ibis.backends.duckdb.compiler.relations import translate_rel
from ibis.backends.duckdb.compiler.values import translate_val

translate = functools.partial(
    _translate, translate_rel=translate_rel, translate_val=translate_val
)
