"""Feldera SQL compiler.

Feldera parses SQL with Apache Calcite using a Postgres-flavoured surface,
so we subclass :class:`PostgresCompiler` and only declare the (small) set of
operations that Feldera/Calcite does not (yet) support.
"""

from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import SQLGlotCompiler
from ibis.backends.sql.compilers.postgres import PostgresCompiler
from ibis.backends.sql.datatypes import FelderaType
from ibis.backends.sql.dialects import Feldera


class FelderaCompiler(PostgresCompiler):
    __slots__ = ()

    dialect = Feldera
    type_mapper = FelderaType

    # Operations we know Feldera/Calcite doesn't support today.  This list is
    # intentionally conservative; we will trim it as the operation matrix is
    # validated against a running pipeline (see `ibis/backends/feldera/tests`).
    UNSUPPORTED_OPS = (
        ops.Sample,
        ops.RandomScalar,
        ops.RandomUUID,
        ops.Arbitrary,
        ops.Mode,
        ops.Kurtosis,
        ops.MultiQuantile,
        ops.ApproxMultiQuantile,
    )

    # Postgres lowers Sample to TABLESAMPLE; Feldera has no equivalent.
    LOWERED_OPS = {ops.Sample: None}


compiler = FelderaCompiler()
