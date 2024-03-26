from __future__ import annotations

import concurrent.futures
import itertools
import os
import shlex
import tempfile
from collections.abc import Mapping
from pathlib import Path
from subprocess import Popen
from typing import Any

import pandas as pd

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends import BaseBackend, NoUrl
from ibis.backends.pandas.rewrites import (
    PandasJoin,
    bind_unbound_table,
    replace_parameter,
    rewrite_join,
)
from ibis.backends.unix.compiler import Offset, translate
from ibis.common.patterns import replace


@replace(PandasJoin)
def sort_before_join(_):
    return _.copy(
        left=ops.Sort(_.left, keys=_.left_on), right=ops.Sort(_.right, keys=_.right_on)
    )


@replace(ops.Limit)
def split_limit_with_offset(_, **__):
    if _.offset:
        return _.copy(parent=Offset(parent=_.parent, n=_.offset), offset=0, n=_.n)
    return _


class Backend(BaseBackend, NoUrl):
    name = "unix"
    dialect = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = {}

    def do_connect(self, tables: Mapping[str, Path | str] | None = None) -> None:
        """Construct a client from a dictionary of paths.

        Parameters
        ----------
        tables
            An optional mapping of string table names to paths.

        """
        if tables is not None and not isinstance(tables, Mapping):
            raise TypeError("Input to ibis.unix.connect must be a mapping")

        # tables are emphemeral
        self._tables.clear()

        for name, table in (tables or {}).items():
            self._add_table(name, table)

    def disconnect(self) -> None:
        self._tables.clear()

    @property
    def version(self) -> str:
        return "🤡"

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self._tables.keys()), like)

    def table(self, name: str) -> ir.Table:
        import ibis.expr.schema as sch

        schema = sch.infer(pd.read_csv(self._tables[name], header=0, nrows=100))
        return ops.DatabaseTable(name, schema, self).to_expr()

    def _add_table(self, name: str, obj: Path | str) -> None:
        self._tables[name] = str(Path(obj).absolute())

    def _remove_table(self, name: str) -> None:
        del self._tables[name]

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        *,
        pipe_dir: str | None = None,
        **_: Any,
    ):
        if pipe_dir is None:
            pipe_dir = tempfile.mkdtemp()

        if params is None:
            params = dict()
        else:
            params = {param.op(): value for param, value in params.items()}

        node = expr.as_table().op()
        node = node.replace(
            rewrite_join
            | replace_parameter
            | bind_unbound_table
            | split_limit_with_offset,
            context={"params": params, "backend": self},
        ).replace(sort_before_join)

        counter = itertools.count()

        commands = []

        def fn(node, _, **kwargs):
            source = translate(node, **kwargs)
            # relations are the only nodes that are executed
            #
            # value expressions are strings that are passed to the command
            # (likely awk for any non-trivial computation)
            if isinstance(node, ops.Relation):
                path = Path(pipe_dir, f"t{next(counter)}")
                commands.append((source, path))
                return path

            return source

        node.map(fn)
        return commands

    def to_pyarrow(self, *args, **kwargs) -> Any:
        import pyarrow as pa

        return pa.Table.from_pandas(self.execute(*args, **kwargs))

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        **kwargs: Any,
    ):
        schema = expr.as_table().schema()
        columns = schema.names
        pandas_schema = dict(schema.to_pandas())
        with (
            tempfile.TemporaryDirectory() as pipe_dir,
            concurrent.futures.ThreadPoolExecutor() as exe,
        ):
            plan = self.compile(
                expr, params=params, pipe_dir=pipe_dir, exe=exe, **kwargs
            )

            for cmd, path in plan:
                os.mkfifo(path)
                exe.submit(
                    lambda cmd, path: Popen(cmd, stdout=path.open(mode="w")),
                    cmd=cmd,
                    path=path,
                )

            # start consuming the output
            return pd.read_csv(
                path,
                header=None,
                names=columns,
                dtype=pandas_schema,
            )

    def explain(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        **kwargs: Any,
    ) -> str:
        with tempfile.TemporaryDirectory() as pipe_dir:
            plan = self.compile(expr, params=params, pipe_dir=pipe_dir, **kwargs)

        return "\n".join(
            f"{shlex.join(list(map(str, cmd)))} > {output.name}" for cmd, output in plan
        )

    @classmethod
    def has_operation(self, operation: type[ops.Value]) -> bool:
        import random

        return random.random() > 0.5

    def create_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)


if __name__ == "__main__":
    # Create a backend with some data
    unix = Backend()
    backend = unix.connect({"p": "/data/penguins.csv", "q": "/data/penguins.csv"})
    # Create an expression
    t = backend.table("p")
    q = backend.table("q")
    expr = (
        t.filter([t.year == 2009])
        .select("year", "flipper_length_mm", island=lambda t: t.island.lower())
        .group_by("island")
        .agg(n=lambda t: t.count())
        .order_by("n")
        .mutate(ilength=lambda t: t.island.length())
        .limit(5)
    )
    pipeline = backend.explain(expr)
    print(pipeline)  # noqa: T201

    result = expr.execute()
    print(result)  # noqa: T201

    result = t.join(t, ["year"]).select("year").execute()
    print(result)  # noqa: T201
