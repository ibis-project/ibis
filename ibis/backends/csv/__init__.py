from __future__ import annotations

import pandas as pd
import toolz

import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base.file import BaseFileBackend
from ibis.backends.pandas.core import execute, execute_node


def _read_csv(path, schema, **kwargs):
    dtypes = dict(schema.to_pandas())

    dates = list(toolz.valfilter(lambda s: s == 'datetime64[ns]', dtypes))
    dtypes = toolz.dissoc(dtypes, *dates)

    return pd.read_csv(
        str(path), dtype=dtypes, parse_dates=dates, encoding='utf-8', **kwargs
    )


class CSVTable(ops.DatabaseTable):
    def __init__(self, name, schema, source, **kwargs):
        super().__init__(name, schema, source)
        self.read_csv_kwargs = kwargs


class Backend(BaseFileBackend):
    name = 'csv'
    extension = 'csv'
    table_class = CSVTable

    def insert(self, path, expr, index=False, **kwargs):
        path = self.root / path
        data = execute(expr)
        data.to_csv(str(path), index=index, **kwargs)

    def table(self, name, path=None, schema=None, **kwargs):
        if name not in self.list_tables(path):
            raise AttributeError(name)

        if path is None:
            path = self.root

        # get the schema
        f = path / f"{name}.{self.extension}"

        # read sample
        schema = schema or sch.schema([])
        sample = _read_csv(f, schema=schema, header=0, nrows=50, **kwargs)

        # infer sample's schema and define table
        schema = sch.infer(sample, schema=schema)
        table = self.table_class(name, schema, self, **kwargs).to_expr()

        self.dictionary[name] = f

        return table


@execute_node.register(Backend.table_class, Backend)
def csv_read_table(op, client, scope, **kwargs):
    path = client.dictionary[op.name]
    df = _read_csv(path, schema=op.schema, header=0, **op.read_csv_kwargs)
    return df
