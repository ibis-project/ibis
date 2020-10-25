import pandas as pd
import toolz
from pkg_resources import parse_version

import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base_file import FileClient
from ibis.backends.pandas import PandasDialect
from ibis.backends.pandas.core import execute, execute_node, pre_execute
from ibis.backends.pandas.execution.selection import physical_tables
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

dialect = PandasDialect


def _read_csv(path, schema, **kwargs):
    dtypes = dict(schema.to_pandas())

    dates = list(toolz.valfilter(lambda s: s == 'datetime64[ns]', dtypes))
    dtypes = toolz.dissoc(dtypes, *dates)

    return pd.read_csv(
        str(path), dtype=dtypes, parse_dates=dates, encoding='utf-8', **kwargs
    )


def connect(path):
    """Create a CSVClient for use with Ibis

    Parameters
    ----------
    path: str or pathlib.Path

    Returns
    -------
    CSVClient
    """
    return CSVClient(path)


class CSVTable(ops.DatabaseTable):
    def __init__(self, name, schema, source, **kwargs):
        super().__init__(name, schema, source)
        self.read_csv_kwargs = kwargs


class CSVClient(FileClient):

    dialect = dialect
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
        f = path / "{}.{}".format(name, self.extension)

        # read sample
        schema = schema or sch.schema([])
        sample = _read_csv(f, schema=schema, header=0, nrows=50, **kwargs)

        # infer sample's schema and define table
        schema = sch.infer(sample, schema=schema)
        table = self.table_class(name, schema, self, **kwargs).to_expr()

        self.dictionary[name] = f

        return table

    def list_tables(self, path=None):
        return self._list_tables_files(path)

    def list_databases(self, path=None):
        return self._list_databases_dirs(path)

    def compile(self, expr, *args, **kwargs):
        return expr

    @property
    def version(self):
        return parse_version(pd.__version__)


@execute_node.register(CSVClient.table_class, CSVClient)
def csv_read_table(op, client, scope, **kwargs):
    path = client.dictionary[op.name]
    df = _read_csv(path, schema=op.schema, header=0, **op.read_csv_kwargs)
    return df


@pre_execute.register(ops.Selection, CSVClient)
def csv_pre_execute_selection(
    op: ops.Node,
    client: CSVClient,
    scope: Scope,
    timecontext: TimeContext = None,
    **kwargs,
):
    tables = filter(
        lambda t: scope.get_value(t, timecontext) is None,
        physical_tables(op.table.op()),
    )

    ops = Scope()
    for table in tables:
        path = client.dictionary[table.name]
        usecols = None

        if op.selections:
            header = _read_csv(path, schema=table.schema, header=0, nrows=1)
            usecols = [
                getattr(s.op(), 'name', None) or s.get_name()
                for s in op.selections
            ]

            # we cannot read all the columns that we would like
            if len(pd.Index(usecols) & header.columns) != len(usecols):
                usecols = None
        result = _read_csv(path, table.schema, usecols=usecols, header=0)
        ops = ops.merge_scope(Scope({table: result}, timecontext))

    return ops
