import pandas as pd
import ibis.expr.operations as ops
from ibis.file.client import FileClient
from ibis.pandas.core import pre_execute, execute  # noqa
from ibis.pandas.client import pandas_dtypes_to_ibis_schema
from ibis.pandas.execution.selection import physical_tables


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
    pass


class CSVClient(FileClient):
    extension = 'csv'

    def insert(self, path, expr, index=False, **kwargs):
        path = self.root / path
        data = execute(expr)
        data.to_csv(str(path), index=index, **kwargs)

    def table(self, name, path=None):
        if name not in self.list_tables(path):
            raise AttributeError(name)

        if path is None:
            path = self.root

        # get the schema
        f = path / "{}.{}".format(name, self.extension)
        df = pd.read_csv(str(f), header=0, nrows=10)
        schema = pandas_dtypes_to_ibis_schema(df, {})

        t = CSVTable(name, schema, self).to_expr()
        self.dictionary[name] = f
        return t

    def list_tables(self, path=None):
        return self._list_tables_files(path)

    def list_databases(self, path=None):
        return self._list_databases_dirs(path)


@pre_execute.register(CSVTable, CSVClient)
def csv_pre_execute_table(op, client, scope, **kwargs):

    # cache
    if isinstance(scope.get(op), pd.DataFrame):
        return {}

    path = client.dictionary[op.name]
    df = pd.read_csv(str(path), header=0)
    return {op: df}


@pre_execute.register(ops.Selection, CSVClient)
def csv_pre_execute(op, client, scope, **kwargs):

    tables = physical_tables(op.table.op())

    ops = {}
    for table in tables:
        if table in scope:
            continue

        path = client.dictionary[table.name]

        if op.selections:

            header = pd.read_csv(str(path), header=0, nrows=1)
            usecols = [getattr(s.op(), 'name', None) or s.get_name()
                       for s in op.selections]

            # we cannot read all the columns taht we would like
            if len(pd.Index(usecols) & header.columns) != len(usecols):
                usecols = None

        else:

            usecols = None

        df = pd.read_csv(str(path), usecols=usecols, header=0)
        ops[table] = df
    return ops
