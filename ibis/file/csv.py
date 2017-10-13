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

    def insert(self, path, t, index=False, **kwargs):
        path = self.root / path
        data = execute(t)
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
        # tables are files in a dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_file():
                    if str(d).endswith(self.extension):
                        tables.append(d.stem)
        elif path.is_file():
            if str(path).endswith(self.extension):
                tables.append(path.stem)
        return tables

    def list_databases(self, path=None):
        # databases are dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
        return tables


@pre_execute.register(CSVTable, CSVClient)
def csv_pre_execute_table(op, client, scope=None, **kwargs):
    path = client.dictionary[op.name]
    df = pd.read_csv(str(path), header=0)
    return {op: df}


@pre_execute.register(ops.Selection, CSVClient)
def csv_pre_execute(op, client, scope=None, **kwargs):

    pt = physical_tables(op.table.op())
    pt = pt[0]

    path = client.dictionary[pt.name]

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
    return {op: df}
