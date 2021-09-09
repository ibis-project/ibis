import warnings

import pandas as pd

import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base.file import BaseFileBackend, FileClient
from ibis.backends.pandas.core import execute, execute_node


class HDFTable(ops.DatabaseTable):
    pass


class HDFClient(FileClient):
    def insert(
        self, path, key, expr, format='table', data_columns=True, **kwargs
    ):

        path = self.root / path
        data = execute(expr)
        data.to_hdf(
            str(path), key, format=format, data_columns=data_columns, **kwargs
        )

    def table(self, name, path):
        if name not in self.list_tables(path):
            raise AttributeError(name)

        # get the schema
        with pd.HDFStore(str(path), mode='r') as store:
            df = store.select(name, start=0, stop=0)
            schema = sch.infer(df)

        t = self.table_class(name, schema, self).to_expr()
        self.dictionary[name] = path
        return t


class Backend(BaseFileBackend):
    name = 'hdf5'
    extension = 'h5'
    table_class = HDFTable
    client_class = HDFClient

    def list_tables(self, path=None, like=None, database=None):
        """
        For HDF5, tables are the HDF5 tables inside the file.
        """
        path = path or self.path

        if path.is_file() and path.suffix == '.' + self.extension:
            with pd.HDFStore(str(path), mode='r') as store:
                # strip leading /
                return [k[1:] for k in store.keys()]

        return []

    def _list_databases_dirs_or_files(self, path=None):
        # databases are dir & file
        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
                elif d.is_file():
                    if str(d).endswith(self.extension):
                        tables.append(d.stem)
        elif path.is_file():
            # by definition we are at the db level at this point
            pass

        return tables

    def list_databases(self, path=None, like=None):
        if path is None:
            path = self.path
        else:
            warnings.warn(
                'The `path` argument of `list_databases` is deprecated and '
                'will be removed in a future version of Ibis. Connect to a '
                'different path with the `connect()` method instead.',
                FutureWarning,
            )
        databases = self._list_databases_dirs_or_files(path)
        return self._filter_with_like(databases, like)


@execute_node.register(Backend.table_class, HDFClient)
def hdf_read_table(op, client, scope, **kwargs):
    key = op.name
    path = client.dictionary[key]
    df = pd.read_hdf(str(path), key, mode='r')
    return df
