import pandas as pd

import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base.file import BaseFileBackend
from ibis.backends.pandas.core import execute, execute_node
from ibis.util import warn_deprecated


class HDFTable(ops.DatabaseTable):
    pass


class Backend(BaseFileBackend):
    name = 'hdf5'
    extension = 'h5'
    table_class = HDFTable

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
            warn_deprecated(
                'The `path` argument of `list_databases`',
                version='2.0',
                instead='`connect()` with a different path',
            )
        databases = self._list_databases_dirs_or_files(path)
        return self._filter_with_like(databases, like)

    def insert(
        self, path, key, expr, format='table', data_columns=True, **kwargs
    ):

        path = self.root / path
        data = execute(expr)
        data.to_hdf(
            str(path), key, format=format, data_columns=data_columns, **kwargs
        )

    def table(self, name, path=None):
        if path is None:
            path = self.root / f"{name}.{self.extension}"

        if name not in self.list_tables(path):
            raise AttributeError(name)

        # get the schema
        with pd.HDFStore(str(path), mode='r') as store:
            df = store.select(name, start=0, stop=50)
            schema = sch.infer(df)

        t = self.table_class(name, schema, self).to_expr()
        self.dictionary[name] = path
        return t


@execute_node.register(Backend.table_class, Backend)
def hdf_read_table(op, client, scope, **kwargs):
    key = op.name
    path = client.dictionary[key]
    df = pd.read_hdf(str(path), key, mode='r')
    return df
