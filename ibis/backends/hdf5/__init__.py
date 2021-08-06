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

    def list_databases(self, path=None):
        return self._list_databases_dirs_or_files(path)


class Backend(BaseFileBackend):
    name = 'hdf5'
    extension = 'h5'
    table_class = HDFTable
    client_class = HDFClient

    def list_tables(self, like=None):
        """
        For HDF5, tables are the HDF5 tables inside the file.
        """
        if self.path.is_file() and self.path.suffix == '.' + self.extension:
            with pd.HDFStore(str(self.path), mode='r') as store:
                # strip leading /
                return [k[1:] for k in store.keys()]

        return []


@execute_node.register(Backend.table_class, HDFClient)
def hdf_read_table(op, client, scope, **kwargs):
    key = op.name
    path = client.dictionary[key]
    df = pd.read_hdf(str(path), key, mode='r')
    return df
