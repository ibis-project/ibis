from __future__ import absolute_import

import six

import numpy as np
import pandas as pd

import ibis
import ibis.client as client
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


try:
    infer_dtype = pd.api.types.infer_dtype
except AttributeError:
    infer_dtype = pd.lib.infer_dtype


_DTYPE_TO_IBIS_TYPE = {
    'float64': dt.double,
    'float32': dt.float,
    'datetime64[ns]': dt.timestamp,
}


_INFERRED_DTYPE_TO_IBIS_TYPE = {
    'string': 'string',
    'unicode': 'string',
    'bytes': 'string',
}


def pandas_dtypes_to_ibis_schema(df, schema):
    dtypes = df.dtypes

    pairs = []

    for column_name, dtype in dtypes.iteritems():
        if not isinstance(column_name, six.string_types):
            raise TypeError(
                'Column names must be strings to use the pandas backend'
            )

        if column_name in schema:
            ibis_type = dt.validate_type(schema[column_name])
        elif dtype == np.object_:
            inferred_dtype = infer_dtype(df[column_name].dropna())

            if inferred_dtype == 'mixed':
                raise TypeError(
                    'Unable to infer type of column {0!r}. Try instantiating '
                    'your table from the client with client.table('
                    "'my_table', schema={{{0!r}: <explicit type>}})".format(
                        column_name
                    )
                )
            ibis_type = _INFERRED_DTYPE_TO_IBIS_TYPE[inferred_dtype]
        elif hasattr(dtype, 'tz'):
            ibis_type = dt.Timestamp(str(dtype.tz))
        else:
            dtype_string = str(dtype)
            ibis_type = _DTYPE_TO_IBIS_TYPE.get(dtype_string, dtype_string)

        pairs.append((column_name, ibis_type))
    return ibis.schema(pairs)


class PandasClient(client.Client):

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def table(self, name, schema=None):
        df = self.dictionary[name]
        schema = pandas_dtypes_to_ibis_schema(
            df, schema if schema is not None else {}
        )
        return ops.DatabaseTable(name, schema, self).to_expr()

    def execute(self, query, params=None, limit='default', async=False):
        from ibis.pandas.execution import execute

        if limit != 'default':
            raise ValueError(
                'limit parameter to execute is not yet implemented in the '
                'pandas backend'
            )

        if async:
            raise ValueError(
                'async is not yet supported in the pandas backend'
            )

        assert isinstance(query, ir.Expr)
        return execute(query, params=params)
