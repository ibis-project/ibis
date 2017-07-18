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


def pandas_dtypes_to_ibis_schema(df):
    dtypes = df.dtypes

    pairs = []

    for column_name, dtype in dtypes.iteritems():
        if not isinstance(column_name, six.string_types):
            raise TypeError(
                'Column names must be strings to use the pandas backend'
            )

        if dtype == np.object_:
            ibis_type = _INFERRED_DTYPE_TO_IBIS_TYPE[
                infer_dtype(df[column_name].dropna())
            ]
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

    def table(self, name):
        df = self.dictionary[name]
        schema = pandas_dtypes_to_ibis_schema(df)
        return ops.DatabaseTable(name, schema, self).to_expr()

    def execute(self, query, *args, **kwargs):
        from ibis.pandas.execution import execute

        assert isinstance(query, ir.Expr)
        return execute(query)
