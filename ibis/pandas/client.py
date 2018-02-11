from __future__ import absolute_import

import six
import toolz
import numpy as np
import pandas as pd

import ibis.client as client
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.compat import PY2, DatetimeTZDtype, CategoricalDtype, parse_version


try:
    infer_pandas_dtype = pd.api.types.infer_dtype
except AttributeError:
    infer_pandas_dtype = pd.lib.infer_dtype


_ibis_dtypes = {
    dt.Boolean: 'bool',
    dt.Null: 'object',
    dt.Array: 'object',
    dt.String: 'object',
    dt.Binary: 'object',
    dt.Category: 'category',
    dt.Date: 'datetime64[ns]',
    dt.Time: 'datetime64[ns]',
    dt.Timestamp: 'datetime64[ns]',
    dt.Int8: 'int8',
    dt.Int16: 'int16',
    dt.Int32: 'int32',
    dt.Int64: 'int64',
    dt.UInt8: 'uint8',
    dt.UInt16: 'uint16',
    dt.UInt32: 'uint32',
    dt.UInt64: 'uint64',
    dt.Float32: 'float32',
    dt.Float64: 'float64',
    dt.Decimal: 'float64',
    dt.Struct: 'object',
}


_numpy_dtypes = toolz.keymap(np.dtype, {
    'bool': dt.boolean,
    'int8': dt.int8,
    'int16': dt.int16,
    'int32': dt.int32,
    'int64': dt.int64,
    'uint8': dt.uint8,
    'uint16': dt.uint16,
    'uint32': dt.uint32,
    'uint64': dt.uint64,
    'float16': dt.float16,
    'float32': dt.float32,
    'float64': dt.float64,
    'double': dt.double,
    'unicode': dt.string,
    'str': dt.string,
    'datetime64': dt.timestamp,
    'datetime64[ns]': dt.timestamp,
    'timedelta64': dt.interval,
    'timedelta64[ns]': dt.Interval('ns')
})


_inferable_pandas_dtypes = {
    'boolean': dt.boolean,
    'string': dt.string,
    'unicode': dt.string,
    'bytes': dt.string,
    'empty': dt.string,
}


@dt.dtype.register(np.dtype)
def from_numpy_dtype(value):
    return _numpy_dtypes[value]


@dt.dtype.register(DatetimeTZDtype)
def from_pandas_tzdtype(value):
    return dt.Timestamp(timezone=str(value.tz))


@dt.dtype.register(CategoricalDtype)
def from_pandas_categorical(value):
    return dt.Category()


@dt.infer.register(
    (np.generic,) + tuple(
        frozenset(
            np.signedinteger.__subclasses__() +
            np.unsignedinteger.__subclasses__()  # np.int64, np.uint64, etc.
        )
    )  # we need this because in Python 2 int is a parent of np.integer
)
def infer_numpy_scalar(value):
    return dt.dtype(value.dtype)


@dt.infer.register(pd.Timestamp)
def infer_pandas_timestamp(value):
    return dt.Timestamp(timezone=str(value.tz))


@dt.infer.register(np.ndarray)
def infer_array(value):
    # TODO(kszucs): infer series
    return dt.Array(dt.dtype(value.dtype.name))


@sch.schema.register(pd.Series)
def schema_from_series(s):
    return sch.schema(tuple(s.iteritems()))


@sch.infer.register(pd.DataFrame)
def infer_pandas_schema(df, schema=None):
    schema = schema if schema is not None else {}

    pairs = []
    for column_name, pandas_dtype in df.dtypes.iteritems():
        if not isinstance(column_name, six.string_types):
            raise TypeError(
                'Column names must be strings to use the pandas backend'
            )

        if column_name in schema:
            ibis_dtype = dt.dtype(schema[column_name])
        elif pandas_dtype == np.object_:
            inferred_dtype = infer_pandas_dtype(df[column_name].dropna())
            if inferred_dtype == 'mixed':
                raise TypeError(
                    'Unable to infer type of column {0!r}. Try instantiating '
                    'your table from the client with client.table('
                    "'my_table', schema={{{0!r}: <explicit type>}})".format(
                        column_name
                    )
                )
            ibis_dtype = _inferable_pandas_dtypes[inferred_dtype]
        else:
            ibis_dtype = dt.dtype(pandas_dtype)

        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


def ibis_dtype_to_pandas(ibis_dtype):
    """Convert ibis dtype to the pandas / numpy alternative"""
    if isinstance(ibis_dtype, dt.Timestamp) and ibis_dtype.timezone:
        return DatetimeTZDtype('ns', ibis_dtype.timezone)
    elif (isinstance(ibis_dtype, dt.DataType) and
            type(ibis_dtype) not in _ibis_dtypes):
        return 'object'
    return _ibis_dtypes[type(ibis_dtype)]


def ibis_schema_to_pandas(schema):
    return list(zip(schema.names, map(ibis_dtype_to_pandas, schema.types)))


def ibis_schema_apply_to(schema, df):
    """Applies the Ibis schema on a pandas dataframe"""

    for column, dtype in schema.items():
        df[column] = df[column].astype(dtype.to_pandas(), errors='ignore')
        if PY2 and dtype == dt.string:
            df[column] = df[column].str.decode('utf-8', errors='ignore')

    return df


dt.DataType.to_pandas = ibis_dtype_to_pandas
sch.Schema.to_pandas = ibis_schema_to_pandas
sch.Schema.apply_to = ibis_schema_apply_to


class PandasTable(ops.DatabaseTable):
    pass


class PandasClient(client.Client):

    dialect = None  # defined in ibis.pandas.api

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def table(self, name, schema=None):
        df = self.dictionary[name]
        schema = sch.infer(df, schema=schema)
        return PandasTable(name, schema, self).to_expr()

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

    def compile(self, expr, *args, **kwargs):
        return expr

    def database(self, name=None):
        return PandasDatabase(name, self)

    @property
    def version(self):
        return parse_version(pd.__version__)


class PandasDatabase(client.Database):
    pass
