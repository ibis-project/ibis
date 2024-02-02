from __future__ import annotations

from typing import Callable

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ibis.backends.pandas.helpers import PandasUtils


class DaskUtils(PandasUtils):
    @classmethod
    def merge(cls, *args, **kwargs):
        return dd.merge(*args, **kwargs)

    @classmethod
    def merge_asof(cls, *args, **kwargs):
        return dd.merge_asof(*args, **kwargs)

    @classmethod
    def concat(cls, dfs, **kwargs):
        if isinstance(dfs, dict):
            dfs = [v.rename(k) for k, v in dfs.items()]
        return dd.concat(dfs, **kwargs)

    @classmethod
    def asseries(cls, value, like=None):
        """Ensure that value is a pandas Series object, broadcast if necessary."""

        if isinstance(value, dd.Series):
            return value
        elif isinstance(value, dd.core.Scalar):
            # Create a Dask array from the Dask scalar
            try:
                dtype = value.dtype
            except AttributeError:
                #     @property
                #     def dtype(self):
                # >       return self._meta.dtype
                # E       AttributeError: 'Timestamp' object has no attribute 'dtype'
                dtype = object
            array = da.from_delayed(value.to_delayed(), (1,), dtype=dtype)
            # Create a Dask series from the Dask array
            return dd.from_array(array)
        elif isinstance(value, pd.Series):
            return dd.from_pandas(value, npartitions=1)
        elif like is not None:
            if isinstance(value, (tuple, list, dict)):
                fn = lambda df: pd.Series([value] * len(df), index=df.index)
            else:
                fn = lambda df: pd.Series(value, index=df.index)
            return like.map_partitions(fn)
        else:
            return dd.from_pandas(pd.Series([value]), npartitions=1)

    @classmethod
    def asframe(cls, values: dict | tuple):
        # TODO(kszucs): prefer using assign instead of concat
        """Construct a DataFrame from a dict or tuple of Series objects."""
        if isinstance(values, dict):
            names, values = zip(*values.items())
        elif isinstance(values, tuple):
            names = [f"_{i}" for i in range(len(values))]
        else:
            raise TypeError(f"values must be a dict, or tuple; got {type(values)}")

        all_scalars = True
        representative = None
        for v in values:
            if isinstance(v, dd.Series):
                all_scalars = False
                representative = v
                break

        columns = [cls.asseries(v, like=representative) for v in values]
        columns = [v.rename(k) for k, v in zip(names, columns)]

        # dd.concat turns decimal.Decimal("NaN") into np.nan for some reason
        df = dd.concat(columns, axis=1)
        return df, all_scalars

    @classmethod
    def rowwise(cls, func: Callable, operands, name, dtype):
        if dtype == np.dtype("<M8[D]"):
            # Getting the following error otherwise:
            #     TypeError: dtype=datetime64[D] is not supported. Supported
            #     resolutions are 's', 'ms', 'us', and 'ns'
            dtype = "datetime64[s]"
        # dealing with a collection of series objects
        df, _ = cls.asframe(operands)
        return df.apply(func, axis=1, meta=(name, dtype))

    @classmethod
    def elementwise(cls, func, operands, name, dtype):
        meta = (name, dtype)
        value = operands.pop(next(iter(operands)))
        if isinstance(value, dd.Series):
            # dealing with a single series object
            if operands:
                return value.apply(func, **operands, meta=meta)
            else:
                return value.map(func, meta=meta, na_action="ignore")
        else:
            # dealing with a single scalar object
            return func(value, **operands)

    @classmethod
    def partitionwise(cls, func, operands, name, dtype):
        cols = {}
        kwargs = {}
        for name, operand in operands.items():
            if isinstance(operand, (tuple, list)):
                for i, v in enumerate(operand):
                    cols[f"{name}_{i}"] = v
                kwargs[name] = tuple(f"{name}_{i}" for i in range(len(operand)))
            else:
                cols[name] = operand
                kwargs[name] = name

        def mapper(df):
            unpacked = {}
            for k, operand in kwargs.items():
                if isinstance(operand, (tuple, list)):
                    unpacked[k] = [df[col] for col in operand]
                else:
                    unpacked[k] = df[operand]
            return func(df, **unpacked)

        df, _ = cls.asframe(cols)
        return df.map_partitions(mapper, meta=(name, dtype))


def add_globally_consecutive_column(
    df: dd.DataFrame | dd.Series,
    name: str = "_ibis_index",
    set_as_index: bool = True,
) -> dd.DataFrame:
    """Add a column that is globally consecutive across the distributed data.

    By construction, this column is already sorted and can be used to partition
    the data.
    This column can act as if we had a global index across the distributed data.
    This index needs to be consecutive in the range of [0, len(df)), allows
    downstream operations to work properly.
    The default index of dask dataframes is to be consecutive within each partition.

    Important properties:

    - Each row has a unique id (i.e. a value in this column)
    - The global index that's added is consecutive in the same order that the rows currently are in.
    - IDs within each partition are already sorted

    We also do not explicitly deal with overflow in the bounds.

    Parameters
    ----------
    df: dd.DataFrame
        Dataframe to add the column to
    name: str
        Name of the column to use. Default is _ibis_index
    set_as_index: bool
        If True, will set the consecutive column as the index. Default is True.

    Returns
    -------
    dd.DataFrame
        New dask dataframe with sorted partitioned index

    """
    if isinstance(df, dd.Series):
        df = df.to_frame()

    if name in df.columns:
        raise ValueError(f"Column {name} is already present in DataFrame")

    df = df.assign(**{name: 1})
    df = df.assign(**{name: df[name].cumsum() - 1})
    if set_as_index:
        df = df.reset_index(drop=True)
        df = df.set_index(name, sorted=True)

    # No elegant way to rename index https://github.com/dask/dask/issues/4950
    df = df.map_partitions(pd.DataFrame.rename_axis, None, axis="index")

    return df
