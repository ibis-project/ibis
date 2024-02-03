from __future__ import annotations

import operator
from functools import reduce

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

import ibis.backends.dask.kernels as dask_kernels
import ibis.expr.operations as ops
from ibis.backends.dask.convert import DaskConverter
from ibis.backends.dask.helpers import (
    DaskUtils,
    add_globally_consecutive_column,
)
from ibis.backends.pandas.executor import PandasExecutor
from ibis.backends.pandas.rewrites import (
    PandasAggregate,
    PandasJoin,
    PandasLimit,
    PandasResetIndex,
    PandasScalarSubquery,
    plan,
)
from ibis.common.exceptions import UnboundExpressionError
from ibis.formats.pandas import PandasData, PandasType
from ibis.util import gen_name

# ruff: noqa: F811


class DaskExecutor(PandasExecutor, DaskUtils):
    name = "dask"
    kernels = dask_kernels

    @classmethod
    def visit(cls, op: ops.Node, **kwargs):
        return super().visit(op, **kwargs)

    @classmethod
    def visit(cls, op: ops.Cast, arg, to):
        if arg is None:
            return None
        elif isinstance(arg, dd.Series):
            return DaskConverter.convert_column(arg, to)
        else:
            return DaskConverter.convert_scalar(arg, to)

    @classmethod
    def visit(
        cls, op: ops.SimpleCase | ops.SearchedCase, cases, results, default, base=None
    ):
        def mapper(df, cases, results, default):
            cases = [case.astype("bool") for case in cases]
            cases.append(pd.Series(True, index=df.index))

            results.append(default)
            out = np.select(cases, results)

            return pd.Series(out, index=df.index)

        dtype = PandasType.from_ibis(op.dtype)
        if base is not None:
            cases = tuple(base == case for case in cases)
        kwargs = dict(cases=cases, results=results, default=default)

        return cls.partitionwise(mapper, kwargs, name=op.name, dtype=dtype)

    @classmethod
    def visit(cls, op: ops.TimestampTruncate | ops.DateTruncate, arg, unit):
        # TODO(kszucs): should use serieswise()
        unit = {"m": "Min", "ms": "L"}.get(unit.short, unit.short)
        try:
            return arg.dt.floor(unit)
        except ValueError:
            return arg.dt.to_period(unit).dt.to_timestamp()

    @classmethod
    def visit(cls, op: ops.IntervalFromInteger, unit, **kwargs):
        if unit.short in {"Y", "Q", "M", "W"}:
            return cls.elementwise(
                lambda v: pd.DateOffset(**{unit.plural: v}),
                kwargs,
                name=op.name,
                dtype=object,
            )
        else:
            return cls.serieswise(
                lambda arg: arg.astype(f"timedelta64[{unit.short}]"), kwargs
            )

    @classmethod
    def visit(cls, op: ops.BetweenTime, arg, lower_bound, upper_bound):
        if getattr(arg.dtype, "tz", None) is not None:
            localized = arg.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            localized = arg

        time = localized.dt.time.astype(str)
        indexer = ((time >= lower_bound) & (time <= upper_bound)).to_dask_array(True)

        result = da.zeros(len(arg), dtype=np.bool_)
        result[indexer] = True
        return dd.from_array(result)

    @classmethod
    def visit(cls, op: ops.FindInSet, needle, values):
        def mapper(df, cases):
            thens = [i for i, _ in enumerate(cases)]
            out = np.select(cases, thens, default=-1)
            return pd.Series(out, index=df.index)

        dtype = PandasType.from_ibis(op.dtype)
        cases = [needle == value for value in values]
        kwargs = dict(cases=cases)
        return cls.partitionwise(mapper, kwargs, name=op.name, dtype=dtype)

    @classmethod
    def visit(cls, op: ops.Array, exprs):
        return cls.rowwise(
            lambda row: np.array(row, dtype=object), exprs, name=op.name, dtype=object
        )

    @classmethod
    def visit(cls, op: ops.ArrayConcat, arg):
        dtype = PandasType.from_ibis(op.dtype)
        return cls.rowwise(
            lambda row: np.concatenate(row.values), arg, name=op.name, dtype=dtype
        )

    @classmethod
    def visit(cls, op: ops.Unnest, arg):
        arg = cls.asseries(arg)
        mask = arg.map(lambda v: bool(len(v)), na_action="ignore")
        return arg[mask].explode()

    @classmethod
    def visit(
        cls, op: ops.ElementWiseVectorizedUDF, func, func_args, input_type, return_type
    ):
        """Execute an elementwise UDF."""

        def mapper(df):
            cols = [df[col] for col in df]
            return func(*cols)

        df, _ = cls.asframe(func_args)
        result = df.map_partitions(mapper)
        if op.dtype.is_struct():
            result = result.apply(lambda row: row.to_dict(), axis=1)
        return result

    ############################# Reductions ##################################

    @classmethod
    def visit(cls, op: ops.ArgMin | ops.ArgMax, arg, key, where):
        # TODO(kszucs): raise a warning about triggering compute()?
        if isinstance(op, ops.ArgMin):
            func = lambda x: x.idxmin()
        else:
            func = lambda x: x.idxmax()

        if where is None:

            def agg(df):
                indices = func(df[key.name])
                if isinstance(indices, (dd.Series, dd.core.Scalar)):
                    # to support both aggregating within a group and globally
                    indices = indices.compute()
                return df[arg.name].loc[indices]
        else:

            def agg(df):
                mask = df[where.name]
                filtered = df[mask]
                indices = func(filtered[key.name])
                if isinstance(indices, (dd.Series, dd.core.Scalar)):
                    # to support both aggregating within a group and globally
                    indices = indices.compute()
                return filtered[arg.name].loc[indices]

        return agg

    @classmethod
    def visit(cls, op: ops.Correlation, left, right, where, how):
        if where is None:

            def agg(df):
                return df[left.name].corr(df[right.name])
        else:

            def agg(df):
                mask = df[where.name]
                lhs = df[left.name][mask].compute()
                rhs = df[right.name][mask].compute()
                return lhs.corr(rhs)

        return agg

    @classmethod
    def visit(cls, op: ops.Covariance, left, right, where, how):
        # TODO(kszucs): raise a warning about triggering compute()?
        ddof = {"pop": 0, "sample": 1}[how]
        if where is None:

            def agg(df):
                lhs = df[left.name].compute()
                rhs = df[right.name].compute()
                return lhs.cov(rhs, ddof=ddof)
        else:

            def agg(df):
                mask = df[where.name]
                lhs = df[left.name][mask].compute()
                rhs = df[right.name][mask].compute()
                return lhs.cov(rhs, ddof=ddof)

        return agg

    @classmethod
    def visit(
        cls, op: ops.ReductionVectorizedUDF, func, func_args, input_type, return_type
    ):
        def agg(df):
            # if df is a dask dataframe then we collect it to a pandas dataframe
            # because the user-defined function expects a pandas dataframe
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            args = [df[col.name] for col in func_args]
            return func(*args)

        return agg

    @classmethod
    def visit(
        cls, op: ops.AnalyticVectorizedUDF, func, func_args, input_type, return_type
    ):
        def agg(df, order_keys):
            # if df is a dask dataframe then we collect it to a pandas dataframe
            # because the user-defined function expects a pandas dataframe
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            args = [df[col.name] for col in func_args]
            res = func(*args)
            if isinstance(res, pd.DataFrame):
                # it is important otherwise it is going to fill up the memory
                res = res.apply(lambda row: row.to_dict(), axis=1)
            return res

        return agg

    ############################ Window functions #############################

    @classmethod
    def visit(cls, op: ops.WindowFrame, table, start, end, **kwargs):
        table = table.compute()
        if isinstance(start, dd.Series):
            start = start.compute()
        if isinstance(end, dd.Series):
            end = end.compute()
        return super().visit(op, table=table, start=start, end=end, **kwargs)

    @classmethod
    def visit(cls, op: ops.WindowFunction, func, frame):
        result = super().visit(op, func=func, frame=frame)
        return cls.asseries(result)

    ############################ Relational ###################################

    @classmethod
    def visit(cls, op: ops.DatabaseTable, name, schema, source, namespace):
        try:
            return source.dictionary[name]
        except KeyError:
            raise UnboundExpressionError(
                f"{name} is not a table in the {source.name!r} backend, you "
                "probably tried to execute an expression without a data source"
            )

    @classmethod
    def visit(cls, op: ops.InMemoryTable, name, schema, data):
        df = data.to_frame().reset_index(drop=True)
        return dd.from_pandas(df, npartitions=1)

    @classmethod
    def visit(cls, op: ops.DummyTable, values):
        df, _ = cls.asframe(values)
        return df

    @classmethod
    def visit(cls, op: PandasLimit, parent, n, offset):
        n = n.compute().iat[0, 0]
        offset = offset.compute().iat[0, 0]

        name = gen_name("limit")
        df = add_globally_consecutive_column(parent, name, set_as_index=False)
        if n is None:
            df = df[df[name] >= offset]
        else:
            df = df[df[name].between(offset, offset + n - 1)]

        return df.drop(columns=[name])

    @classmethod
    def visit(cls, op: PandasResetIndex, parent):
        return add_globally_consecutive_column(parent)

    @classmethod
    def visit(cls, op: PandasJoin, **kwargs):
        df = super().visit(op, **kwargs)
        return add_globally_consecutive_column(df)

    @classmethod
    def visit(cls, op: ops.Project, parent, values):
        df, all_scalars = cls.asframe(values)
        if all_scalars and len(parent) != len(df):
            df = dd.concat([df] * len(parent))
        return df

    @classmethod
    def visit(cls, op: ops.Filter, parent, predicates):
        if predicates:
            pred = reduce(operator.and_, predicates)
            parent = parent.loc[pred].reset_index(drop=True)
        return parent

    @classmethod
    def visit(cls, op: ops.Sort, parent, keys):
        # 1. add sort key columns to the dataframe if they are not already present
        # 2. sort the dataframe using those columns
        # 3. drop the sort key columns
        ascending = [key.ascending for key in op.keys]
        newcols = {gen_name("sort_key"): col for col in keys}
        names = list(newcols.keys())
        df = parent.assign(**newcols)
        df = df.sort_values(by=names, ascending=ascending)
        return df.drop(names, axis=1)

    @classmethod
    def visit(cls, op: PandasAggregate, parent, groups, metrics):
        if not groups:
            results = {k: v(parent) for k, v in metrics.items()}
            combined, _ = cls.asframe(results)
            return combined

        parent = parent.groupby([col.name for col in groups.values()])

        measures = {}
        for name, metric in metrics.items():
            meta = pd.Series(
                name=name,
                dtype=PandasType.from_ibis(op.metrics[name].dtype),
                index=pd.MultiIndex(
                    levels=[[] for _ in groups],
                    codes=[[] for _ in groups],
                    names=list(groups.keys()),
                ),
            )
            measures[name] = parent.apply(metric, meta=meta)

        result = cls.concat(measures, axis=1).reset_index()
        renames = {v.name: k for k, v in op.groups.items()}
        return result.rename(columns=renames)

    @classmethod
    def visit(cls, op: ops.InValues, value, options):
        if isinstance(value, dd.Series):
            return value.isin(options)
        else:
            return value in options

    @classmethod
    def visit(cls, op: ops.InSubquery, rel, needle):
        first_column = rel.compute().iloc[:, 0]
        if isinstance(needle, dd.Series):
            return needle.isin(first_column)
        else:
            return needle in first_column

    @classmethod
    def visit(cls, op: PandasScalarSubquery, rel):
        # TODO(kszucs): raise a warning about triggering compute()?
        # could the compute be avoided here?
        return rel.compute().iat[0, 0]

    @classmethod
    def compile(cls, node, backend, params):
        def fn(node, _, **kwargs):
            return cls.visit(node, **kwargs)

        node = node.to_expr().as_table().op()
        node = plan(node, backend=backend, params=params)
        return node.map_clear(fn)

    @classmethod
    def execute(cls, node, backend, params):
        original = node
        node = node.to_expr().as_table().op()
        result = cls.compile(node, backend=backend, params=params)

        # should happen when the result is empty
        if isinstance(result, pd.DataFrame):
            assert result.empty
        else:
            assert isinstance(result, dd.DataFrame)
            result = result.compute()

        result = PandasData.convert_table(result, node.schema)
        if isinstance(original, ops.Value):
            if original.shape.is_scalar():
                return result.iloc[0, 0]
            elif original.shape.is_columnar():
                return result.iloc[:, 0]
            else:
                raise TypeError(f"Unexpected shape: {original.shape}")
        else:
            return result
