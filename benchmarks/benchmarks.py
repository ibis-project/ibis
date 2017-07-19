import ibis
import numpy as np
import pandas as pd


class Suite:
    def setup(self):
        self.t = t = ibis.table((('_timestamp', 'int32'),
                                 ('dim1', 'int32'),
                                 ('dim2', 'int32'),
                                 ('valid_seconds', 'int32'),
                                 ('meas1', 'int32'),
                                 ('meas2', 'int32'),
                                 ('year', 'int32'),
                                 ('month', 'int32'),
                                 ('day', 'int32'),
                                 ('hour', 'int32'),
                                 ('minute', 'int32')), name='t')
        self.base = (
            (t.year > 2016) | (
                (t.year == 2016) & (t.month > 6)) | (
                    (t.year == 2016) & (t.month == 6) &
                    (t.day > 6)) | (
                        (t.year == 2016) & (t.month == 6) &
                        (t.day == 6) & (t.hour > 6)) |
            ((t.year == 2016) & (t.month == 6) &
             (t.day == 6) & (t.hour == 6) &
             (t.minute >= 5))) & ((t.year < 2016) | (
                 (t.year == 2016) & (t.month < 6)) | (
                     (t.year == 2016) & (t.month == 6) &
                     (t.day < 6)) | (
                         (t.year == 2016) & (t.month == 6) &
                         (t.day == 6) & (t.hour < 6)) | (
                             (t.year == 2016) &
                             (t.month == 6) & (t.day == 6) &
                             (t.hour == 6) &
                             (t.minute <= 5)))
        self.expr = self.large_expr

    @property
    def large_expr(self):
        src_table = self.t[self.base]
        src_table = src_table.mutate(_timestamp=(
            src_table['_timestamp'] - src_table['_timestamp'] % 3600
        ).cast('int32').name('_timestamp'), valid_seconds=300)

        aggs = []
        for meas in ['meas1', 'meas2']:
            aggs.append(src_table[meas].sum().cast('float').name(meas))
        src_table = src_table.aggregate(
            aggs, by=['_timestamp', 'dim1', 'dim2', 'valid_seconds'])

        part_keys = ['year', 'month', 'day', 'hour', 'minute']
        ts_col = src_table['_timestamp'].cast('timestamp')
        new_cols = {}
        for part_key in part_keys:
            part_col = getattr(ts_col, part_key)()
            new_cols[part_key] = part_col
        src_table = src_table.mutate(**new_cols)
        return src_table[[
            '_timestamp', 'dim1', 'dim2', 'meas1', 'meas2',
            'year', 'month', 'day', 'hour', 'minute'
        ]]


class Construction(Suite):

    def time_large_expr_construction(self):
        self.large_expr


class Formatting(Suite):

    def time_base_expr_formatting(self):
        str(self.base)

    def time_large_expr_formatting(self):
        str(self.expr)


class Compilation(Suite):

    def time_impala_base_compile(self):
        ibis.impala.compile(self.base)

    def time_impala_large_expr_compile(self):
        ibis.impala.compile(self.expr)


class PandasBackend:

    def setup(self):
        n = int(5e6)
        data = pd.DataFrame({
            'key': np.random.choice(16000, size=n),
            'value': np.random.rand(n),
        })
        t = ibis.pandas.connect({'df': data}).table('df')
        self.expr = t.groupby(t.key).aggregate(avg_value=t.value.mean())

    def time_high_cardinality_group_by(self):
        self.expr.execute()
