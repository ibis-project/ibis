import numpy as np
import pandas as pd

import ibis
import ibis.expr.datatypes as dt


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
        n = 30 * int(2e5)
        data = pd.DataFrame({
            'key': np.random.choice(16000, size=n),
            'low_card_key': np.random.choice(30, size=n),
            'value': np.random.rand(n),
            'timestamps': pd.date_range(
                start='now', periods=n, freq='s'
            ).values,
            'timestamp_strings': pd.date_range(
                start='now', periods=n, freq='s'
            ).values.astype(str),
            'repeated_timestamps': pd.date_range(
                start='2018-09-01', periods=30).repeat(int(n / 30))
        })

        t = ibis.pandas.connect({'df': data}).table('df')

        self.high_card_group_by = t.groupby(t.key).aggregate(
            avg_value=t.value.mean()
        )

        self.cast_to_dates = t.timestamps.cast(dt.date)
        self.cast_to_dates_from_strings = t.timestamp_strings.cast(dt.date)

        self.multikey_group_by_with_mutate = t.mutate(
            dates=t.timestamps.cast('date')
        ).groupby(['low_card_key', 'dates']).aggregate(
            avg_value=lambda t: t.value.mean()
        )

        self.simple_sort = t.sort_by([t.key])

        self.simple_sort_projection = t[['key', 'value']].sort_by(['key'])

        self.multikey_sort = t.sort_by(['low_card_key', 'key'])

        self.multikey_sort_projection = t[[
            'low_card_key', 'key', 'value'
        ]].sort_by(['low_card_key', 'key'])

        low_card_window = ibis.trailing_range_window(
            2 * ibis.day(),
            order_by=t.repeated_timestamps,
            group_by=t.low_card_key)
        self.low_card_grouped_rolling = t.value.mean().over(low_card_window)

        high_card_window = ibis.trailing_range_window(
            2 * ibis.day(),
            order_by=t.repeated_timestamps,
            group_by=t.key)
        self.high_card_grouped_rolling = t.value.mean().over(high_card_window)

    def time_high_cardinality_group_by(self):
        self.high_card_group_by.execute()

    def time_cast_to_date(self):
        self.cast_to_dates.execute()

    def time_cast_to_date_from_string(self):
        self.cast_to_dates_from_strings.execute()

    def time_multikey_group_by_with_mutate(self):
        self.multikey_group_by_with_mutate.execute()

    def time_simple_sort(self):
        self.simple_sort.execute()

    def time_multikey_sort(self):
        self.multikey_sort.execute()

    def time_simple_sort_projection(self):
        self.simple_sort_projection.execute()

    def time_multikey_sort_projection(self):
        self.multikey_sort_projection.execute()

    def time_low_card_grouped_rolling(self):
        self.low_card_grouped_rolling.execute()

    def time_high_card_grouped_rolling(self):
        self.high_card_grouped_rolling.execute()
