# @pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
# def test_simple_window(alltypes, func, df):
#     t = alltypes
#     f = getattr(t.double_col, func)
#     df_f = getattr(df.double_col, func)
#     result = t.projection([
#         (t.double_col - f()).name('double_col')
#     ]).execute().double_col
#     expected = df.double_col - df_f()
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
# def test_rolling_window(alltypes, func, df):
#     t = alltypes
#     df = df[['double_col', 'timestamp_col']].sort_values(
#         'timestamp_col'
#     ).reset_index(drop=True)
#     window = ibis.window(
#         order_by=t.timestamp_col,
#         preceding=6,
#         following=0
#     )
#     f = getattr(t.double_col, func)
#     df_f = getattr(df.double_col.rolling(7, min_periods=0), func)
#     result = t.projection([
#         f().over(window).name('double_col')
#     ]).execute().double_col
#     expected = df_f()
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
# def test_partitioned_window(alltypes, func, df):
#     t = alltypes
#     window = ibis.window(
#         group_by=t.string_col,
#         order_by=t.timestamp_col,
#         preceding=6,
#         following=0,
#     )

#     def roller(func):
#         def rolled(df):
#             torder = df.sort_values('timestamp_col')
#             rolling = torder.double_col.rolling(7, min_periods=0)
#             return getattr(rolling, func)()
#         return rolled

#     f = getattr(t.double_col, func)
#     expr = f().over(window).name('double_col')
#     result = t.projection([expr]).execute().double_col
#     expected = df.groupby('string_col').apply(
#         roller(func)
#     ).reset_index(drop=True)
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['sum', 'min', 'max'])
# def test_cumulative_simple_window(alltypes, func, df):
#     t = alltypes
#     f = getattr(t.double_col, func)
#     col = t.double_col - f().over(ibis.cumulative_window())
#     expr = t.projection([col.name('double_col')])
#     result = expr.execute().double_col
#     expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['sum', 'min', 'max'])
# def test_cumulative_partitioned_window(alltypes, func, df):
#     t = alltypes
#     df = df.sort_values('string_col').reset_index(drop=True)
#     window = ibis.cumulative_window(group_by=t.string_col)
#     f = getattr(t.double_col, func)
#     expr = t.projection([
#         (t.double_col - f().over(window)).name('double_col')
#     ])
#     result = expr.execute().double_col
#     expected = df.groupby(df.string_col).double_col.transform(
#         lambda c: c - getattr(c, 'cum%s' % func)()
#     )
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['sum', 'min', 'max'])
# def test_cumulative_ordered_window(alltypes, func, df):
#     t = alltypes
#     df = df.sort_values('timestamp_col').reset_index(drop=True)
#     window = ibis.cumulative_window(order_by=t.timestamp_col)
#     f = getattr(t.double_col, func)
#     expr = t.projection([
#         (t.double_col - f().over(window)).name('double_col')
#     ])
#     result = expr.execute().double_col
#     expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize('func', ['sum', 'min', 'max'])
# def test_cumulative_partitioned_ordered_window(alltypes, func, df):
#     t = alltypes
#     df = df.sort_values(['string_col', 'timestamp_col']).reset_index(drop=True)
#     window = ibis.cumulative_window(
#         order_by=t.timestamp_col, group_by=t.string_col
#     )
#     f = getattr(t.double_col, func)
#     expr = t.projection([
#         (t.double_col - f().over(window)).name('double_col')
#     ])
#     result = expr.execute().double_col
#     expected = df.groupby(df.string_col).double_col.transform(
#         lambda c: c - getattr(c, 'cum%s' % func)()
#     )
#     tm.assert_series_equal(result, expected)


# def test_window_with_arithmetic(alltypes, df):
#     t = alltypes
#     w = ibis.window(order_by=t.timestamp_col)
#     expr = t.mutate(new_col=ibis.row_number().over(w) / 2)

#     df = df[['timestamp_col']].sort_values('timestamp_col').reset_index(
#         drop=True
#     )
#     expected = df.assign(new_col=[x / 2. for x in range(len(df))])
#     result = expr['timestamp_col', 'new_col'].execute()
#     tm.assert_frame_equal(result, expected)
