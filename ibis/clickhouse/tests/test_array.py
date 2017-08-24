# @pytest.fixture
# def array_types(con):
#     return con.table('array_types')


# def test_array_length(array_types):
#     expr = array_types.projection([
#         array_types.x.length().name('x_length'),
#         array_types.y.length().name('y_length'),
#         array_types.z.length().name('z_length'),
#     ])
#     result = expr.execute()
#     expected = pd.DataFrame({
#         'x_length': [3, 2, 2, 3, 3, 4],
#         'y_length': [3, 2, 2, 3, 3, 4],
#         'z_length': [3, 2, 2, 0, None, 4],
#     })

#     tm.assert_frame_equal(result, expected)


# def test_array_schema(array_types):
#     assert array_types.x.type() == dt.Array(dt.int64)
#     assert array_types.y.type() == dt.Array(dt.string)
#     assert array_types.z.type() == dt.Array(dt.double)


# def test_array_collect(array_types):
#     expr = array_types.group_by(
#         array_types.grouper
#     ).aggregate(collected=array_types.scalar_column.collect())
#     result = expr.execute().sort_values('grouper').reset_index(drop=True)
#     expected = pd.DataFrame({
#         'grouper': list('abc'),
#         'collected': [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]],
#     })[['grouper', 'collected']]
#     tm.assert_frame_equal(result, expected, check_column_type=False)


# @pytest.mark.parametrize(
#     ['start', 'stop'],
#     [
#         (1, 3),
#         (1, 1),
#         (2, 3),
#         (2, 5),

#         (None, 3),
#         (None, None),
#         (3, None),

#         # negative slices are not supported
#         pytest.mark.xfail(
#             (-3, None),
#             raises=ValueError,
#             reason='Negative slicing not supported'
#         ),
#         pytest.mark.xfail(
#             (None, -3),
#             raises=ValueError,
#             reason='Negative slicing not supported'
#         ),
#         pytest.mark.xfail(
#             (-3, -1),
#             raises=ValueError,
#             reason='Negative slicing not supported'
#         ),
#     ]
# )
# def test_array_slice(array_types, start, stop):
#     expr = array_types[array_types.y[start:stop].name('sliced')]
#     result = expr.execute()
#     expected = pd.DataFrame({
#         'sliced': array_types.y.execute().map(lambda x: x[start:stop])
#     })
#     tm.assert_frame_equal(result, expected)


# @pytest.mark.parametrize('index', [1, 3, 4, 11])
# def test_array_index(array_types, index):
#     expr = array_types[array_types.y[index].name('indexed')]
#     result = expr.execute()
#     expected = pd.DataFrame({
#         'indexed': array_types.y.execute().map(
#             lambda x: x[index] if index < len(x) else None
#         )
#     })
#     tm.assert_frame_equal(result, expected)


# @pytest.mark.parametrize('n', [1, 3, 4, 7, -2])
# @pytest.mark.parametrize('mul', [lambda x, n: x * n, lambda x, n: n * x])
# def test_array_repeat(array_types, n, mul):
#     expr = array_types.projection([mul(array_types.x, n).name('repeated')])
#     result = expr.execute()
#     expected = pd.DataFrame({
#         'repeated': array_types.x.execute().map(lambda x, n=n: mul(x, n))
#     })
#     tm.assert_frame_equal(result, expected)


# @pytest.mark.parametrize('catop', [lambda x, y: x + y, lambda x, y: y + x])
# def test_array_concat(array_types, catop):
#     t = array_types
#     x, y = t.x.cast('array<string>').name('x'), t.y
#     expr = t.projection([catop(x, y).name('catted')])
#     result = expr.execute()
#     tuples = t.projection([x, y]).execute().itertuples(index=False)
#     expected = pd.DataFrame({'catted': [catop(i, j) for i, j in tuples]})
#     tm.assert_frame_equal(result, expected)


# def test_array_concat_mixed_types(array_types):
#     with pytest.raises(TypeError):
#         array_types.x + array_types.x.cast('array<double>')
