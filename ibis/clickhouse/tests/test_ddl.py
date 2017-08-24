# @pytest.fixture
# def t(con, guid):
#     con.raw_sql(
#         """
#         CREATE TABLE "{}" (
#           id SERIAL PRIMARY KEY,
#           name TEXT
#         )
#         """.format(guid)
#     )
#     return con.table(guid)


# @pytest.fixture
# def s(con, t, guid, guid2):
#     assert t.op().name == guid
#     assert t.op().name != guid2

#     con.raw_sql(
#         """
#         CREATE TABLE "{}" (
#           id SERIAL PRIMARY KEY,
#           left_t_id INTEGER REFERENCES "{}",
#           cost DOUBLE PRECISION
#         )
#         """.format(guid2, guid)
#     )
#     return con.table(guid2)


# @pytest.fixture
# def trunc(con, guid):
#     con.raw_sql(
#         """
#         CREATE TABLE "{}" (
#           id SERIAL PRIMARY KEY,
#           name TEXT
#         )
#         """.format(guid)
#     )
#     con.raw_sql(
#         """INSERT INTO "{}" (name) VALUES ('a'), ('b'), ('c')""".format(
#             guid
#         )
#     )
#     return con.table(guid)


# def test_create_table_from_expr(con, trunc, guid2):
#     con.create_table(guid2, expr=trunc)
#     t = con.table(guid2)
#     assert list(t.name.execute()) == list('abc')


# def test_truncate_table(con, trunc):
#     assert list(trunc.name.execute()) == list('abc')
#     con.truncate_table(trunc.op().name)
#     assert not len(trunc.execute())


# def test_head(con):
#     t = con.table('functional_alltypes')
#     result = t.head().execute()
#     expected = t.limit(5).execute()
#     tm.assert_frame_equal(result, expected)
