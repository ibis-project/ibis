from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis

cc = pytest.importorskip("clickhouse_connect")


@pytest.fixture(scope="module")
def diamonds(con):
    return con.table("diamonds")


@pytest.fixture(scope="module")
def batting(con):
    return con.table("batting")


@pytest.fixture(scope="module")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="module")
def time_left(con):
    return con.table("time_df1")


@pytest.fixture(scope="module")
def time_right(con):
    return con.table("time_df2")


def test_timestamp_extract_field(alltypes, assert_sql):
    t = alltypes.timestamp_col
    expr = alltypes.select(
        t.year().name("year"),
        t.month().name("month"),
        t.day().name("day"),
        t.hour().name("hour"),
        t.minute().name("minute"),
        t.second().name("second"),
    )
    assert_sql(expr)


def test_isin_notin_in_select(alltypes, assert_sql):
    values = ["foo", "bar"]
    filtered = alltypes.filter(alltypes.string_col.isin(values))
    assert_sql(filtered, "out1.sql")

    filtered = alltypes.filter(alltypes.string_col.notin(values))
    assert_sql(filtered, "out2.sql")


def test_head(alltypes):
    result = alltypes.head().execute()
    expected = alltypes.limit(5).execute()
    tm.assert_frame_equal(result, expected)


def test_limit_offset(alltypes):
    expected = alltypes.execute()

    tm.assert_frame_equal(alltypes.limit(4).execute(), expected.head(4))
    tm.assert_frame_equal(alltypes.limit(8).execute(), expected.head(8))
    tm.assert_frame_equal(
        alltypes.limit(4, offset=2).execute(),
        expected.iloc[2:6].reset_index(drop=True),
    )


def test_subquery(alltypes, df):
    t = alltypes

    expr = t.mutate(d=t.double_col).limit(100).group_by("string_col").size()
    result = expr.execute()

    result = result.sort_values("string_col").reset_index(drop=True)
    expected = (
        df.assign(d=df.double_col.fillna(0))
        .head(100)
        .groupby("string_col")
        .string_col.count()
        .rename("CountStar()")
        .reset_index()
        .sort_values("string_col")
        .reset_index(drop=True)
    )

    tm.assert_frame_equal(result, expected)


def test_simple_scalar_aggregates(alltypes, assert_sql):
    # Things like table.column.{sum, mean, ...}()
    table = alltypes

    expr = table.filter(table.int_col > 0).float_col.sum()
    assert_sql(expr)


# def test_scalar_aggregates_multiple_tables(alltypes):
#     # #740
#     table = ibis.table([('flag', 'string'),
#                         ('value', 'double')],
#                        'tbl')

#     flagged = table[table.flag == '1']
#     unflagged = table[table.flag == '0']

#     expr = flagged.value.mean() / unflagged.value.mean() - 1

#     result = ibis.clickhouse.compile(expr)
#     expected = """\
# SELECT (t0.`mean` / t1.`mean`) - 1 AS `tmp`
# FROM (
#   SELECT avg(`value`) AS `mean`
#   FROM tbl
#   WHERE `flag` = '1'
# ) t0
#   CROSS JOIN (
#     SELECT avg(`value`) AS `mean`
#     FROM tbl
#     WHERE `flag` = '0'
#   ) t1"""
#     assert result == expected

#     fv = flagged.value
#     uv = unflagged.value

#     expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())
#     result = ibis.clickhouse.compile(expr)
#     expected = """\
# SELECT t0.`tmp` - t1.`tmp` AS `tmp`
# FROM (
#   SELECT avg(`value`) / sum(`value`) AS `tmp`
#   FROM tbl
#   WHERE `flag` = '1'
# ) t0
#   CROSS JOIN (
#     SELECT avg(`value`) / sum(`value`) AS `tmp`
#     FROM tbl
#     WHERE `flag` = '0'
#   ) t1"""
#     assert result == expected


def test_table_column_unbox(alltypes, assert_sql):
    m = alltypes.float_col.sum().name("total")
    agged = alltypes.filter(alltypes.int_col > 0).group_by("string_col").aggregate([m])
    expr = agged.string_col
    assert_sql(expr)


def test_complex_array_expr_projection(alltypes, assert_sql):
    # May require finding the base table and forming a projection.
    expr = (
        alltypes.group_by("string_col")
        .aggregate([alltypes.count().name("count")])
        .string_col.cast("double")
    )
    assert_sql(expr)


@pytest.mark.parametrize(
    "expr",
    [param(ibis.now(), id="now"), param(ibis.literal(1) + ibis.literal(2), id="add")],
)
def test_scalar_exprs_no_table_refs(expr, assert_sql):
    assert_sql(expr)


def test_isnull_case_expr_rewrite_failure(alltypes, assert_sql):
    # #172, case expression that was not being properly converted into an
    # aggregation
    reduction = alltypes.string_col.isnull().ifelse(1, 0).sum()
    assert_sql(reduction)


def test_physical_table_reference_translate(alltypes, assert_sql):
    # If an expression's table leaves all reference database tables, verify
    # we translate correctlys
    assert_sql(alltypes)


def test_non_equijoin(alltypes):
    t = alltypes.limit(10)
    t2 = t.view()
    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()
    count = expr.to_pandas()
    assert count == 45


@pytest.mark.parametrize(
    "join_type",
    ["any_inner_join", "inner_join", "any_left_join", "left_join"],
)
@pytest.mark.parametrize(
    ("left_key", "right_key"), [("playerID", "playerID"), ("playerID", "awardID")]
)
def test_simple_joins(
    batting, awards_players, join_type, left_key, right_key, assert_sql
):
    t1, t2 = batting, awards_players
    pred = [t1[left_key] == t2[right_key]]
    expr = getattr(t1, join_type)(t2, pred).select(t1)
    assert_sql(expr)


def test_self_reference_simple(con, alltypes, assert_sql):
    expr = alltypes.view()
    assert_sql(expr)
    assert len(con.execute(expr))


def test_join_self_reference(con, alltypes, assert_sql):
    t1 = alltypes
    t2 = t1.view()
    expr = t1.inner_join(t2, ["id"]).select(t1)
    assert_sql(expr)
    assert len(con.execute(expr))


def test_where_simple_comparisons(con, alltypes, assert_sql):
    t1 = alltypes
    expr = t1.filter([t1.float_col > 0, t1.int_col < t1.float_col * 2])
    assert_sql(expr)
    assert len(con.execute(expr))


def test_where_with_between(con, alltypes, assert_sql):
    t = alltypes

    expr = t.filter([t.int_col > 0, t.float_col.between(0, 1)])
    assert_sql(expr)
    con.execute(expr)


def test_ifelse_use_if(con, alltypes, assert_sql):
    expr = ibis.ifelse(alltypes.float_col > 0, alltypes.int_col, alltypes.bigint_col)
    assert_sql(expr)
    con.execute(expr)


def test_filter_predicates(diamonds):
    predicates = [
        lambda x: x.color.lower().like("%de%"),
        # lambda x: x.color.lower().contains('de'),
        lambda x: x.color.lower().rlike(".*ge.*"),
    ]

    expr = diamonds
    for pred in predicates:
        expr = expr.filter(pred(expr)).select(expr)

    expr.execute()


def test_where_with_timestamp(assert_sql):
    t = ibis.table(
        [("uuid", "string"), ("ts", "timestamp"), ("search_level", "int64")],
        name="t",
    )
    expr = t.group_by(t.uuid).aggregate(min_date=t.ts.min(where=t.search_level == 1))
    assert_sql(expr)


def test_timestamp_scalar_in_filter(alltypes):
    table = alltypes

    expr = table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp("2010-01-01") + ibis.interval(weeks=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()
    expr.execute()


def test_named_from_filter_groupby(assert_sql):
    t = ibis.table([("key", "string"), ("value", "double")], name="t0")
    gb = t.filter(t.value == 42).group_by(t.key)
    sum_expr = lambda t: (t.value + 1 + 2 + 3).sum()
    expr = gb.aggregate(abc=sum_expr)
    assert_sql(expr, "out1.sql")

    expr = gb.aggregate(foo=sum_expr)
    assert_sql(expr, "out2.sql")


def test_join_with_external_table_errors(alltypes):
    external_table = ibis.table(
        [("a", "string"), ("b", "int64"), ("c", "string")], name="external"
    )

    alltypes = alltypes.mutate(b=alltypes.tinyint_col)
    expr = alltypes.inner_join(external_table, ["b"]).select(
        external_table.a, external_table.c, alltypes.id
    )

    with pytest.raises(cc.driver.exceptions.DatabaseError):
        expr.execute()

    with pytest.raises(TypeError):
        expr.execute(external_tables={"external": []})


def test_join_with_external_table(alltypes, df):
    external_df = pd.DataFrame(
        [("alpha", 1, "first"), ("beta", 2, "second"), ("gamma", 3, "third")],
        columns=["a", "b", "c"],
    )
    external_df["b"] = external_df["b"].astype("int8")

    external_table = ibis.table(
        [("a", "string"), ("b", "int64"), ("c", "string")], name="external"
    )

    alltypes = alltypes.mutate(b=alltypes.tinyint_col)
    expr = alltypes.inner_join(external_table, ["b"]).select(
        external_table.a, external_table.c, alltypes.id
    )

    result = expr.execute(external_tables={"external": external_df})
    expected = df.assign(b=df.tinyint_col).merge(external_df, on="b")[["a", "c", "id"]]

    result = result.sort_values("id").reset_index(drop=True)
    expected = expected.sort_values("id").reset_index(drop=True)

    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_asof_join(time_left, time_right):
    expr = time_left.asof_join(
        time_right,
        on=time_left["time"] >= time_right["time"],
        predicates=[
            time_left["key"] == time_right["key"],
        ],
    ).drop("time_right")
    result = expr.execute()
    expected = pd.merge_asof(
        time_left.execute(), time_right.execute(), on="time", suffixes=("", "_right")
    )
    tm.assert_frame_equal(
        result[expected.columns].fillna(pd.NA), expected.fillna(pd.NA)
    )


def test_count_name(assert_sql):
    t = ibis.table(dict(a="string", b="bool"), name="t")

    expr = t.group_by(t.a).agg(
        A=t.count(where=~t.b).fill_null(0), B=t.count(where=t.b).fill_null(0)
    )
    assert_sql(expr)


def test_array_join_in_subquery(assert_sql):
    node_view = ibis.table(dict(id="int64"), name="node_view")
    way_view = ibis.table(dict(ids="array<!int64>"), name="way_view")

    expr = node_view.id.isin(way_view.ids.unnest())
    assert_sql(expr)


def test_complex_join(assert_sql):
    t1 = ibis.table({"a": "int", "b": "int"}, name="s")
    t2 = ibis.table({"c": "int", "d": "int"}, name="t")
    t3 = t1.join(t2, t1.a == t2.c)
    q = t3.mutate(e=t3.c / (t3.a - t3.b))
    assert_sql(q)
