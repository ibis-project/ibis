from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.expr.rewrites import simplify
from ibis.expr.tests.test_newrels import join_tables

# Place to collect esoteric expression analysis bugs and tests


def test_rewrite_join_projection_without_other_ops(con):
    # Star schema with fact table
    table = con.table("star1")
    table2 = con.table("star2")
    table3 = con.table("star3")

    filtered = table[table["f"] > 0]

    pred1 = table["foo_id"] == table2["foo_id"]
    pred2 = filtered["bar_id"] == table3["bar_id"]

    j1 = filtered.left_join(table2, [pred1])
    j2 = j1.inner_join(table3, [pred2])
    # Project out the desired fields
    view = j2[[filtered, table2["value1"], table3["value2"]]]

    with join_tables(filtered, table2, table3) as (r1, r2, r3):
        # Construct the thing we expect to obtain
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink(
                    how="left",
                    table=r2,
                    predicates=[r1["foo_id"] == r2["foo_id"]],
                ),
                ops.JoinLink(
                    how="inner",
                    table=r3,
                    predicates=[r1["bar_id"] == r3["bar_id"]],
                ),
            ],
            values={
                "c": r1.c,
                "f": r1.f,
                "foo_id": r1.foo_id,
                "bar_id": r1.bar_id,
                "value1": r2.value1,
                "value2": r3.value2,
            },
        )
        assert view.op() == expected


def test_multiple_join_deeper_reference():
    # Join predicates down the chain might reference one or more root
    # tables in the hierarchy.
    table1 = ibis.table({"key1": "string", "key2": "string", "value1": "double"})
    table2 = ibis.table({"key3": "string", "value2": "double"})
    table3 = ibis.table({"key4": "string", "value3": "double"})

    joined = table1.inner_join(table2, [table1["key1"] == table2["key3"]])
    joined2 = joined.inner_join(table3, [table1["key2"] == table3["key4"]])

    # it works, what more should we test here?
    repr(joined2)


def test_filter_on_projected_field(con):
    # See #173. Impala and other SQL engines do not allow filtering on a
    # just-created alias in a projection
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")
    customer = con.table("tpch_customer")
    orders = con.table("tpch_orders")

    fields_of_interest = [
        customer,
        region.r_name.name("region"),
        orders.o_totalprice.name("amount"),
        orders.o_orderdate.cast("timestamp").name("odate"),
    ]

    all_join = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)
    )

    tpch = all_join[fields_of_interest]

    # Correlated subquery, yikes!
    t2 = tpch.view()
    conditional_avg = t2[(t2.region == tpch.region)].amount.mean()

    # `amount` is part of the projection above as an aliased field
    amount_filter = tpch.amount > conditional_avg

    result = tpch.filter([amount_filter])

    # Now then! Predicate pushdown here is inappropriate, so we check that
    # it didn't occur.
    assert isinstance(result.op(), ops.Filter)
    assert result.op().parent == tpch.op()


def test_join_predicate_from_derived_raises():
    # Join predicate references a derived table, but we can salvage and
    # rewrite it to get the join semantics out
    # see ibis #74
    table = ibis.table([("c", "int32"), ("f", "double"), ("g", "string")], "foo_table")

    table2 = ibis.table([("key", "string"), ("value", "double")], "bar_table")

    filter_pred = table["f"] > 0
    table3 = table[filter_pred]

    with pytest.raises(com.IntegrityError, match="they belong to another relation"):
        # TODO(kszucs): could be smarter actually and rewrite the predicate
        # to contain the conditions from the filter
        table.inner_join(table2, [table3["g"] == table2["key"]])


def test_bad_join_predicate_raises():
    table = ibis.table([("c", "int32"), ("f", "double"), ("g", "string")], "foo_table")
    table2 = ibis.table([("key", "string"), ("value", "double")], "bar_table")
    table3 = ibis.table([("key", "string"), ("value", "double")], "baz_table")

    with pytest.raises(com.IntegrityError):
        table.inner_join(table2, [table["g"] == table3["key"]])


def test_filter_self_join():
    # GH #667
    purchases = ibis.table(
        [
            ("region", "string"),
            ("kind", "string"),
            ("user", "int64"),
            ("amount", "double"),
        ],
        "purchases",
    )

    metric = purchases.amount.sum().name("total")
    agged = purchases.group_by(["region", "kind"]).aggregate(metric)
    assert agged.op() == ops.Aggregate(
        parent=purchases,
        groups={"region": purchases.region, "kind": purchases.kind},
        metrics={"total": purchases.amount.sum()},
    )

    left = agged[agged.kind == "foo"]
    right = agged[agged.kind == "bar"]
    assert left.op() == ops.Filter(
        parent=agged,
        predicates=[agged.kind == "foo"],
    )
    assert right.op() == ops.Filter(
        parent=agged,
        predicates=[agged.kind == "bar"],
    )

    cond = left.region == right.region
    joined = left.join(right, cond)
    metric = (left.total - right.total).name("diff")
    what = [left.region, metric]
    projected = joined.select(what)

    with join_tables(left, right) as (r1, r2):
        join = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.region == r2.region]),
            ],
            values={
                "region": r1.region,
                "diff": r1.total - r2.total,
            },
        )
        assert projected.op() == join


def test_is_ancestor_analytic():
    x = ibis.table(ibis.schema([("col", "int32")]), "x")
    with_filter_col = x[x.columns + [ibis.null().name("filter")]]
    filtered = with_filter_col[with_filter_col["filter"].isnull()]
    subquery = filtered[filtered.columns]

    with_analytic = subquery[subquery.columns + [subquery.count().name("analytic")]]

    assert not subquery.op().equals(with_analytic.op())


# Pr 2635
def test_mutation_fusion_no_overwrite():
    """Test fusion with chained mutation that doesn't overwrite existing
    columns."""
    t = ibis.table(ibis.schema([("col", "int32")]), "t")

    result = t
    result = result.mutate(col1=t["col"] + 1)
    result = result.mutate(col2=t["col"] + 2)
    result = result.mutate(col3=t["col"] + 3)

    simplified = simplify(result.op())
    assert simplified == ops.Project(
        parent=t,
        values={
            "col": t["col"],
            "col1": t["col"] + 1,
            "col2": t["col"] + 2,
            "col3": t["col"] + 3,
        },
    )


# Pr 2635
def test_mutation_fusion_overwrite():
    """Test fusion with chained mutation that overwrites existing columns."""
    t = ibis.table(ibis.schema([("col", "int32")]), "t")

    result = t

    result = result.mutate(col1=t["col"] + 1)
    result = result.mutate(col2=t["col"] + 2)
    result = result.mutate(col3=t["col"] + 3)
    result = result.mutate(col=t["col"] - 1)

    with pytest.raises(com.IntegrityError):
        # unable to dereference the column since result doesn't contain it anymore
        result.mutate(col4=t["col"] + 4)

    simplified = simplify(result.op())
    assert simplified == ops.Project(
        parent=t,
        values={
            "col": t["col"] - 1,
            "col1": t["col"] + 1,
            "col2": t["col"] + 2,
            "col3": t["col"] + 3,
        },
    )


# Pr 2635
def test_select_filter_mutate_fusion():
    """Test fusion with filter followed by mutation on the same input."""

    t = ibis.table(ibis.schema([("col", "float32")]), "t")

    t1 = t[["col"]]
    assert t1.op() == ops.Project(parent=t, values={"col": t.col})

    t2 = t1[t1["col"].isnan()]
    assert t2.op() == ops.Filter(parent=t1, predicates=[t1.col.isnan()])

    t3 = t2.mutate(col=t2["col"].cast("int32"))
    assert t3.op() == ops.Project(parent=t2, values={"col": t2.col.cast("int32")})

    # create the expected expression
    filt = ops.Filter(parent=t, predicates=[t.col.isnan()]).to_expr()
    proj = ops.Project(parent=filt, values={"col": filt.col.cast("int32")}).to_expr()

    t3_opt = simplify(t3.op()).to_expr()
    assert t3_opt.equals(proj)


def test_agg_selection_does_not_share_roots():
    t = ibis.table(dict(a="string"), name="t")
    s = ibis.table(dict(b="float64"), name="s")
    gb = t.group_by("a")
    n = s.count()

    with pytest.raises(com.IntegrityError, match=" they belong to another relation"):
        gb.aggregate(n=n)
