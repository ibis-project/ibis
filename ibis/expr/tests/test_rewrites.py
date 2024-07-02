from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
from ibis.expr.rewrites import (
    rebase_predicates_on_table_underlying_join_reference,
    rewrite_join_chain_for_semi_anti_join,
    simplify,
)


@pytest.fixture
def schema():
    return {
        "bool_col": "boolean",
        "int_col": "int64",
        "float_col": "float64",
        "string_col": "string",
    }


@pytest.fixture
def t(schema):
    return ibis.table(
        name="t",
        schema=schema,
    )


def test_simplify_full_reprojection(t):
    t1 = t.select(t)
    t1_opt = simplify(t1.op())
    assert t1_opt == t.op()


def test_simplify_subsequent_field_selections(t):
    t1 = t.select(t.bool_col, t.int_col, t.float_col)
    assert t1.op() == ops.Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
        },
    )

    t2 = t1.select(t1.bool_col, t1.int_col)
    t2_opt = simplify(t2.op())
    assert t2_opt == ops.Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
        },
    )

    t3 = t2.select(t2.bool_col)
    t3_opt = simplify(t3.op())
    assert t3_opt == ops.Project(parent=t, values={"bool_col": t.bool_col})


def test_simplify_subsequent_value_selections(t):
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    t2_opt = simplify(t2.op())
    assert t2_opt == ops.Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": t.float_col * 3,
        },
    )

    t3 = t2.select(
        t2.bool_col,
        t2.int_col,
        float_col=t2.float_col * 2,
        another_col=t1.float_col - 1,
    )
    t3_opt = simplify(t3.op())
    assert t3_opt == ops.Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": (t.float_col * 3) * 2,
            "another_col": (t.float_col * 3) - 1,
        },
    )


def test_simplify_subsequent_filters(t):
    f1 = t.filter(t.bool_col)
    f2 = f1.filter(t.int_col > 0)
    f2_opt = simplify(f2.op())
    assert f2_opt == ops.Filter(t, predicates=[t.bool_col, t.int_col > 0])


def test_simplify_project_filter_project(t):
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.filter(t1.bool_col)
    t3 = t2.filter(t2.int_col > 0)
    t4 = t3.select(t3.bool_col, t3.int_col)

    filt = ops.Filter(parent=t, predicates=[~t.bool_col, t.int_col + 1 > 0]).to_expr()
    proj = ops.Project(
        parent=filt, values={"bool_col": ~filt.bool_col, "int_col": filt.int_col + 1}
    ).to_expr()

    t4_opt = simplify(t4.op())
    assert t4_opt == proj.op()


@pytest.fixture
def t2(schema):
    return ibis.table(
        name="t2",
        schema=schema,
    )


@pytest.fixture(params=[["int_col"]])
def predicates(request):
    return request.param


@pytest.fixture(params=["anti", "semi"])
def semi_or_anti_join(t, t2, predicates, request):
    return t.join(
        t2,
        predicates=predicates,
        how=request.param,
    )


def test_rebase_predicates_on_table_underlying_join_reference(semi_or_anti_join):
    join_chain = semi_or_anti_join.op()
    for join_link in join_chain.rest:
        for pred in join_link.predicates:
            table_underlying_join_table = pred.left.rel.parent
            pred = pred.replace(rebase_predicates_on_table_underlying_join_reference)
            assert pred.left.rel == table_underlying_join_table


def assert_on_filter_replacing_join_chain(join_chain, filter_):
    top_filters = filter_.find_topmost(ops.Filter)
    assert top_filters
    top_filter = top_filters[0]
    assert isinstance(top_filter.parent, ops.Relation)

    top_exists_subqueries = filter_.find_topmost(ops.ExistsSubquery)
    assert top_exists_subqueries
    top_exists_subquery = top_exists_subqueries[0]

    # Check for the semi-anti-join-link
    semi_anti_links = [
        link for link in join_chain.find(ops.JoinLink) if link.how in {"anti", "semi"}
    ]
    assert len(semi_anti_links) == 1
    link = semi_anti_links[0]

    # Check for the `ExistsSubquery` and `Not` ops.
    assert top_filter.predicates
    if link.how == "semi":
        assert top_filter.predicates[0] == top_exists_subquery
    elif link.how == "anti":
        assert isinstance(top_filter.predicates[0], ops.Not)
        assert top_filter.predicates[0].arg == top_exists_subquery

    # Compare the number of predicates, before and after replace.
    assert len(top_exists_subquery.rel.predicates) == len(link.predicates)

    # Check that left and right tables of `filter_` is inherited
    # correctly from `join_chain`.
    field_below_top_exists_subquery = top_exists_subquery.rel
    assert isinstance(field_below_top_exists_subquery, ops.Filter)
    left_right_table_types = sorted(
        [
            type(top_filter.parent).__name__,
            type(field_below_top_exists_subquery.parent).__name__,
        ]
    )
    non_semi_anti_links = [
        link
        for link in join_chain.find(ops.JoinLink)
        if link.how not in {"anti", "semi"}
    ]
    assert left_right_table_types == (
        ["JoinChain", "UnboundTable"]
        if non_semi_anti_links
        else ["UnboundTable", "UnboundTable"]
    )

    # Check if each predicate contains one field based on the
    # left-table and another based on the right-table.
    for pred in field_below_top_exists_subquery.predicates:
        assert {pred.left.rel, pred.right.rel} == {
            top_filter.parent,
            field_below_top_exists_subquery.parent,
        }


def test_rewrite_join_chain_for_semi_anti_join(semi_or_anti_join):
    join_chain = semi_or_anti_join.op()
    join_links = join_chain.find(ops.JoinLink)
    assert len(join_links) == 1

    filter_ = join_chain.replace(rewrite_join_chain_for_semi_anti_join)
    assert not filter_.find(ops.JoinChain)
    assert not filter_.find(ops.JoinLink)
    assert not filter_.find(ops.JoinReference)

    assert_on_filter_replacing_join_chain(join_chain, filter_)


@pytest.fixture
def t3(schema):
    return ibis.table(name="t3", schema=schema)


@pytest.fixture
def inner_join(t, t2, predicates):
    return t.join(t2, predicates=predicates, how="inner")


@pytest.fixture(params=["anti", "semi"])
def inner_join_dot_semi_anti_join(t3, inner_join, predicates, request):
    return inner_join.join(
        t3,
        predicates=predicates,
        how=request.param,
    )


def test_rewrite_join_chain_for_semi_anti_join_for_inner_join_dot_semi_anti_join(
    inner_join_dot_semi_anti_join,
):
    join_chain = inner_join_dot_semi_anti_join.op()
    join_links = join_chain.find(ops.JoinLink)
    assert len(join_links) == 2

    filter_ = join_chain.replace(rewrite_join_chain_for_semi_anti_join)

    assert len(filter_.find(ops.JoinChain)) == 1
    assert len(filter_.find(ops.JoinLink)) == 1
    assert len(filter_.find(ops.JoinReference)) == 2

    assert_on_filter_replacing_join_chain(join_chain, filter_)


@pytest.fixture(params=["anti", "semi"])
def semi_anti_join_dot_inner_join(t3, inner_join, predicates, request):
    return t3.join(
        inner_join,
        predicates=predicates,
        how=request.param,
    )


def test_rewrite_join_chain_for_semi_anti_join_for_semi_anti_join_dot_inner_join(
    semi_anti_join_dot_inner_join,
):
    join_chain = semi_anti_join_dot_inner_join.op()
    assert len(join_chain.find(ops.JoinChain)) == 2
    join_links = join_chain.find(ops.JoinLink)
    assert len(join_links) == 2

    filter_ = join_chain.replace(rewrite_join_chain_for_semi_anti_join)

    assert len(filter_.find(ops.JoinChain)) == 1
    assert len(filter_.find(ops.JoinLink)) == 1
    assert len(filter_.find(ops.JoinReference)) == 2

    assert_on_filter_replacing_join_chain(join_chain, filter_)
