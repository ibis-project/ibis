from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
from ibis.expr.types.relations import DerefMap

t = ibis.table(
    [
        ("int_col", "int32"),
        ("double_col", "double"),
        ("string_col", "string"),
    ],
    name="t",
)


def dereference_expect(expected):
    return {k.op(): v.op() for k, v in expected.items()}


def test_dereference_project():
    p = t.select([t.int_col, t.double_col])

    mapping = DerefMap.from_targets([p.op()])

    assert tuple(
        mapping.dereference(
            p.int_col.op(), p.double_col.op(), t.int_col.op(), t.double_col.op()
        )
    ) == (
        p.int_col.op(),
        p.double_col.op(),
        p.int_col.op(),
        p.double_col.op(),
    )


@pytest.mark.parametrize("column", ["int_col", "double_col", "string_col"])
def test_dereference_mapping_self_reference(column):
    v = t.view()

    mapping = DerefMap.from_targets([v.op()])
    assert tuple(mapping.dereference(v[column].op())) == (v[column].op(),)


customers = ibis.table({"customer_id": "int64", "balance": "int64"}, name="customers")
orders = ibis.table({"customer_id": "int64", "quantity": "int64"}, name="orders")
refunds = ibis.table({"customer_id": "int64", "amount": "int64"}, name="refunds")


def refunded(customer_id):
    return refunds.filter(refunds.customer_id == customer_id).amount.sum()


def test_dereference_correlated_subquery_to_join_reference():
    chain = customers.join(orders, "customer_id").op()
    customers_ref, orders_ref = chain.first, chain.rest[0].table
    mapping = DerefMap.from_targets([customers_ref, orders_ref])

    (result,) = mapping.dereference(refunded(customers.customer_id).op())

    assert result == refunded(customers_ref.to_expr().customer_id).op()


def test_dereference_correlated_subquery_to_join_chain():
    chain = customers.join(orders, "customer_id").op()
    mapping = DerefMap.from_targets([chain])

    (result,) = mapping.dereference(refunded(customers.customer_id).op())

    assert result == refunded(ops.Field(chain, "customer_id").to_expr()).op()


def test_dereference_subquery_keeps_its_own_references():
    chain = customers.join(orders, "customer_id").op()
    mapping = DerefMap.from_targets([chain.first, chain.rest[0].table])
    # `customers` is the input of the filter, not a reference to the join
    value = customers.filter(customers.balance > 10).balance.sum().op()

    (result,) = mapping.dereference(value)

    assert result == value


def test_dereference_correlated_subquery_to_non_join_unchanged():
    # the compiler can merge the filter into the outer query, so the subquery
    # keeps referencing `customers`
    filtered = customers.filter(customers.balance > 0)
    mapping = DerefMap.from_targets([filtered.op()])
    value = refunded(customers.customer_id).op()

    (result,) = mapping.dereference(value)

    assert result == value
