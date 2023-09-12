from __future__ import annotations

import itertools
from typing import Any

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.egraph import DisjointSet, EGraph, ENode, Pattern, Rewrite, Variable
from ibis.common.graph import Graph, Node
from ibis.common.grounds import Concrete
from ibis.util import promote_tuple


def test_disjoint_set():
    ds = DisjointSet()
    ds.add(1)
    ds.add(2)
    ds.add(3)
    ds.add(4)

    ds1 = DisjointSet([1, 2, 3, 4])
    assert ds == ds1
    assert ds[1] == {1}
    assert ds[2] == {2}
    assert ds[3] == {3}
    assert ds[4] == {4}

    assert ds.union(1, 2) is True
    assert ds[1] == {1, 2}
    assert ds[2] == {1, 2}
    assert ds.union(2, 3) is True
    assert ds[1] == {1, 2, 3}
    assert ds[2] == {1, 2, 3}
    assert ds[3] == {1, 2, 3}
    assert ds.union(1, 3) is False
    assert ds[4] == {4}
    assert ds != ds1
    assert 1 in ds
    assert 2 in ds
    assert 5 not in ds

    assert ds.find(1) == 1
    assert ds.find(2) == 1
    assert ds.find(3) == 1
    assert ds.find(4) == 4

    assert ds.connected(1, 2) is True
    assert ds.connected(1, 3) is True
    assert ds.connected(1, 4) is False

    # test mapping api get
    assert ds.get(1) == {1, 2, 3}
    assert ds.get(4) == {4}
    assert ds.get(5) is None
    assert ds.get(5, 5) == 5
    assert ds.get(5, default=5) == 5

    # test mapping api keys
    assert set(ds.keys()) == {1, 2, 3, 4}
    assert set(ds) == {1, 2, 3, 4}

    # test mapping api values
    assert tuple(ds.values()) == ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {4})

    # test mapping api items
    assert tuple(ds.items()) == (
        (1, {1, 2, 3}),
        (2, {1, 2, 3}),
        (3, {1, 2, 3}),
        (4, {4}),
    )

    # check that the disjoint set doesn't get corrupted by adding an existing element
    ds.verify()
    ds.add(1)
    ds.verify()

    with pytest.raises(RuntimeError, match="DisjointSet is corrupted"):
        ds._parents[1] = 1
        ds._classes[1] = {1}
        ds.verify()


class PatternNamespace:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        klass = getattr(self.module, name)

        def pattern(*args):
            return Pattern(klass, args)

        return pattern


p = PatternNamespace(ops)

one = ibis.literal(1)
two = one * 2
two_ = one + one
two__ = ibis.literal(2)
three = one + two
six = three * two_
seven = six + 1
seven_ = seven * 1
eleven = seven_ + 4

a, b, c = Variable("a"), Variable("b"), Variable("c")
x, y, z = Variable("x"), Variable("y"), Variable("z")


class Base(Concrete, Node):
    def __class_getitem__(self, args):
        args = promote_tuple(args)
        return Pattern(self, args)


class Lit(Base):
    value: Any


class Add(Base):
    x: Any
    y: Any


class Mul(Base):
    x: Any
    y: Any


def test_enode():
    node = ENode(1, (2, 3))
    assert node == ENode(1, (2, 3))
    assert node != ENode(1, [2, 4])
    assert node != ENode(1, [2, 3, 4])
    assert node != ENode(1, [2])
    assert hash(node) == hash(ENode(1, (2, 3)))
    assert hash(node) != hash(ENode(1, (2, 4)))

    with pytest.raises(AttributeError, match="immutable"):
        node.head = 2
    with pytest.raises(AttributeError, match="immutable"):
        node.args = (2, 3)


class MyNode(Concrete, Node):
    a: int
    b: int
    c: str


def test_enode_roundtrip():
    # create e-node from node
    node = MyNode(a=1, b=2, c="3")
    enode = ENode.from_node(node)
    assert enode == ENode(MyNode, (1, 2, "3"))

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


class MySecondNode(Concrete, Node):
    a: int
    b: tuple[int, ...]


def test_enode_roundtrip_with_variadic_arg():
    # create e-node from node
    node = MySecondNode(a=1, b=(2, 3))
    enode = ENode.from_node(node)
    assert enode == ENode(MySecondNode, (1, (2, 3)))

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


class MyInt(Concrete, Node):
    value: int


class MyThirdNode(Concrete, Node):
    a: int
    b: tuple[MyInt, ...]


def test_enode_roundtrip_with_nested_arg():
    # create e-node from node
    node = MyThirdNode(a=1, b=(MyInt(value=2), MyInt(value=3)))
    enode = ENode.from_node(node)
    assert enode == ENode(MyThirdNode, (1, (ENode(MyInt, (2,)), ENode(MyInt, (3,)))))

    # reconstruct node from e-node
    node_ = enode.to_node()
    assert node_ == node


class MyFourthNode(Concrete, Node):
    pass


class MyLit(MyFourthNode):
    value: int


class MyAdd(MyFourthNode):
    a: MyFourthNode
    b: MyFourthNode


class MyMul(MyFourthNode):
    a: MyFourthNode
    b: MyFourthNode


def test_disjoint_set_with_enode():
    # number postfix highlights the depth of the node
    one = MyLit(value=1)
    two = MyLit(value=2)
    two1 = MyAdd(a=one, b=one)
    three1 = MyAdd(a=one, b=two)
    six2 = MyMul(a=three1, b=two1)
    seven2 = MyAdd(a=six2, b=one)

    # expected enodes postfixed with an underscore
    one_ = ENode(MyLit, (1,))
    two_ = ENode(MyLit, (2,))
    three_ = ENode(MyLit, (3,))
    two1_ = ENode(MyAdd, (one_, one_))
    three1_ = ENode(MyAdd, (one_, two_))
    six2_ = ENode(MyMul, (three1_, two1_))
    seven2_ = ENode(MyAdd, (six2_, one_))

    enode = ENode.from_node(seven2)
    assert enode == seven2_

    assert enode.to_node() == seven2

    ds = DisjointSet()
    for enode in Graph.from_bfs(seven2_):
        ds.add(enode)
        assert ds.find(enode) == enode

    # merging identical nodes should return False
    assert ds.union(three1_, three1_) is False
    assert ds.find(three1_) == three1_
    assert ds[three1_] == {three1_}

    # now merge a (1 + 2) and (3) nodes, but first add `three_` to the set
    ds.add(three_)
    assert ds.union(three1_, three_) is True
    assert ds.find(three1_) == three1_
    assert ds.find(three_) == three1_
    assert ds[three_] == {three_, three1_}


def test_pattern():
    Pattern._counter = itertools.count()

    p = Pattern(ops.Literal, (1, dt.int8))
    assert p.head == ops.Literal
    assert p.args == (1, dt.int8)
    assert p.name is None

    p = "name" @ Pattern(ops.Literal, (1, dt.int8))
    assert p.head == ops.Literal
    assert p.args == (1, dt.int8)
    assert p.name == "name"


def test_pattern_flatten():
    # using auto-generated names
    one = Pattern(ops.Literal, (1, dt.int8))
    two = Pattern(ops.Literal, (2, dt.int8))
    three = Pattern(ops.Add, (one, two))

    result = dict(three.flatten())
    expected = {
        Variable(0): Pattern(ops.Add, (Variable(1), Variable(2))),
        Variable(2): Pattern(ops.Literal, (2, dt.int8)),
        Variable(1): Pattern(ops.Literal, (1, dt.int8)),
    }
    assert result == expected

    # using user-provided names which helps capturing variables
    one = "one" @ Pattern(ops.Literal, (1, dt.int8))
    two = "two" @ Pattern(ops.Literal, (2, dt.int8))
    three = "three" @ Pattern(ops.Add, (one, two))

    result = tuple(three.flatten())
    expected = (
        (Variable("one"), Pattern(ops.Literal, (1, dt.int8))),
        (Variable("two"), Pattern(ops.Literal, (2, dt.int8))),
        (Variable("three"), Pattern(ops.Add, (Variable("one"), Variable("two")))),
    )
    assert result == expected


def test_egraph_match_simple():
    eg = EGraph()
    eg.add(eleven.op())

    pat = p.Multiply(a, "lit" @ p.Literal(1, dt.int8))
    res = eg.match(pat)

    enode = ENode.from_node(seven_.op())
    matches = res[enode]
    assert matches["a"] == ENode.from_node(seven.op())
    assert matches["lit"] == ENode.from_node(one.op())


def test_egraph_match_wrong_argnum():
    two = one + one
    four = two + two

    eg = EGraph()
    eg.add(four.op())

    # here we have an extra `2` among the literal's arguments
    pat = p.Add(a, p.Add(p.Literal(1, dt.int8, 2), b))
    res = eg.match(pat)

    assert res == {}

    pat = p.Add(a, p.Add(p.Literal(1, dt.int8), b))
    res = eg.match(pat)

    expected = {
        ENode.from_node(four.op()): {
            0: ENode.from_node(four.op()),
            1: ENode.from_node(two.op()),
            2: ENode.from_node(one.op()),
            "a": ENode.from_node(two.op()),
            "b": ENode.from_node(one.op()),
        }
    }
    assert res == expected


def test_egraph_match_nested():
    node = eleven.op()
    enode = ENode.from_node(node)

    eg = EGraph()
    eg.add(enode)

    result = eg.match(p.Multiply(a, p.Literal(1, b)))
    matched = ENode.from_node(seven_.op())

    expected = {
        matched: {
            0: matched,
            1: ENode.from_node(one.op()),
            "a": ENode.from_node(seven.op()),
            "b": dt.int8,
        }
    }
    assert result == expected


def test_egraph_apply_nested():
    node = eleven.op()
    enode = ENode.from_node(node)

    eg = EGraph()
    eg.add(enode)

    r3 = p.Multiply(a, p.Literal(1, dt.int8)) >> a
    eg.apply(r3)

    result = eg.extract(seven_.op())
    expected = seven.op()
    assert result == expected


def test_egraph_extract_simple():
    eg = EGraph()
    eg.add(eleven.op())

    res = eg.extract(one.op())
    assert res == one.op()


def test_egraph_extract_minimum_cost():
    eg = EGraph()
    eg.add(two.op())  # 1 * 2
    eg.add(two_.op())  # 1 + 1
    eg.add(two__.op())  # 2
    assert eg.extract(two.op()) == two.op()

    eg.union(two.op(), two_.op())
    assert eg.extract(two.op()) in {two.op(), two_.op()}

    eg.union(two.op(), two__.op())
    assert eg.extract(two.op()) == two__.op()

    eg.union(two.op(), two__.op())
    assert eg.extract(two.op()) == two__.op()


def test_egraph_rewrite_to_variable():
    eg = EGraph()
    eg.add(eleven.op())

    # rule with a variable on the right-hand side
    rule = Rewrite(p.Multiply(a, "lit" @ p.Literal(1, dt.int8)), a)
    eg.apply(rule)
    assert eg.equivalent(seven_.op(), seven.op())


def test_egraph_rewrite_to_constant_raises():
    node = (one * 0).op()

    eg = EGraph()
    eg.add(node)

    # rule with a constant on the right-hand side
    with pytest.raises(TypeError):
        Rewrite(p.Multiply(a, "lit" @ p.Literal(0, dt.int8)), 0)


def test_egraph_rewrite_to_pattern():
    eg = EGraph()
    eg.add(three.op())

    # rule with a pattern on the right-hand side
    rule = Rewrite(p.Multiply(a, "lit" @ p.Literal(2, dt.int8)), p.Add(a, a))
    eg.apply(rule)
    assert eg.equivalent(two.op(), two_.op())


def test_egraph_rewrite_dynamic():
    def applier(egraph, match, a, mul, times):
        return ENode(ops.Add, (a, a))

    node = (one * 2).op()

    eg = EGraph()
    eg.add(node)

    # rule with a dynamic pattern on the right-hand side
    rule = Rewrite(
        "mul" @ p.Multiply(a, p.Literal(Variable("times"), dt.int8)), applier
    )
    eg.apply(rule)

    assert eg.extract(node) in {two.op(), two_.op()}


def test_egraph_rewrite_commutative():
    rules = [
        Mul[a, b] >> Mul[b, a],
        Mul[a, Lit[1]] >> a,
    ]
    node = Mul(Lit(2), Mul(Lit(1), Lit(3)))
    expected = {Mul(Lit(2), Lit(3)), Mul(Lit(3), Lit(2))}

    egraph = EGraph()
    egraph.add(node)
    egraph.run(rules, 200)
    best = egraph.extract(node)

    assert best in expected


@pytest.mark.parametrize(
    ("node", "expected"),
    [(Mul(Lit(0), Lit(42)), Lit(0)), (Add(Lit(0), Mul(Lit(1), Lit(2))), Lit(2))],
)
def test_egraph_rewrite(node, expected):
    rules = [
        Add[a, b] >> Add[b, a],
        Mul[a, b] >> Mul[b, a],
        Add[a, Lit[0]] >> a,
        Mul[a, Lit[0]] >> Lit[0],
        Mul[a, Lit[1]] >> a,
    ]
    egraph = EGraph()
    egraph.add(node)
    egraph.run(rules, 100)
    best = egraph.extract(node)

    assert best == expected


def is_equal(a, b, rules, iters=7):
    egraph = EGraph()
    id_a = egraph.add(a)
    id_b = egraph.add(b)
    egraph.run(rules, iters)
    return egraph.equivalent(id_a, id_b)


def test_math_associate_adds(benchmark):
    math_rules = [Add[a, b] >> Add[b, a], Add[a, Add[b, c]] >> Add[Add[a, b], c]]

    expr_a = Add(1, Add(2, Add(3, Add(4, Add(5, Add(6, 7))))))
    expr_b = Add(7, Add(6, Add(5, Add(4, Add(3, Add(2, 1))))))
    assert is_equal(expr_a, expr_b, math_rules, iters=500)

    expr_a = Add(6, Add(Add(1, 5), Add(0, Add(4, Add(2, 3)))))
    expr_b = Add(6, Add(Add(4, 5), Add(Add(0, 2), Add(3, 1))))
    assert is_equal(expr_a, expr_b, math_rules, iters=500)

    benchmark(is_equal, expr_a, expr_b, math_rules, iters=500)


def replace_add(egraph, enode, **kwargs):
    node = egraph.extract(enode)
    enode = egraph.add(node)
    return enode


def test_dynamic_rewrite():
    rules = [Rewrite(Add[x, Mul[z, y]], replace_add)]
    node = Add(1, Mul(2, 3))

    egraph = EGraph()
    egraph.add(node)
    egraph.run(rules, 100)
    best = egraph.extract(node)

    assert best == node


def test_dynamic_condition():
    pass
