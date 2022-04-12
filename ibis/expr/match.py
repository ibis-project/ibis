from __future__ import annotations

import itertools
from typing import Generic, Iterable, SupportsFloat, SupportsInt, TypeVar

import matchpy
import toolz
from matchpy import (
    Arity,
    CustomConstraint,
    Operation,
    Pattern,
    ReplacementRule,
    Symbol,
    Wildcard,
)

T = TypeVar("T")


class Literal(Symbol, Generic[T]):
    def __init__(self, value: T) -> None:
        super().__init__(str(value))
        self.value = value


class IntLiteral(Literal[int]):
    def __init__(self, value: SupportsInt) -> None:
        super().__init__(int(value))

    def __int__(self) -> int:
        return self.value


class FloatLiteral(Literal[float]):
    def __init__(self, value: SupportsFloat) -> None:
        super().__init__(float(value))

    def __float__(self) -> float:
        return self.value


class BoolLiteral(Literal[bool]):
    def __init__(self, value: bool) -> None:
        super().__init__(value)

    def __bool__(self) -> bool:
        return self.value


class NullLiteral(Literal[None]):
    def __init__(self) -> None:
        super().__init__(None)


true = BoolLiteral(True)
false = BoolLiteral(False)
null = NullLiteral()

zero_int = IntLiteral(0)
one_int = IntLiteral(1)

zero_float = FloatLiteral(0.0)
one_float = FloatLiteral(1.0)

table = Wildcard.symbol("table")
column = Wildcard.symbol("column")

int_ = Wildcard.symbol("int_", IntLiteral)
int1_ = Wildcard.symbol("int1_", IntLiteral)
int2_ = Wildcard.symbol("int2_", IntLiteral)

float_ = Wildcard.symbol("float_", FloatLiteral)
float1_ = Wildcard.symbol("float1_", FloatLiteral)
float2_ = Wildcard.symbol("float2_", FloatLiteral)

refs = Wildcard.dot("refs")
refs1 = Wildcard.dot("refs1")
refs2 = Wildcard.dot("refs2")

rel = Wildcard.dot("rel")
rel1 = Wildcard.dot("rel1")
rel2 = Wildcard.dot("rel2")

predicate = Wildcard.dot("predicate")
predicate1 = Wildcard.dot("predicate1")
predicate2 = Wildcard.dot("predicate2")

expr = Wildcard.dot("expr")
expr1 = Wildcard.dot("expr1")
expr2 = Wildcard.dot("expr2")

exprs00 = Wildcard.star("exprs00")
exprs0 = Wildcard.star("exprs0")
exprs1 = Wildcard.plus("exprs1")
exprs2 = Wildcard.plus("exprs2")


class Ref(Operation):
    name = "#"
    arity = Arity.polyadic
    infix = False

    def __str__(self) -> str:
        return ".".join(map(str, self.operands))


Read = Operation.new("R", Arity.unary, "Read")

# Rel, Refs
ProjectedRead = Operation.new("R#", Arity.binary, "ProjectedRead")

# Rel, predicate
SelectedRead = Operation.new("R>", Arity.binary, "SelectedRead")

# Rel, Refs, predicate
OptimizedRead = Operation.new("R+", Arity.ternary, "OptimizedRead")

# Left, right, predicate
Join = Operation.new("⋈", Arity.ternary, "Join")

# Rel, Exprs
Project = Operation.new("Π", Arity.binary, "Project")

# Rel, predicate
Select = Operation.new("σ", Arity.binary, "Select")

# At least one rel
Difference = Operation.new(
    "∖",
    Arity(min_count=1, fixed_size=False),
    "Difference",
    one_identity=True,
)

# At least one rel
Union = Operation.new(
    "⋃",
    Arity(min_count=1, fixed_size=False),
    "Union",
    commutative=True,
    associative=True,
    one_identity=True,
)

# At least one rel
Intersection = Operation.new(
    "⋂",
    Arity(min_count=1, fixed_size=False),
    "Intersection",
    commutative=True,
    associative=True,
    one_identity=True,
)

# Encoding identity elements without type information might be possible with a
# special Identity symbol, but for now we require at least one operand to
# arithmetic operations
Add = Operation.new(
    "+",
    Arity(min_count=1, fixed_size=False),
    "Add",
    associative=True,
    commutative=True,
    one_identity=True,
    infix=True,
)
Sub = Operation.new(
    "−",
    Arity(min_count=1, fixed_size=False),
    "Add",
    associative=True,
    commutative=True,
    one_identity=True,
    infix=True,
)
Mul = Operation.new(
    "×",
    Arity(min_count=1, fixed_size=False),
    "Mul",
    associative=True,
    commutative=True,
    one_identity=True,
    infix=True,
)
Div = Operation.new(
    "∕",
    Arity(min_count=1, fixed_size=False),
    "Div",
    one_identity=True,
    infix=True,
)
FloorDiv = Operation.new(
    "∕∕",
    Arity(min_count=1, fixed_size=False),
    "FloorDiv",
    one_identity=True,
    infix=True,
)

Refs = Operation.new("#*", Arity(min_count=1, fixed_size=False), "Refs")
Exprs = Operation.new("@", Arity(min_count=1, fixed_size=False), "Exprs")

Eq = Operation.new(
    "=",
    Arity.binary,
    "Eq",
    commutative=True,
    infix=True,
)
Ne = Operation.new(
    "≠",
    Arity.binary,
    "Ne",
    commutative=True,
    infix=True,
)
Lt = Operation.new(
    "<",
    Arity.binary,
    "Lt",
    infix=True,
)
Gt = Operation.new(
    ">",
    Arity.binary,
    "Gt",
    infix=True,
)
Le = Operation.new(
    "≤",
    Arity.binary,
    "Le",
    infix=True,
)
Ge = Operation.new(
    "≥",
    Arity.binary,
    "Ge",
    infix=True,
)

And = Operation.new(
    "⋀",
    Arity.variadic,
    "And",
    associative=True,
    commutative=True,
    one_identity=True,
)

Or = Operation.new(
    "⋁",
    Arity.variadic,
    "Or",
    associative=True,
    commutative=True,
    one_identity=True,
)

Not = Operation.new("¬", Arity.unary, "Not")


AND_RULES = (
    # Or identity
    ReplacementRule(Pattern(And()), lambda: true),
    ReplacementRule(Pattern(And(expr, expr)), lambda expr: expr),
    ReplacementRule(
        Pattern(And(exprs0, true, exprs1)),
        lambda exprs0, exprs1: And(*exprs0, *exprs1),
    ),
    ReplacementRule(
        Pattern(And(exprs1, true, exprs0)),
        lambda exprs1, exprs0: And(*exprs1, *exprs0),
    ),
    ReplacementRule(
        Pattern(And(exprs0, false, exprs1)),
        lambda **_: false,
    ),
    ReplacementRule(
        Pattern(And(exprs1, false, exprs0)),
        lambda **_: false,
    ),
    ReplacementRule(
        Pattern(And(exprs1, expr, exprs00, expr, exprs0)),
        lambda exprs1, expr, exprs00, exprs0: And(
            *exprs1,
            expr,
            *exprs00,
            *exprs0,
        ),
    ),
    ReplacementRule(
        Pattern(And(exprs0, expr, exprs00, expr, exprs1)),
        lambda exprs0, expr, exprs00, exprs1: And(
            *exprs0,
            expr,
            *exprs00,
            *exprs1,
        ),
    ),
)

OR_RULES = (
    # Or identity
    ReplacementRule(Pattern(Or()), lambda: false),
    ReplacementRule(Pattern(Or(expr, expr)), lambda expr: expr),
    ReplacementRule(
        Pattern(Or(exprs0, false, exprs1)),
        lambda exprs0, exprs1: Or(*exprs0, *exprs1),
    ),
    ReplacementRule(
        Pattern(Or(exprs1, false, exprs0)),
        lambda exprs1, exprs0: Or(*exprs1, *exprs0),
    ),
    ReplacementRule(
        Pattern(Or(exprs0, true, exprs1)),
        lambda **_: true,
    ),
    ReplacementRule(
        Pattern(Or(exprs1, true, exprs0)),
        lambda **_: true,
    ),
)

COMPARISON_RULES = (
    # a > b => b < a
    ReplacementRule(
        Pattern(Gt(expr1, expr2)), lambda expr1, expr2: Lt(expr2, expr1)
    ),
    # a >= b => b <= a
    ReplacementRule(
        Pattern(Ge(expr1, expr2)), lambda expr1, expr2: Le(expr2, expr1)
    ),
)

LOGICAL_RULES = (
    # not (a != b) => a == b
    ReplacementRule(
        Pattern(Not(Ne(expr1, expr2))),
        lambda expr1, expr2: Eq(expr1, expr2),
    ),
    # not (a == b) => a != b
    ReplacementRule(
        Pattern(Not(Eq(expr1, expr2))),
        lambda expr1, expr2: Ne(expr1, expr2),
    ),
    # not (not a) => a
    ReplacementRule(Pattern(Not(Not(expr))), lambda: expr),
    # not True => False
    ReplacementRule(Pattern(Not(true)), lambda: false),
    # not False => True
    ReplacementRule(Pattern(Not(false)), lambda: true),
    # not None => None
    ReplacementRule(Pattern(Not(null)), lambda: null),
)

MUL_RULES = (
    # int * 1
    ReplacementRule(Pattern(Mul(int_, one_int)), lambda int_: int_),
    ReplacementRule(Pattern(Mul(one_int, int_)), lambda int_: int_),
    # int * 0
    ReplacementRule(Pattern(Mul(int_, zero_int)), lambda **_: zero_int),
    ReplacementRule(Pattern(Mul(zero_int, int_)), lambda **_: zero_int),
    # float * 1
    ReplacementRule(Pattern(Mul(float_, one_float)), lambda float_: float_),
    ReplacementRule(Pattern(Mul(one_float, float_)), lambda float_: float_),
    # float * 0
    ReplacementRule(Pattern(Mul(float_, zero_float)), lambda **_: zero_float),
    ReplacementRule(Pattern(Mul(zero_float, float_)), lambda **_: zero_float),
    # int * 1.0
    ReplacementRule(
        Pattern(Mul(int_, one_float)),
        lambda int_: FloatLiteral(int_.value),
    ),
    ReplacementRule(
        Pattern(Mul(one_float, int_)),
        lambda int_: FloatLiteral(int_.value),
    ),
    # int * 0.0
    ReplacementRule(Pattern(Mul(int_, zero_float)), lambda **_: zero_float),
    ReplacementRule(Pattern(Mul(zero_float, int_)), lambda **_: zero_float),
    # int * float
    ReplacementRule(
        Pattern(Mul(int_, float_)),
        lambda int_, float_: FloatLiteral(int(int_) * float(float_)),
    ),
    ReplacementRule(
        Pattern(Mul(float_, int_)),
        lambda float_, int_: FloatLiteral(float(float_) * int(int_)),
    ),
)

ADD_RULES = (
    ReplacementRule(
        Pattern(Add(int1_, int2_)),
        lambda int1_, int2_: IntLiteral(int1_.value + int2_.value),
    ),
    ReplacementRule(
        Pattern(Add(float1_, float2_)),
        lambda float1_, float2_: FloatLiteral(float1_.value + float2_.value),
    ),
    ReplacementRule(
        Pattern(Add(float_, int_)),
        lambda float_, int_: FloatLiteral(float_.value + int_.value),
    ),
    ReplacementRule(
        Pattern(Add(int_, float_)),
        lambda int_, float_: FloatLiteral(int_.value + float_.value),
    ),
    ReplacementRule(
        Pattern(Add(exprs0, int1_, int2_, exprs1)),
        lambda exprs0, int1_, int2_, exprs1: Add(
            *exprs0,
            IntLiteral(int1_.value + int2_.value),
            *exprs1,
        ),
    ),
    ReplacementRule(
        Pattern(Add(exprs1, int1_, int2_, exprs0)),
        lambda exprs0, int1_, int2_, exprs1: Add(
            *exprs1,
            IntLiteral(int1_.value + int2_.value),
            *exprs0,
        ),
    ),
    ReplacementRule(
        Pattern(Add(exprs0, float1_, float2_, exprs1)),
        lambda exprs0, float1_, float2_, exprs1: Add(
            *exprs0,
            FloatLiteral(float1_.value + float2_.value),
            *exprs1,
        ),
    ),
    ReplacementRule(
        Pattern(Add(exprs1, float1_, float2_, exprs0)),
        lambda exprs0, float1_, float2_, exprs1: Add(
            *exprs1,
            FloatLiteral(float1_.value + float2_.value),
            *exprs0,
        ),
    ),
    ReplacementRule(
        Pattern(Add(exprs0, int_, float_, exprs1)),
        lambda exprs0, int_, float_, exprs1: Add(
            *exprs0,
            FloatLiteral(int_.value + float_.value),
            *exprs1,
        ),
    ),
    ReplacementRule(
        Pattern(Add(exprs1, float_, int_, exprs0)),
        lambda exprs0, float_, int_, exprs1: Add(
            *exprs1,
            FloatLiteral(float_.value + int_.value),
            *exprs0,
        ),
    ),
)


def can_replace_join(predicate, rel1, rel2):
    rel1_predicates, rel2_predicates, _ = partition_predicate(
        predicate=predicate,
        rel1=rel1,
        rel2=rel2,
    )
    return rel1_predicates or rel2_predicates


def select_join_replacement(predicate, rel1, rel2):
    rel1_predicates, rel2_predicates, remaining = itertools.starmap(
        And,
        partition_predicate(
            predicate=predicate,
            rel1=rel1,
            rel2=rel2,
        ),
    )
    return Join(
        Select(rel1, rel1_predicates),
        Select(rel2, rel2_predicates),
        remaining,
    )


def refs_are_subset(rel, exprs2, exprs1):
    return frozenset(exprs2).issubset(exprs1)


RELATION_RULES = (
    # a filter of `true` can be eliminated
    ReplacementRule(Pattern(Select(rel, true)), lambda rel: rel),
    # diff(select(t, p), select(s, p)) => select(diff(t, s), p)
    ReplacementRule(
        Pattern(Difference(Select(rel1, predicate), Select(rel2, predicate))),
        lambda rel1, rel2, predicate: Select(
            Difference(rel1, rel2),
            predicate,
        ),
    ),
    # diff(select(rel1), rel2) => select(diff(t, s), p)
    ReplacementRule(
        Pattern(Difference(Select(rel1, predicate), rel2)),
        lambda rel1, predicate, rel2: Select(
            Difference(rel1, rel2),
            predicate,
        ),
    ),
    # union(select(t, p), select(s, p)) => select(union(t, s), p)
    ReplacementRule(
        Pattern(Union(Select(rel1, predicate), Select(rel2, predicate))),
        lambda predicate, rel1, rel2: Select(
            Union(rel1, rel2),
            predicate,
        ),
    ),
    # intersection(select(t, p), select(s, p)) => select(intersection(t, s), p)
    ReplacementRule(
        Pattern(
            Intersection(
                Select(rel1, predicate),
                Select(rel2, predicate),
            )
        ),
        lambda predicate, rel1, rel2: Select(
            Intersection(rel1, rel2),
            predicate,
        ),
    ),
    # intersect(select(rel1), rel2) => select(intersect(t, s), p)
    ReplacementRule(
        Pattern(Intersection(Select(predicate, rel1), rel2)),
        lambda predicate, rel1, rel2: Select(
            Intersection(rel1, rel2),
            predicate,
        ),
    ),
    ReplacementRule(
        Pattern(Intersection(rel1, Select(rel2, predicate))),
        lambda rel1, predicate, rel2: Select(
            Intersection(rel1, rel2),
            predicate,
        ),
    ),
    # avoid unions by turning them into a filter with ORs where possible
    ReplacementRule(
        Pattern(Union(Select(rel, predicate1), Select(rel, predicate2))),
        lambda predicate1, rel1, predicate2, rel2: Select(
            rel, Or(predicate1, predicate2)
        ),
    ),
    # compose filters
    ReplacementRule(
        Pattern(Select(Select(rel, predicate1), predicate2)),
        lambda rel, predicate1, predicate2: Select(
            rel, And(predicate1, predicate2)
        ),
    ),
    # project before filter if the filter predicate refers to a subset of the
    # projection columns
    ReplacementRule(
        Pattern(
            Project(Select(rel, predicate), Exprs(exprs1)),
            CustomConstraint(
                lambda exprs1, predicate, rel: predicate.is_subset_of(exprs1)
            ),
        ),
        lambda exprs1, predicate, rel: Select(
            Project(rel, Exprs(*exprs1)),
            predicate,
        ),
    ),
    # collapse repeated projections
    ReplacementRule(
        Pattern(Project(Project(rel, expr), expr)),
        lambda rel, expr: Project(rel, expr),
    ),
    # remove the child projection if the parent columns are a subset of the
    # child columns
    ReplacementRule(
        Pattern(
            Project(Project(rel, Refs(exprs1)), Refs(exprs2)),
            CustomConstraint(
                lambda rel, exprs1, exprs2: frozenset(exprs2).issubset(exprs1)
            ),
        ),
        lambda rel, exprs1, exprs2: Project(rel, Refs(*exprs2)),
    ),
    # remove the child projection if the parent expressions are a subset of the
    # child expressions
    ReplacementRule(
        Pattern(
            Project(Project(rel, Exprs(exprs1)), Exprs(exprs2)),
            CustomConstraint(
                lambda rel, exprs1, exprs2: frozenset(exprs2).issubset(exprs1)
            ),
        ),
        lambda rel, exprs1, exprs2: Project(rel, Exprs(*exprs2)),
    ),
    ReplacementRule(
        Pattern(
            Project(Project(rel, Refs(exprs1)), Exprs(exprs2)),
            CustomConstraint(
                lambda rel, exprs1, exprs2: frozenset(exprs2).issubset(exprs1)
            ),
        ),
        lambda rel, exprs1, exprs2: Project(rel, Refs(*exprs2)),
    ),
    ReplacementRule(
        Pattern(
            Project(Project(rel, Exprs(exprs1)), Refs(exprs2)),
            CustomConstraint(
                lambda rel, exprs1, exprs2: frozenset(exprs2).issubset(exprs1)
            ),
        ),
        lambda rel, exprs1, exprs2: Project(rel, Refs(*exprs2)),
    ),
    # turn a read -> filter -> project into a project of an optimized read
    # the projection is necessary because we're using expressions
    ReplacementRule(
        Pattern(Project(Select(Read(table), predicate), Exprs(exprs1))),
        lambda table, exprs1, predicate: Project(
            OptimizedRead(table, Refs(*find_refs(exprs1)), predicate),
            Exprs(*exprs1),
        ),
    ),
    # turn a read -> project -> filter into a project of an optimized read
    # the projection is necessary because we're using expressions
    ReplacementRule(
        Pattern(Select(Project(Read(table), Exprs(exprs1)), predicate)),
        lambda table, exprs1, predicate: Project(
            OptimizedRead(table, Refs(*find_refs(exprs1)), predicate),
            Exprs(*exprs1),
        ),
    ),
    # a project of an optimized read of the same columns is redundant
    ReplacementRule(
        Pattern(
            Project(OptimizedRead(rel, Refs(exprs1), predicate), Exprs(exprs1))
        ),
        lambda rel, exprs1, predicate: OptimizedRead(
            rel,
            Refs(*exprs1),
            predicate,
        ),
    ),
    # fuse read -> project into a projected read
    ReplacementRule(
        Pattern(Project(Read(table), Exprs(exprs1))),
        lambda table, exprs1: Project(
            ProjectedRead(table, Refs(*find_refs(exprs1))),
            Exprs(*exprs1),
        ),
    ),
    # fuse read -> project into a projected read
    ReplacementRule(
        Pattern(Project(Read(table), Refs(exprs1))),
        lambda table, exprs1: ProjectedRead(table, Refs(*exprs1)),
    ),
    # remove redundant projections from a projected read
    ReplacementRule(
        Pattern(Project(ProjectedRead(table, Refs(exprs1)), Exprs(exprs1))),
        lambda table, exprs1: ProjectedRead(table, Refs(*exprs1)),
    ),
    ReplacementRule(
        Pattern(Project(ProjectedRead(table, Exprs(exprs1)), Refs(exprs1))),
        lambda table, exprs1: ProjectedRead(table, Refs(*exprs1)),
    ),
    ReplacementRule(
        Pattern(Select(Read(table), predicate)),
        lambda table, predicate: SelectedRead(
            table,
            predicate,
        ),
    ),
    ReplacementRule(
        Pattern(
            Join(rel1, rel2, predicate),
            CustomConstraint(can_replace_join),
        ),
        select_join_replacement,
    ),
    ReplacementRule(
        Pattern(OptimizedRead(rel, refs, true)),
        lambda rel, refs: ProjectedRead(rel, refs),
    ),
)


def partition_predicate(*, predicate, rel1, rel2):
    if not isinstance(predicate, And):
        return And(), And(), predicate

    assert isinstance(predicate, And), f"{type(predicate).__name__}"
    rel1_refs = frozenset(find_refs(rel1))
    rel2_refs = frozenset(find_refs(rel2))
    rel1_operands = []
    rel2_operands = []
    remaining = []
    for operand in predicate:
        operand_refs = frozenset(find_refs(operand))
        # if operand contains only references to rel
        if operand_refs <= rel1_refs and not operand_refs <= rel2_refs:
            rel1_operands.append(operand)
        elif operand_refs <= rel2_refs:
            rel2_operands.append(operand)
        else:
            remaining.append(operand)
    return And(*rel1_operands), And(*rel2_operands), And(*remaining)


RULES = (
    AND_RULES
    + OR_RULES
    + COMPARISON_RULES
    + LOGICAL_RULES
    + ADD_RULES
    + MUL_RULES
    + RELATION_RULES
)


def find_refs(
    refs,
    pattern: Pattern = Pattern(Ref(table, column)),
) -> frozenset[Ref]:
    """Find all unique subexpressions that are field references."""
    pattern = Pattern(Ref(table, column))
    matches = matchpy.match_anywhere(refs, pattern)
    return toolz.unique(
        Ref(match["table"], match["column"]) for match, _ in matches
    )


def optimize(
    expr,
    rules: Iterable[ReplacementRule] = RULES,
) -> matchpy.Expression:
    """Optimize an expression."""
    return matchpy.replace_all(expr, rules)
