from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as exc
import ibis.selectors as s


def test_individual_columns():
    t = ibis.table(dict(x="int", y="int"))
    assert t.relocate("x", after="y").columns == tuple("yx")
    assert t.relocate("y", before="x").columns == tuple("yx")


def test_move_blocks():
    t = ibis.table(dict(x="int", a="string", y="int", b="string"))
    assert t.relocate(s.of_type("string")).columns == tuple("abxy")
    assert t.relocate(s.of_type("string"), after=s.numeric()).columns == tuple("xyab")


def test_duplicates_not_renamed():
    t = ibis.table(dict(x="int", y="int"))
    assert t.relocate("y", s.numeric()).columns == tuple("yx")
    assert t.relocate("y", s.numeric(), "y").columns == tuple("yx")


def test_keep_non_contiguous_variables():
    t = ibis.table(dict.fromkeys("abcde", "int"))
    assert t.relocate("b", after=s.cols("a", "c", "e")).columns == tuple("acdeb")
    assert t.relocate("e", before=s.cols("b", "d")).columns == tuple("aebcd")


def test_before_after_does_not_move_to_front():
    t = ibis.table(dict(x="int", y="int"))
    assert t.relocate("y").columns == tuple("yx")


def test_only_one_of_before_and_after():
    t = ibis.table(dict(x="int", y="int", z="int"))

    with pytest.raises(exc.IbisInputError, match="Cannot specify both"):
        t.relocate("z", before="x", after="y")


def test_respects_order():
    t = ibis.table(dict.fromkeys("axbzy", "int"))
    assert t.relocate("x", "y", "z", before="x").columns == tuple("axyzb")
    assert t.relocate("x", "y", "z", before=s.last()).columns == tuple("abxyz")
    assert t.relocate("x", "a", "z").columns == tuple("xazby")


def test_relocate_can_rename():
    t = ibis.table(dict(a="int", b="int", c="int", d="string", e="string", f=r"string"))
    assert t.relocate(ffff="f").columns == ("ffff", *"abcde")
    assert t.relocate(ffff="f", before="c").columns == (*"ab", "ffff", *"cde")
    assert t.relocate(ffff="f", after="c").columns == (*"abc", "ffff", *"de")


def test_retains_last_duplicate_when_renaming_and_moving():
    t = ibis.table(dict(x="int"))
    assert t.relocate(a="x", b="x").columns == ("b",)

    # TODO: test against .rename once that's implemented

    t = ibis.table(dict(x="int", y="int"))
    assert t.relocate(a="x", b="y", c="x").columns == tuple("bc")


def test_everything():
    t = ibis.table(dict(w="int", x="int", y="int", z="int"))
    assert t.relocate("y", "z", before=s.all()).columns == tuple("yzwx")
    assert t.relocate("y", "z", after=s.all()).columns == tuple("wxyz")


def test_moves_to_front_with_no_before_and_no_after():
    t = ibis.table(dict(x="int", y="int", z="int"))
    assert t.relocate("z", "y").columns == tuple("zyx")


def test_empty_before_moves_to_front():
    t = ibis.table(dict(x="int", y="int", z="int"))
    assert t.relocate("y", before=s.of_type("string")).columns == tuple("yxz")


def test_empty_after_moves_to_end():
    t = ibis.table(dict(x="int", y="int", z="int"))
    assert t.relocate("y", after=s.of_type("string")).columns == tuple("xzy")


def test_no_arguments():
    t = ibis.table(dict(x="int", y="int", z="int"))
    with pytest.raises(exc.IbisInputError, match="At least one selector"):
        t.relocate()


def test_tuple_input():
    t = ibis.table(dict(x="int", y="int", z="int"))
    assert t.relocate(("y", "z")).columns == tuple("yzx")

    # not allowed, because this would be technically inconsistent with `select`
    # though, the tuple is unambiguous here and could never be interpreted as a
    # scalar array
    with pytest.raises(KeyError):
        t.relocate(("y", "z"), "x")
