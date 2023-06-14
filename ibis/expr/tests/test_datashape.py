from __future__ import annotations

from ibis.expr.datashape import (
    Any,
    Columnar,
    DataShape,
    Scalar,
    Tabular,
    columnar,
    scalar,
    tabular,
)


def test_scalar_shape():
    s = Scalar()
    assert s.ndim == 0
    assert s.is_scalar()
    assert not s.is_columnar()
    assert not s.is_tabular()


def test_columnar_shape():
    c = Columnar()
    assert c.ndim == 1
    assert not c.is_scalar()
    assert c.is_columnar()
    assert not c.is_tabular()


def test_tabular_shape():
    t = Tabular()
    assert t.ndim == 2
    assert not t.is_scalar()
    assert not t.is_columnar()
    assert t.is_tabular()


def test_shapes_are_singletons():
    assert Scalar() is scalar
    assert Scalar() is Scalar()
    assert Columnar() is columnar
    assert Columnar() is Columnar()
    assert Tabular() is tabular
    assert Tabular() is Tabular()


def test_shape_comparison():
    assert Scalar() < Columnar()
    assert Scalar() <= Columnar()
    assert Columnar() > Scalar()
    assert Columnar() >= Scalar()
    assert Scalar() != Columnar()
    assert Scalar() == Scalar()
    assert Columnar() == Columnar()
    assert Tabular() == Tabular()
    assert Tabular() != Columnar()
    assert Tabular() != Scalar()
    assert Tabular() > Columnar()
    assert Tabular() > Scalar()
    assert Tabular() >= Columnar()
    assert Tabular() >= Scalar()


def test_shapes_are_hashable():
    assert hash(Scalar()) == hash(Scalar())
    assert hash(Columnar()) == hash(Columnar())
    assert hash(Tabular()) == hash(Tabular())
    assert hash(Scalar()) != hash(Columnar())
    assert hash(Scalar()) != hash(Tabular())
    assert hash(Columnar()) != hash(Tabular())
    assert len({Scalar(), Columnar(), Tabular()}) == 3


def test_backward_compat_aliases():
    assert DataShape.SCALAR is scalar
    assert DataShape.COLUMNAR is columnar
    assert DataShape.TABULAR is tabular


def test_any_alias_for_datashape():
    # useful for typehints like `ds.Any`
    assert DataShape is Any
