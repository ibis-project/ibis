from __future__ import annotations

import sqlalchemy as sa

import ibis.expr.operations as ops

# used for literal translate
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
)


class _String:
    @staticmethod
    def find(t, op):
        args = [t.translate(op.substr), t.translate(op.arg)]
        if (start := op.start) is not None:
            args.append(t.translate(start) + 1)
        return sa.func.locate(*args) - 1

    @staticmethod
    def translate(t, op):
        func = fixed_arity(sa.func.translate, 3)
        return func(t, op)


class _Registry:
    _unsupported = {ops.StringJoin}

    _supported = {
        ops.Translate: _String.translate,
        ops.StringFind: _String.find,
    }

    @classmethod
    def create(cls):
        registry = sqlalchemy_operation_registry.copy()
        registry = {k: v for k, v in registry.items() if k not in cls._unsupported}
        registry.update(cls._supported)
        return registry


def create():
    """Create an operation registry for an Exasol backend."""
    return _Registry.create()
