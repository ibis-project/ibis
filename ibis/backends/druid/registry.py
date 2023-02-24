from __future__ import annotations

from ibis.backends.base.sql.alchemy import (
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
)

operation_registry = sqlalchemy_operation_registry.copy()

operation_registry.update(sqlalchemy_window_functions_registry)
