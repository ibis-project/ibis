from ibis.backends.base.sql.alchemy.registry import sqlalchemy_operation_registry

operation_registry = sqlalchemy_operation_registry.copy()

# TODO: trino doesn't support `& |` for bitwise ops, it wants `bitwise_and` and `bitwise_or``
