"""
Shared functions for the SQL-based backends.

Eventually this should be converted to a base class inherited
from the SQL-based backends.
"""
import ibis
import ibis.config

with ibis.config.config_prefix('sql'):
    ibis.config.register_option(
        'default_limit',
        10_000,
        'Number of rows to be retrieved for an unlimited table expression',
    )
