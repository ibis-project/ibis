"""Converter for Materialize backend.

Materialize uses PostgreSQL wire protocol, so we can reuse PostgreSQL converters.
"""

from __future__ import annotations

# Re-export PostgreSQL converters for Materialize compatibility
from ibis.backends.postgres.converter import *  # noqa: F403
