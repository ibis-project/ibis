"""Isolated namespace for chDB ``Python(<name>)`` memtable resolution.

chDB resolves ``Python(<name>)`` by scanning the globals of caller stack
frames, innermost first. Queries are issued from this module so that
registered tables are visible to the engine without user-chosen names ever
shadowing the backend module's imports.
"""

from __future__ import annotations


def _register(name, table):
    globals()[name] = table


def _unregister(name):
    globals().pop(name, None)


def _query(session, sql, fmt="CSV"):
    return session.query(sql, fmt)


def _send_query(session, sql, fmt="Arrow"):
    return session.send_query(sql, fmt)
