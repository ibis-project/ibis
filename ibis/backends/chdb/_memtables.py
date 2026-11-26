"""Isolated namespace for chDB ``Python(<name>)`` memtable resolution.

chDB resolves ``Python(<name>)`` by scanning the locals and then the globals of
caller stack frames, innermost first. Queries are issued from this module so
that registered tables are visible to the engine without user-chosen names ever
shadowing the backend module's imports.
"""

from __future__ import annotations

import functools


def _register(name, table):
    globals()[name] = table


def _unregister(name):
    globals().pop(name, None)


def _query(session, sql, fmt="CSV"):
    return session.query(sql, fmt)


def _send_query(session, sql, fmt="Arrow"):
    return session.send_query(sql, fmt)


@functools.cache
def _reserved():
    """Names a registered table can't use.

    chDB checks these before this module's globals: the locals of its own
    ``Connection`` method and of the function here that issued the query, and
    that method's module globals, where any subscriptable or Arrow-stream value
    ends the search. Registering over one of this module's globals would
    clobber it.
    """
    from chdb.state import sqlitelike

    frames = (
        sqlitelike.Connection.query,
        sqlitelike.Connection.send_query,
        _query,
        _send_query,
    )
    return _OWN.union(
        *(frame.__code__.co_varnames for frame in frames),
        (
            name
            for name, value in vars(sqlitelike).items()
            if hasattr(value, "__getitem__") or hasattr(value, "__arrow_c_stream__")
        ),
    )


_OWN = frozenset([*globals(), "_OWN"])
