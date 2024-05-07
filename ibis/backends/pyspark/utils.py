from __future__ import annotations


def format_interval_as_string(interval):
    return f"{interval.op().value} {interval.op().dtype.unit.name.lower()}"
