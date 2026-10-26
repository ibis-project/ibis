"""Helpers shared by backends that connect through ADBC drivers."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import adbc_driver_manager

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def detect_missing_adbc_driver(driver: str, /) -> Iterator[None]:
    """Explain how to install an ADBC driver that the driver manager couldn't load."""
    try:
        yield
    except adbc_driver_manager.ProgrammingError as e:
        # The driver manager reports a driver it can't find as NOT_FOUND with a
        # "[Driver Manager]" prefix; a missing table or database is also
        # NOT_FOUND, but never has that prefix.
        if (
            e.status_code != adbc_driver_manager.AdbcStatusCode.NOT_FOUND
            or "[Driver Manager] Could not load" not in str(e)
        ):
            raise
        raise ImportError(
            f"The ADBC {driver!r} driver is not installed. Ibis needs it in "
            "addition to the Python package. Install it with:\n\n"
            f"    uvx dbc install {driver}\n\n"
            "or, inside an activated virtual environment, `pip install dbc` "
            f"and then `dbc install {driver}`. For other options see "
            f"https://docs.adbc-drivers.org/drivers/{driver}/\n\n"
            "If the driver is installed in a non-standard location, set the "
            "ADBC_DRIVER_PATH environment variable to that directory."
        ) from e
