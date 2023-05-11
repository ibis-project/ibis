from __future__ import annotations

from typing import Any

from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

import ibis
from ibis.backends.base import BaseBackend

__all__ = ["IbisConnection"]


class IbisConnection(ExperimentalBaseConnection[BaseBackend]):
    def _connect(self, **kwargs: Any) -> BaseBackend:
        """Connect to the backend and return a client object.

        This method is invoked when `st.experimental_connection` is called and
        pulls information from streamlit secrets. `_connect` is part of the
        streamlit connection API to be implemented by developers of specific
        connection types.

        Here's an example not-so-secret configuration:

        ```toml
        [connections.ch]
        url = "clickhouse://play:clickhouse@play.clickhouse.com:9440/?secure=1"
        ```

        Alternatively, you can specify individual arguments under a connection
        whose name matches the backend type. For example:

        ```toml
        [connections.clickhouse]
        user = "play"
        password = "clickhouse"
        host = "play.clickhouse.com"
        port = 9440
        secure = 1
        ```

        This file can be placed at `~/.streamlit/secrets.toml`.

        You can then connect to the backend using:

        ```python
        import streamlit as st

        from ibis.streamlit import IbisConnection

        con = st.experimental_connection("ch", type=IbisConnection)

        # Now you can use `con` as if it were an ibis backend
        con.list_tables()
        ```
        """
        # add secrets config to kwargs
        kwargs.update(self._secrets.to_dict())

        # pop out the connection string
        url = kwargs.pop("url", None)

        # if there's no url parameter, use the connection name to get connection arguments
        if url is None:
            return getattr(ibis, self._connection_name).connect(**kwargs)

        return ibis.connect(url, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._instance, name)

    @cache_data
    def list_tables(_self, *args, **kwargs) -> list[str]:
        return _self._instance.list_tables(*args, **kwargs)
