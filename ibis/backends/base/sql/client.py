import abc

from ibis.backends.base import Client


class SQLClient(Client, metaclass=abc.ABCMeta):
    """Generic SQL client."""
