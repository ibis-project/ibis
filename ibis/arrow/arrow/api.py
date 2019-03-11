"""
    API for in memory DataFrames
"""

from arrow.client import ArrowClient


def connect(dictionary):
    """
    :param dictionary: dictionary containing the different tables
    :return: Client
    """
    return ArrowClient(dictionary)
