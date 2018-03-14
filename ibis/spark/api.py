from __future__ import absolute_import

from ibis.spark.client import SparkClient


__all__ = ['connect']


def connect(dictionary):
    return SparkClient(dictionary)
