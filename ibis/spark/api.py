from ibis.spark.client import SparkClient


def connect(session):
    return SparkClient(session)
