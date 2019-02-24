import pandas as pd
import pyspark
import pytest

from ibis.spark.api import connect


@pytest.fixture(scope='session')
def session():
    s = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()
    yield s
    s.stop()


@pytest.fixture(scope='session')
def df1():
    return pd.DataFrame({'key': [1, 2, 3], 'val1': [2.7, 3.14, 4.2]})


@pytest.fixture(scope='session')
def df2():
    return pd.DataFrame({'key': [1, 2, 3], 'val2': [100, 101, 102]})


@pytest.fixture(scope='session')
def client(session, df1, df2):
    df = session.createDataFrame(df1)
    df.createOrReplaceTempView('table1')

    df = session.createDataFrame(df2)
    df.createOrReplaceTempView('table2')

    return connect(session)
