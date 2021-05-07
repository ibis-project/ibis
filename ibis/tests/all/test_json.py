""" Tests for json data types"""
import json

import pytest
from pytest import param

import ibis
from ibis.tests.backends import Postgres

# add here backends that support json types
all_db_geo_supported = [Postgres]


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_json(backend, con, data, alltypes):
    json_value = json.dumps(data)
    lit = ibis.literal(json_value, type='json').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == json_value


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_jsonb(backend, con, data, alltypes):
    jsonb_value = json.dumps(data).encode('utf8')
    lit = ibis.literal(jsonb_value, type='jsonb').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == jsonb_value
