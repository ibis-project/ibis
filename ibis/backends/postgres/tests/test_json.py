""" Tests for json data types"""
import json

import pytest
from pytest import param

import ibis


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
def test_json(data, alltypes):
    json_value = json.dumps(data)
    lit = ibis.literal(json_value, type='json').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == json_value


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
def test_jsonb(data, alltypes):
    jsonb_value = json.dumps(data).encode('utf8')
    lit = ibis.literal(jsonb_value, type='jsonb').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == jsonb_value
