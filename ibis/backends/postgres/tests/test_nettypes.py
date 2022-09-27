"""Tests for macaddr and inet data types."""

import pytest
from pytest import param

import ibis


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
def test_macaddr(con, data, alltypes):
    macaddr_value = '00:0a:95:9d:68:16'
    lit = ibis.literal(macaddr_value, type='macaddr').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == macaddr_value


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
def test_inet(con, data, alltypes):
    inet_value = '00:0a:95:9d:68:16'
    lit = ibis.literal(inet_value, type='inet').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == inet_value
