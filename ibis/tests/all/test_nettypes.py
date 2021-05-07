""" Tests for macaddr and inet data types"""

import pytest
from pytest import param

import ibis
from ibis.tests.backends import Postgres

# add here backends that support json types
net_types_supported = [Postgres]


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
@pytest.mark.only_on_backends(net_types_supported)
def test_macaddr(backend, con, data, alltypes):
    macaddr_value = '00:0a:95:9d:68:16'
    lit = ibis.literal(macaddr_value, type='macaddr').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == macaddr_value


@pytest.mark.parametrize('data', [param({'status': True}, id='status')])
@pytest.mark.only_on_backends(net_types_supported)
def test_inet(backend, con, data, alltypes):
    inet_value = '00:0a:95:9d:68:16'
    lit = ibis.literal(inet_value, type='inet').name('tmp')
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df['tmp'].iloc[0] == inet_value
