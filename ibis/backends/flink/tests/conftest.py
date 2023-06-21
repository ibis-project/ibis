import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES


@pytest.fixture
def batting() -> ir.Table:
    return ibis.table(schema=TEST_TABLES["batting"], name="batting")


@pytest.fixture
def awards_players() -> ir.Table:
    return ibis.table(schema=TEST_TABLES["awards_players"], name="awards_players")
