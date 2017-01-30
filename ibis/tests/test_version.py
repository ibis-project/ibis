import ibis
from pkg_resources import parse_version, SetuptoolsLegacyVersion


def test_version():
    assert not isinstance(
        parse_version(ibis.__version__),
        SetuptoolsLegacyVersion
    )
