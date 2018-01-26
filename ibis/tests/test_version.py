from pkg_resources import parse_version, SetuptoolsLegacyVersion

import sh

import ibis


def test_import_time():
    lines = [
        'from timeit import timeit',
        "print(timeit('import ibis'))",
    ]

    delta = float(str(sh.python(c='; '.join(lines))))
    assert delta < 2.0


def test_version():
    assert not isinstance(
        parse_version(ibis.__version__),
        SetuptoolsLegacyVersion
    )
