import json
import os
import pathlib
import tempfile

import pytest
from plumbum import local

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.backends.pyarrow.compiler import PyArrowCompiler


class FlatbufferHelpers:
    def __init__(self, arrow_path):
        self.arrow_path = pathlib.Path(arrow_path)
        self.plan_fbs = (
            self.arrow_path / "experimental" / "computeir" / "Plan.fbs"
        )

    def generate(self, dest):
        """
        Generate python code from flatbuffers definitions.
        """
        dest = pathlib.Path(dest).resolve()
        cmd = local.cmd.flatc[
            '--gen-all', '--python', str(self.plan_fbs), '-o'
        ]
        return cmd(str(dest))

    def to_json(
        self, data, root_type="org.apache.arrow.computeir.flatbuf.Plan"
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            cmd = local.cmd.flatc[
                '-o',
                str(tmpdir),
                '--raw-binary',
                '--json',
                '--strict-json',
                '--defaults-json',
                str(self.plan_fbs),
                '--root-type',
                root_type,
                '--',
            ]

            # write serialized data to file
            src = tmpdir / "serialized"
            src.write_bytes(data)
            # deserialize to json
            cmd(str(src))
            # read back the json file
            dest = src.with_suffix('.json')
            with dest.open() as fp:
                return json.load(fp)

    def save_fixture(self, json_data, name, sql=None):
        fixtures_dir = pathlib.Path(__file__).parent / 'fixtures'

        pretty_json = json.dumps(json_data, indent=4, sort_keys=True)
        (fixtures_dir / f"{name}.json").write_text(pretty_json)

        if sql:
            (fixtures_dir / f"{name}.sql").write_text(sql)


function_names = {
    ops.Add: "add",
    ops.Subtract: "sub",
    ops.Multiply: "mul",
    ops.Less: "less",
    ops.Greater: "greater",
    ops.And: "and",
    ops.Sum: "sum",
    ops.Mean: "mean",
}


@pytest.fixture
def helpers():
    arrow_path = os.environ.get("ARROW_PATH", "/Users/kszucs/Workspace/arrow")
    return FlatbufferHelpers(arrow_path)


@pytest.fixture
def compiler():
    return PyArrowCompiler(function_names)


@pytest.fixture
def table():
    table = ibis.table(
        name='tbl',
        schema=[('foo', 'int32'), ('bar', 'int64'), ('baz', 'double')],
    )
    return table


@pytest.mark.parametrize(
    'expr',
    [
        ibis.literal(1),
        ibis.literal(1) + ibis.literal(2),
        ibis.literal(1) - ibis.literal(2),
        ibis.literal(1) * ibis.literal(2),
    ],
)
def test_produce_arrow_ir(compiler, expr):
    compiler.serialize(expr)


@pytest.mark.parametrize(
    ('dtype', 'typtyp', 'typ'),
    [
        (dt.int8, 'Int', dict(bitWidth=8, is_signed=True)),
        (dt.int16, 'Int', dict(bitWidth=16, is_signed=True)),
        (dt.int32, 'Int', dict(bitWidth=32, is_signed=True)),
        (dt.int64, 'Int', dict(bitWidth=64, is_signed=True)),
        (dt.uint8, 'Int', dict(bitWidth=8, is_signed=False)),
        (dt.uint16, 'Int', dict(bitWidth=16, is_signed=False)),
        (dt.uint32, 'Int', dict(bitWidth=32, is_signed=False)),
        (dt.uint64, 'Int', dict(bitWidth=64, is_signed=False)),
        (dt.float16, 'FloatingPoint', dict(precision="HALF")),
        (dt.float32, 'FloatingPoint', dict(precision="SINGLE")),
        (dt.float64, 'FloatingPoint', dict(precision="DOUBLE")),
        (dt.string, 'Utf8', dict()),
        (dt.date, 'Date', dict(unit="MILLISECOND")),
        (dt.time, 'Time', dict(bitWidth=64, unit="NANOSECOND")),
        (dt.timestamp, 'Timestamp', dict(unit="NANOSECOND")),
        (
            dt.Timestamp(timezone="Europe/Budapest"),
            'Timestamp',
            dict(unit="NANOSECOND", timezone="Europe/Budapest"),
        ),
        (dt.interval, 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('Y'), 'Interval', dict(unit="YEAR_MONTH")),
        (dt.Interval('Q'), 'Interval', dict(unit="YEAR_MONTH")),
        (dt.Interval('M'), 'Interval', dict(unit="YEAR_MONTH")),
        (dt.Interval('W'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('D'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('h'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('m'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('s'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('ms'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('us'), 'Interval', dict(unit="MONTH_DAY_NANO")),
        (dt.Interval('ns'), 'Interval', dict(unit="MONTH_DAY_NANO")),
    ],
    ids=str,
)
@pytest.mark.parametrize(
    'nullable', [True, False], ids=['nullable', 'non-nullable']
)
def test_compile_primitive_types(
    compiler, helpers, dtype, typtyp, typ, nullable
):
    dtype = dtype(nullable=nullable)
    schema = ibis.schema([('test', dtype)])

    result = compiler.serialize(schema)
    json = helpers.to_json(result, root_type="org.apache.arrow.flatbuf.Schema")

    expected = {
        "endianness": "Little",
        "fields": [
            {
                "name": "test",
                "nullable": nullable,
                "type_type": typtyp,
                "type": typ,
            }
        ],
    }
    assert json == expected


def test_compile_array_type(compiler, helpers):
    dtype = dt.Array(dt.int64)
    schema = ibis.schema([('test', dtype)])
    result = compiler.serialize(schema)
    json = helpers.to_json(result, root_type="org.apache.arrow.flatbuf.Schema")
    expected = {
        "endianness": "Little",
        "fields": [
            {
                "name": "test",
                "nullable": True,
                "type_type": "List",
                "type": {},
                "children": [
                    {
                        "nullable": True,
                        "type_type": "Int",
                        "type": {"bitWidth": 64, "is_signed": True},
                    }
                ],
            }
        ],
    }
    assert json == expected


def test_projection_with_filter(compiler, helpers, table):
    expr = table[table.foo < 3][table.bar, table.baz]
    flat = compiler.serialize(expr)

    helpers.save_fixture(
        helpers.to_json(flat),
        name="projection_with_filter",
        sql="SELECT bar, baz FROM tbl WHERE foo < 3;",
    )


def test_projection_with_filter_and_sort(compiler, helpers, table):
    expr = table[table].sort_by([table.foo, ibis.desc(table.bar)])
    flat = compiler.serialize(expr)

    helpers.save_fixture(
        helpers.to_json(flat),
        name="projection_with_filter_and_sort",
        sql="SELECT * FROM tbl ORDER BY foo ASC, bar DESC;",
    )


def test_aggregation_simple(compiler, helpers, table):
    t = table
    stats = [t.bar.sum().name('total_bar'), t.baz.mean().name('avg_baz')]
    agged = t.groupby(t.foo).aggregate(stats)
    flat = compiler.serialize(agged)

    helpers.save_fixture(
        helpers.to_json(flat),
        name="aggregation_simple",
        sql=(
            "SELECT SUM(bar) AS total_bar, MEAN(baz) AS avg_baz "
            "FROM tbl GROUP BY foo;"
        ),
    )


def test_aggregation_with_having(compiler, helpers, table):
    t = table
    total_bar = t.bar.sum().name('total_bar')
    avg_baz = t.baz.mean().name('avg_baz')
    agged = t[t.foo < 3].aggregate(
        [total_bar, avg_baz], by=[t.foo], having=[total_bar > 10]
    )
    flat = compiler.serialize(agged)

    helpers.save_fixture(
        helpers.to_json(flat),
        name="aggregation_with_having",
        sql=(
            "SELECT SUM(bar) AS total_bar, MEAN(baz) AS avg_baz FROM tbl "
            "WHERE foo < 3 GROUP BY foo HAVING total_bar > 10;"
        ),
    )
