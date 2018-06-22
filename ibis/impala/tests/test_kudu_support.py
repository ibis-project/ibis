import unittest
import os

import pytest

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

ksupport = pytest.importorskip('ibis.impala.kudu_support')
kudu = pytest.importorskip('kudu')

from ibis.expr.tests.mocks import MockConnection  # noqa: E402
from ibis.impala.client import build_ast  # noqa: E402
from ibis.impala.tests.common import IbisTestEnv, ImpalaE2E  # noqa: E402
from ibis.tests.util import assert_equal  # noqa: E402
import ibis.expr.datatypes as dt  # noqa: E402
import ibis.util as util  # noqa: E402
import ibis  # noqa: E402

pytestmark = pytest.mark.kudu


class KuduImpalaTestEnv(IbisTestEnv):

    def __init__(self):
        super(KuduImpalaTestEnv, self).__init__()

        # band-aid until Kudu support merged into Impala mainline
        self.test_host = os.getenv('IBIS_TEST_KIMPALA_HOST',
                                   'quickstart.cloudera')

        # XXX
        self.impala_host = self.test_host
        self.impala_port = 21050
        self.master_host = os.getenv('IBIS_TEST_KUDU_MASTER', self.test_host)
        self.master_port = os.getenv('IBIS_TEST_KUDU_MASTER_PORT', 7051)
        self.nn_host = os.environ.get('IBIS_TEST_KUDU_NN_HOST', self.test_host)

        self.webhdfs_port = int(os.environ.get('IBIS_TEST_WEBHDFS_PORT',
                                               50070))
        self.hdfs_superuser = os.environ.get('IBIS_TEST_HDFS_SUPERUSER',
                                             'hdfs')


ENV = KuduImpalaTestEnv()


class TestKuduTools(unittest.TestCase):

    # Test schema conversion, DDL statements, etc.

    def test_kudu_schema_convert(self):
        spec = [
            # name, type, is_nullable, is_primary_key
            ('a', dt.Int8(False), 'int8', False, True),
            ('b', dt.Int16(False), 'int16', False, True),
            ('c', dt.Int32(False), 'int32', False, False),
            ('d', dt.Int64(True), 'int64', True, False),
            ('e', dt.String(True), 'string', True, False),
            ('f', dt.Boolean(False), 'bool', False, False),
            ('g', dt.Float(False), 'float', False, False),
            ('h', dt.Double(True), 'double', True, False),

            # TODO
            # ('i', 'binary', False, False),

            ('j', dt.Timestamp(True), 'timestamp', True, False)
        ]

        builder = kudu.schema_builder()
        primary_keys = []
        ibis_types = []
        for name, itype, type_, is_nullable, is_primary_key in spec:
            builder.add_column(name, type_, nullable=is_nullable)

            if is_primary_key:
                primary_keys.append(name)

            ibis_types.append((name, itype))

        builder.set_primary_keys(primary_keys)
        kschema = builder.build()

        ischema = ksupport.schema_kudu_to_ibis(kschema)
        expected = ibis.schema(ibis_types)

        assert_equal(ischema, expected)

    def test_create_external_ddl(self):
        schema = ibis.schema([('key1', 'int32'),
                              ('key2', 'int64'),
                              ('value1', 'double')])

        stmt = ksupport.CreateTableKudu('impala_name', 'kudu_name',
                                        ['master1.d.com:7051',
                                         'master2.d.com:7051'],
                                        schema, ['key1', 'key2'])

        result = stmt.compile()
        expected = """\
CREATE EXTERNAL TABLE `impala_name`
(`key1` int,
 `key2` bigint,
 `value1` double)
TBLPROPERTIES (
  'kudu.key_columns'='key1, key2',
  'kudu.master_addresses'='master1.d.com:7051, master2.d.com:7051',
  'kudu.table_name'='kudu_name',
  'storage_handler'='com.cloudera.kudu.hive.KuduStorageHandler'
)"""
        assert result == expected

    def test_ctas_ddl(self):
        con = MockConnection()

        select = build_ast(con.table('test1')).queries[0]
        statement = ksupport.CTASKudu(
            'another_table', 'kudu_name', ['dom.d.com:7051'],
            select, ['string_col'], external=True,
            can_exist=False, database='foo')
        result = statement.compile()

        expected = """\
CREATE EXTERNAL TABLE foo.`another_table`
TBLPROPERTIES (
  'kudu.key_columns'='string_col',
  'kudu.master_addresses'='dom.d.com:7051',
  'kudu.table_name'='kudu_name',
  'storage_handler'='com.cloudera.kudu.hive.KuduStorageHandler'
) AS
SELECT *
FROM test1"""
        assert result == expected


class TestKuduE2E(ImpalaE2E, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ImpalaE2E.setup_e2e(cls, ENV)

        cls.temp_tables = []

        cls.kclient = kudu.connect(cls.env.master_host, cls.env.master_port)

        cls.con.kudu.connect(cls.env.master_host, cls.env.master_port)

    def _new_kudu_example_table(self, kschema):
        kudu_name = 'ibis-tmp-{0}'.format(util.guid())

        self.kclient.create_table(kudu_name, kschema)
        self.temp_tables.append(kudu_name)

        return kudu_name

    @classmethod
    def tearDownClass(cls):
        cls.teardown_e2e(cls)

        for table in cls.temp_tables:
            cls.kclient.delete_table(table)

    @classmethod
    def example_schema(cls):
        builder = kudu.schema_builder()
        builder.add_column('key', kudu.int32, nullable=False)
        builder.add_column('int_val', kudu.int32)
        builder.add_column('string_val', kudu.string)
        builder.set_primary_keys(['key'])

        return builder.build()

    def _write_example_data(self, table_name, nrows=100):
        table = self.kclient.table(table_name)
        session = self.kclient.new_session()
        for i in range(nrows):
            op = table.new_insert()
            row = i, i * 2, 'hello_%d' % i
            op['key'] = row[0]
            op['int_val'] = row[1]
            op['string_val'] = row[2]
            session.apply(op)
        session.flush()

    def test_external_kudu_table(self):
        kschema = self.example_schema()
        kudu_name = self._new_kudu_example_table(kschema)

        nrows = 100
        self._write_example_data(kudu_name, nrows)

        table = self.con.kudu.table(kudu_name)
        result = table.execute()
        assert len(result) == 100

        ischema = ksupport.schema_kudu_to_ibis(kschema, drop_nn=True)
        assert_equal(table.schema(), ischema)

    def test_internal_kudu_table(self):
        kschema = self.example_schema()
        kudu_name = self._new_kudu_example_table(kschema)

        nrows = 100
        self._write_example_data(kudu_name, nrows)

        impala_name = self._temp_impala_name()
        impala_db = self.env.test_data_db
        self.con.kudu.table(kudu_name, name=impala_name,
                            database=impala_db,
                            external=True,
                            persist=True)

        t = self.con.table(impala_name, database=impala_db)
        assert len(t.execute()) == nrows

        # Make internal
        t.set_external(False)
        t.drop()

        assert not self.con.kudu.table_exists(kudu_name)

    def test_create_table_as_select_ctas(self):
        # TODO
        kschema = self.example_schema()
        kudu_name = self._new_kudu_example_table(kschema)

        nrows = 100
        self._write_example_data(kudu_name, nrows)

        impala_name = self._temp_impala_name()
        impala_db = self.env.test_data_db
        self.con.kudu.table(kudu_name, name=impala_name,
                            database=impala_db,
                            external=True,
                            persist=True)

        impala_name2 = self._temp_impala_name()
        expr = self.con.table(impala_name, database=impala_db)

        kudu_name2 = 'ibis-ctas-{0}'.format(util.guid())

        self.con.kudu.create_table(impala_name2, kudu_name2,
                                   primary_keys=['key'],
                                   obj=expr, database=impala_db)

        # TODO: should some stats be automatically computed?
        itable = self.con.table(impala_name2, database=impala_db)
        assert len(itable.execute()) == len(expr.execute())

        ktable = self.kclient.table(kudu_name2)
        assert ktable.schema.primary_keys() == ['key']

    def test_create_empty_internal_table(self):
        kschema = self.example_schema()
        ischema = ksupport.schema_kudu_to_ibis(kschema, drop_nn=True)

        impala_name = self._temp_impala_name()
        kudu_name = 'ibis-empty-{0}'.format(util.guid())

        self.con.kudu.create_table(impala_name, kudu_name,
                                   primary_keys=['key'],
                                   schema=ischema,
                                   database=self.env.test_data_db)

        ktable = self.kclient.table(kudu_name)
        assert ktable.schema.equals(kschema)
        self.temp_tables.append(kudu_name)

    def _temp_impala_name(self):
        return 'kudu_test_{0}'.format(util.guid())
