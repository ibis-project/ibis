"""Initialize default configuration."""
# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ibis.config as cf

cf.register_option('interactive', False, validator=cf.is_bool)
cf.register_option('verbose', False, validator=cf.is_bool)
cf.register_option('verbose_log', None)

cf.register_option(
    'graphviz_repr',
    True,
    """\
Whether to render expressions as GraphViz PNGs when repr-ing in a Jupyter
notebook.
""",
    validator=cf.is_bool,
)

cf.register_option('default_backend', None)

sql_default_limit_doc = """
Number of rows to be retrieved for an unlimited table expression
"""


with cf.config_prefix('sql'):
    cf.register_option('default_limit', 10000, sql_default_limit_doc)


impala_temp_db_doc = """
Database to use for temporary tables, views. functions, etc.
"""

impala_temp_hdfs_path_doc = """
HDFS path for storage of temporary data
"""


with cf.config_prefix('impala'):
    cf.register_option('temp_db', '__ibis_tmp', impala_temp_db_doc)
    cf.register_option(
        'temp_hdfs_path', '/tmp/ibis', impala_temp_hdfs_path_doc
    )


clickhouse_temp_db_doc = """
Database to use for temporary tables, views. functions, etc.
"""

with cf.config_prefix('clickhouse'):
    cf.register_option('temp_db', '__ibis_tmp', clickhouse_temp_db_doc)


with cf.config_prefix('bigquery'):
    cf.register_option('partition_col', 'PARTITIONTIME')


with cf.config_prefix('pandas'):
    cf.register_option(
        'enable_trace',
        False,
        """
Whether enable tracing for pandas execution. See ibis.panads.trace for details.
""",
        validator=cf.is_bool,
    )
