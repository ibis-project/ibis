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

from ibis.client import SQLClient
from ibis.expr.datatypes import Schema
import ibis


class MockConnection(SQLClient):

    _tables = {
        'alltypes': [
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean'),
            ('i', 'timestamp')
        ],
        'star1': [
            ('c', 'int32'),
            ('f', 'double'),
            ('foo_id', 'string'),
            ('bar_id', 'string'),
        ],
        'star2': [
            ('foo_id', 'string'),
            ('value1', 'double'),
            ('value3', 'double')
        ],
        'star3': [
            ('bar_id', 'string'),
            ('value2', 'double')
        ],
        'test1': [
            ('c', 'int32'),
            ('f', 'double'),
            ('g', 'string')
        ],
        'test2': [
            ('key', 'string'),
            ('value', 'double')
        ],
        'tpch_region': [
            ('r_regionkey', 'int16'),
            ('r_name', 'string'),
            ('r_comment', 'string')
        ],
        'tpch_nation': [
            ('n_nationkey', 'int16'),
            ('n_name', 'string'),
            ('n_regionkey', 'int16'),
            ('n_comment', 'string')
        ],
        'tpch_lineitem': [
            ('l_orderkey', 'int64'),
            ('l_partkey', 'int64'),
            ('l_suppkey', 'int64'),
            ('l_linenumber', 'int32'),
            ('l_quantity', 'decimal(12,2)'),
            ('l_extendedprice', 'decimal(12,2)'),
            ('l_discount', 'decimal(12,2)'),
            ('l_tax', 'decimal(12,2)'),
            ('l_returnflag', 'string'),
            ('l_linestatus', 'string'),
            ('l_shipdate', 'string'),
            ('l_commitdate', 'string'),
            ('l_receiptdate', 'string'),
            ('l_shipinstruct', 'string'),
            ('l_shipmode', 'string'),
            ('l_comment', 'string')
        ],
        'tpch_customer': [
            ('c_custkey', 'int64'),
            ('c_name', 'string'),
            ('c_address', 'string'),
            ('c_nationkey', 'int16'),
            ('c_phone', 'string'),
            ('c_acctbal', 'decimal'),
            ('c_mktsegment', 'string'),
            ('c_comment', 'string')
        ],
        'tpch_orders': [
            ('o_orderkey', 'int64'),
            ('o_custkey', 'int64'),
            ('o_orderstatus', 'string'),
            ('o_totalprice', 'decimal(12,2)'),
            ('o_orderdate', 'string'),
            ('o_orderpriority', 'string'),
            ('o_clerk', 'string'),
            ('o_shippriority', 'int32'),
            ('o_comment', 'string')
        ],
        'functional_alltypes': [
            ('id', 'int32'),
            ('bool_col', 'boolean'),
            ('tinyint_col', 'int8'),
            ('smallint_col', 'int16'),
            ('int_col', 'int32'),
            ('bigint_col', 'int64'),
            ('float_col', 'float'),
            ('double_col', 'double'),
            ('date_string_col', 'string'),
            ('string_col', 'string'),
            ('timestamp_col', 'timestamp'),
            ('year', 'int32'),
            ('month', 'int32')
        ],
        'airlines': [
            ('year', 'int32'),
            ('month', 'int32'),
            ('day', 'int32'),
            ('dayofweek', 'int32'),
            ('dep_time', 'int32'),
            ('crs_dep_time', 'int32'),
            ('arr_time', 'int32'),
            ('crs_arr_time', 'int32'),
            ('carrier', 'string'),
            ('flight_num', 'int32'),
            ('tail_num', 'int32'),
            ('actual_elapsed_time', 'int32'),
            ('crs_elapsed_time', 'int32'),
            ('airtime', 'int32'),
            ('arrdelay', 'int32'),
            ('depdelay', 'int32'),
            ('origin', 'string'),
            ('dest', 'string'),
            ('distance', 'int32'),
            ('taxi_in', 'int32'),
            ('taxi_out', 'int32'),
            ('cancelled', 'int32'),
            ('cancellation_code', 'string'),
            ('diverted', 'int32'),
            ('carrier_delay', 'int32'),
            ('weather_delay', 'int32'),
            ('nas_delay', 'int32'),
            ('security_delay', 'int32'),
            ('late_aircraft_delay', 'int32')
        ],
        'tpcds_customer': [
            ('c_customer_sk', 'int64'),
            ('c_customer_id', 'string'),
            ('c_current_cdemo_sk', 'int32'),
            ('c_current_hdemo_sk', 'int32'),
            ('c_current_addr_sk', 'int32'),
            ('c_first_shipto_date_sk', 'int32'),
            ('c_first_sales_date_sk', 'int32'),
            ('c_salutation', 'string'),
            ('c_first_name', 'string'),
            ('c_last_name', 'string'),
            ('c_preferred_cust_flag', 'string'),
            ('c_birth_day', 'int32'),
            ('c_birth_month', 'int32'),
            ('c_birth_year', 'int32'),
            ('c_birth_country', 'string'),
            ('c_login', 'string'),
            ('c_email_address', 'string'),
            ('c_last_review_date', 'string')],
        'tpcds_customer_address': [
            ('ca_address_sk', 'bigint'),
            ('ca_address_id', 'string'),
            ('ca_street_number', 'string'),
            ('ca_street_name', 'string'),
            ('ca_street_type', 'string'),
            ('ca_suite_number', 'string'),
            ('ca_city', 'string'),
            ('ca_county', 'string'),
            ('ca_state', 'string'),
            ('ca_zip', 'string'),
            ('ca_country', 'string'),
            ('ca_gmt_offset', 'decimal(5,2)'),
            ('ca_location_type', 'string')],
        'tpcds_customer_demographics': [
            ('cd_demo_sk', 'bigint'),
            ('cd_gender', 'string'),
            ('cd_marital_status', 'string'),
            ('cd_education_status', 'string'),
            ('cd_purchase_estimate', 'int'),
            ('cd_credit_rating', 'string'),
            ('cd_dep_count', 'int'),
            ('cd_dep_employed_count', 'int'),
            ('cd_dep_college_count', 'int')],
        'tpcds_date_dim': [
            ('d_date_sk', 'bigint'),
            ('d_date_id', 'string'),
            ('d_date', 'string'),
            ('d_month_seq', 'int'),
            ('d_week_seq', 'int'),
            ('d_quarter_seq', 'int'),
            ('d_year', 'int'),
            ('d_dow', 'int'),
            ('d_moy', 'int'),
            ('d_dom', 'int'),
            ('d_qoy', 'int'),
            ('d_fy_year', 'int'),
            ('d_fy_quarter_seq', 'int'),
            ('d_fy_week_seq', 'int'),
            ('d_day_name', 'string'),
            ('d_quarter_name', 'string'),
            ('d_holiday', 'string'),
            ('d_weekend', 'string'),
            ('d_following_holiday', 'string'),
            ('d_first_dom', 'int'),
            ('d_last_dom', 'int'),
            ('d_same_day_ly', 'int'),
            ('d_same_day_lq', 'int'),
            ('d_current_day', 'string'),
            ('d_current_week', 'string'),
            ('d_current_month', 'string'),
            ('d_current_quarter', 'string'),
            ('d_current_year', 'string')],
        'tpcds_household_demographics': [
            ('hd_demo_sk', 'bigint'),
            ('hd_income_band_sk', 'int'),
            ('hd_buy_potential', 'string'),
            ('hd_dep_count', 'int'),
            ('hd_vehicle_count', 'int')],
        'tpcds_item': [
            ('i_item_sk', 'bigint'),
            ('i_item_id', 'string'),
            ('i_rec_start_date', 'string'),
            ('i_rec_end_date', 'string'),
            ('i_item_desc', 'string'),
            ('i_current_price', 'decimal(7,2)'),
            ('i_wholesale_cost', 'decimal(7,2)'),
            ('i_brand_id', 'int'),
            ('i_brand', 'string'),
            ('i_class_id', 'int'),
            ('i_class', 'string'),
            ('i_category_id', 'int'),
            ('i_category', 'string'),
            ('i_manufact_id', 'int'),
            ('i_manufact', 'string'),
            ('i_size', 'string'),
            ('i_formulation', 'string'),
            ('i_color', 'string'),
            ('i_units', 'string'),
            ('i_container', 'string'),
            ('i_manager_id', 'int'),
            ('i_product_name', 'string')],
        'tpcds_promotion': [
            ('p_promo_sk', 'bigint'),
            ('p_promo_id', 'string'),
            ('p_start_date_sk', 'int'),
            ('p_end_date_sk', 'int'),
            ('p_item_sk', 'int'),
            ('p_cost', 'decimal(15,2)'),
            ('p_response_target', 'int'),
            ('p_promo_name', 'string'),
            ('p_channel_dmail', 'string'),
            ('p_channel_email', 'string'),
            ('p_channel_catalog', 'string'),
            ('p_channel_tv', 'string'),
            ('p_channel_radio', 'string'),
            ('p_channel_press', 'string'),
            ('p_channel_event', 'string'),
            ('p_channel_demo', 'string'),
            ('p_channel_details', 'string'),
            ('p_purpose', 'string'),
            ('p_discount_active', 'string')],
        'tpcds_store': [
            ('s_store_sk', 'bigint'),
            ('s_store_id', 'string'),
            ('s_rec_start_date', 'string'),
            ('s_rec_end_date', 'string'),
            ('s_closed_date_sk', 'int'),
            ('s_store_name', 'string'),
            ('s_number_employees', 'int'),
            ('s_floor_space', 'int'),
            ('s_hours', 'string'),
            ('s_manager', 'string'),
            ('s_market_id', 'int'),
            ('s_geography_class', 'string'),
            ('s_market_desc', 'string'),
            ('s_market_manager', 'string'),
            ('s_division_id', 'int'),
            ('s_division_name', 'string'),
            ('s_company_id', 'int'),
            ('s_company_name', 'string'),
            ('s_street_number', 'string'),
            ('s_street_name', 'string'),
            ('s_street_type', 'string'),
            ('s_suite_number', 'string'),
            ('s_city', 'string'),
            ('s_county', 'string'),
            ('s_state', 'string'),
            ('s_zip', 'string'),
            ('s_country', 'string'),
            ('s_gmt_offset', 'decimal(5,2)'),
            ('s_tax_precentage', 'decimal(5,2)')],
        'tpcds_store_sales': [
            ('ss_sold_time_sk', 'bigint'),
            ('ss_item_sk', 'bigint'),
            ('ss_customer_sk', 'bigint'),
            ('ss_cdemo_sk', 'bigint'),
            ('ss_hdemo_sk', 'bigint'),
            ('ss_addr_sk', 'bigint'),
            ('ss_store_sk', 'bigint'),
            ('ss_promo_sk', 'bigint'),
            ('ss_ticket_number', 'int'),
            ('ss_quantity', 'int'),
            ('ss_wholesale_cost', 'decimal(7,2)'),
            ('ss_list_price', 'decimal(7,2)'),
            ('ss_sales_price', 'decimal(7,2)'),
            ('ss_ext_discount_amt', 'decimal(7,2)'),
            ('ss_ext_sales_price', 'decimal(7,2)'),
            ('ss_ext_wholesale_cost', 'decimal(7,2)'),
            ('ss_ext_list_price', 'decimal(7,2)'),
            ('ss_ext_tax', 'decimal(7,2)'),
            ('ss_coupon_amt', 'decimal(7,2)'),
            ('ss_net_paid', 'decimal(7,2)'),
            ('ss_net_paid_inc_tax', 'decimal(7,2)'),
            ('ss_net_profit', 'decimal(7,2)'),
            ('ss_sold_date_sk', 'bigint')],
        'tpcds_time_dim': [
            ('t_time_sk', 'bigint'),
            ('t_time_id', 'string'),
            ('t_time', 'int'),
            ('t_hour', 'int'),
            ('t_minute', 'int'),
            ('t_second', 'int'),
            ('t_am_pm', 'string'),
            ('t_shift', 'string'),
            ('t_sub_shift', 'string'),
            ('t_meal_time', 'string')]
    }

    def __init__(self):
        self.executed_queries = []

    def _get_table_schema(self, name):
        name = name.replace('`', '')
        return Schema.from_tuples(self._tables[name])

    def _build_ast(self, expr):
        from ibis.impala.compiler import build_ast
        return build_ast(expr)

    def execute(self, expr, limit=None, async=False):
        if async:
            raise NotImplementedError
        ast = self._build_ast_ensure_limit(expr, limit)
        for query in ast.queries:
            self.executed_queries.append(query.compile())
        return None

    def compile(self, expr, limit=None):
        ast = self._build_ast_ensure_limit(expr, limit)
        queries = [q.compile() for q in ast.queries]
        return queries[0] if len(queries) == 1 else queries


_all_types_schema = [
    ('a', 'int8'),
    ('b', 'int16'),
    ('c', 'int32'),
    ('d', 'int64'),
    ('e', 'float'),
    ('f', 'double'),
    ('g', 'string'),
    ('h', 'boolean')
]


class BasicTestCase(object):

    def setUp(self):
        self.schema = _all_types_schema
        self.schema_dict = dict(self.schema)
        self.table = ibis.table(self.schema, 'schema')

        self.int_cols = ['a', 'b', 'c', 'd']
        self.bool_cols = ['h']
        self.float_cols = ['e', 'f']

        self.con = MockConnection()
