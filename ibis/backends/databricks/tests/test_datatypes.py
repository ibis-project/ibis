from __future__ import annotations

import pytest

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.util import gen_name


@pytest.fixture
def tmp_table(con):
    name = gen_name("databricks_tmp_table")
    yield name
    con.drop_table(name, force=True)


def test_nested(con, tmp_table):
    schema = sch.Schema(
        {
            "nums": "decimal(33, 2)",
            "time": "timestamp('UTC')",
            "operationName": "string",
            "category": "string",
            "tenantId": "string",
            "properties": dt.Struct(
                {
                    "Timestamp": "timestamp('UTC')",
                    "ActionType": "string",
                    "Application": "string",
                    "ApplicationId": "int32",
                    "AppInstanceId": "int32",
                    "AccountObjectId": "string",
                    "AccountId": "string",
                    "AccountDisplayName": "string",
                    "IsAdminOperation": "boolean",
                    "DeviceType": "string",
                    "OSPlatform": "string",
                    "IPAddress": "string",
                    "IsAnonymousProxy": "boolean",
                    "CountryCode": "string",
                    "City": "string",
                    "ISP": "string",
                    "UserAgent": "string",
                    "ActivityType": "string",
                    "ActivityObjects": "string",
                    "ObjectName": "string",
                    "ObjectType": "string",
                    "ObjectId": "string",
                    "ReportId": "string",
                    "AccountType": "string",
                    "IsExternalUser": "boolean",
                    "IsImpersonated": "boolean",
                    "IPTags": "string",
                    "IPCategory": "string",
                    "UserAgentTags": "string",
                    "RawEventData": "string",
                    "AdditionalFields": "string",
                }
            ),
            "Tenant": "string",
            "_rescued_data": "string",
            "timestamp": "timestamp('UTC')",
            "parse_details": dt.Struct(
                {
                    "status": "string",
                    "at": "timestamp('UTC')",
                    "info": dt.Struct({"input-file-name": "string"}),
                }
            ),
            "p_date": "string",
            "foo": "array<struct<a: map<string, array<struct<b: string>>>>>",
            "a": "array<int>",
            "b": "map<string, int>",
        }
    )

    t = con.create_table(tmp_table, schema=schema)
    assert t.schema() == schema
