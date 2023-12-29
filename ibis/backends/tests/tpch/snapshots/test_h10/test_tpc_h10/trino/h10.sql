SELECT
  "t17"."c_custkey",
  "t17"."c_name",
  "t17"."revenue",
  "t17"."c_acctbal",
  "t17"."n_name",
  "t17"."c_address",
  "t17"."c_phone",
  "t17"."c_comment"
FROM (
  SELECT
    "t16"."c_custkey",
    "t16"."c_name",
    "t16"."c_acctbal",
    "t16"."n_name",
    "t16"."c_address",
    "t16"."c_phone",
    "t16"."c_comment",
    SUM("t16"."l_extendedprice" * (
      1 - "t16"."l_discount"
    )) AS "revenue"
  FROM (
    SELECT
      "t15"."c_custkey",
      "t15"."c_name",
      "t15"."c_address",
      "t15"."c_nationkey",
      "t15"."c_phone",
      "t15"."c_acctbal",
      "t15"."c_mktsegment",
      "t15"."c_comment",
      "t15"."o_orderkey",
      "t15"."o_custkey",
      "t15"."o_orderstatus",
      "t15"."o_totalprice",
      "t15"."o_orderdate",
      "t15"."o_orderpriority",
      "t15"."o_clerk",
      "t15"."o_shippriority",
      "t15"."o_comment",
      "t15"."l_orderkey",
      "t15"."l_partkey",
      "t15"."l_suppkey",
      "t15"."l_linenumber",
      "t15"."l_quantity",
      "t15"."l_extendedprice",
      "t15"."l_discount",
      "t15"."l_tax",
      "t15"."l_returnflag",
      "t15"."l_linestatus",
      "t15"."l_shipdate",
      "t15"."l_commitdate",
      "t15"."l_receiptdate",
      "t15"."l_shipinstruct",
      "t15"."l_shipmode",
      "t15"."l_comment",
      "t15"."n_nationkey",
      "t15"."n_name",
      "t15"."n_regionkey",
      "t15"."n_comment"
    FROM (
      SELECT
        "t9"."c_custkey",
        "t9"."c_name",
        "t9"."c_address",
        "t9"."c_nationkey",
        "t9"."c_phone",
        "t9"."c_acctbal",
        "t9"."c_mktsegment",
        "t9"."c_comment",
        "t10"."o_orderkey",
        "t10"."o_custkey",
        "t10"."o_orderstatus",
        "t10"."o_totalprice",
        "t10"."o_orderdate",
        "t10"."o_orderpriority",
        "t10"."o_clerk",
        "t10"."o_shippriority",
        "t10"."o_comment",
        "t11"."l_orderkey",
        "t11"."l_partkey",
        "t11"."l_suppkey",
        "t11"."l_linenumber",
        "t11"."l_quantity",
        "t11"."l_extendedprice",
        "t11"."l_discount",
        "t11"."l_tax",
        "t11"."l_returnflag",
        "t11"."l_linestatus",
        "t11"."l_shipdate",
        "t11"."l_commitdate",
        "t11"."l_receiptdate",
        "t11"."l_shipinstruct",
        "t11"."l_shipmode",
        "t11"."l_comment",
        "t8"."n_nationkey",
        "t8"."n_name",
        "t8"."n_regionkey",
        "t8"."n_comment"
      FROM (
        SELECT
          "t0"."c_custkey",
          "t0"."c_name",
          "t0"."c_address",
          "t0"."c_nationkey",
          "t0"."c_phone",
          CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t0"."c_mktsegment",
          "t0"."c_comment"
        FROM "customer" AS "t0"
      ) AS "t9"
      INNER JOIN (
        SELECT
          "t1"."o_orderkey",
          "t1"."o_custkey",
          "t1"."o_orderstatus",
          CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t1"."o_orderdate",
          "t1"."o_orderpriority",
          "t1"."o_clerk",
          "t1"."o_shippriority",
          "t1"."o_comment"
        FROM "orders" AS "t1"
      ) AS "t10"
        ON "t9"."c_custkey" = "t10"."o_custkey"
      INNER JOIN (
        SELECT
          "t2"."l_orderkey",
          "t2"."l_partkey",
          "t2"."l_suppkey",
          "t2"."l_linenumber",
          CAST("t2"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t2"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t2"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t2"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t2"."l_returnflag",
          "t2"."l_linestatus",
          "t2"."l_shipdate",
          "t2"."l_commitdate",
          "t2"."l_receiptdate",
          "t2"."l_shipinstruct",
          "t2"."l_shipmode",
          "t2"."l_comment"
        FROM "lineitem" AS "t2"
      ) AS "t11"
        ON "t11"."l_orderkey" = "t10"."o_orderkey"
      INNER JOIN (
        SELECT
          "t3"."n_nationkey",
          "t3"."n_name",
          "t3"."n_regionkey",
          "t3"."n_comment"
        FROM "nation" AS "t3"
      ) AS "t8"
        ON "t9"."c_nationkey" = "t8"."n_nationkey"
    ) AS "t15"
    WHERE
      "t15"."o_orderdate" >= FROM_ISO8601_DATE('1993-10-01')
      AND "t15"."o_orderdate" < FROM_ISO8601_DATE('1994-01-01')
      AND "t15"."l_returnflag" = 'R'
  ) AS "t16"
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS "t17"
ORDER BY
  "t17"."revenue" DESC
LIMIT 20