SELECT
  CAST((
    SUM(
      IF("t8"."p_type" LIKE 'PROMO%', "t8"."l_extendedprice" * (
        1 - "t8"."l_discount"
      ), 0)
    ) * 100
  ) AS DOUBLE) / SUM("t8"."l_extendedprice" * (
    1 - "t8"."l_discount"
  )) AS "promo_revenue"
FROM (
  SELECT
    "t7"."l_orderkey",
    "t7"."l_partkey",
    "t7"."l_suppkey",
    "t7"."l_linenumber",
    "t7"."l_quantity",
    "t7"."l_extendedprice",
    "t7"."l_discount",
    "t7"."l_tax",
    "t7"."l_returnflag",
    "t7"."l_linestatus",
    "t7"."l_shipdate",
    "t7"."l_commitdate",
    "t7"."l_receiptdate",
    "t7"."l_shipinstruct",
    "t7"."l_shipmode",
    "t7"."l_comment",
    "t7"."p_partkey",
    "t7"."p_name",
    "t7"."p_mfgr",
    "t7"."p_brand",
    "t7"."p_type",
    "t7"."p_size",
    "t7"."p_container",
    "t7"."p_retailprice",
    "t7"."p_comment"
  FROM (
    SELECT
      "t4"."l_orderkey",
      "t4"."l_partkey",
      "t4"."l_suppkey",
      "t4"."l_linenumber",
      "t4"."l_quantity",
      "t4"."l_extendedprice",
      "t4"."l_discount",
      "t4"."l_tax",
      "t4"."l_returnflag",
      "t4"."l_linestatus",
      "t4"."l_shipdate",
      "t4"."l_commitdate",
      "t4"."l_receiptdate",
      "t4"."l_shipinstruct",
      "t4"."l_shipmode",
      "t4"."l_comment",
      "t5"."p_partkey",
      "t5"."p_name",
      "t5"."p_mfgr",
      "t5"."p_brand",
      "t5"."p_type",
      "t5"."p_size",
      "t5"."p_container",
      "t5"."p_retailprice",
      "t5"."p_comment"
    FROM (
      SELECT
        "t0"."l_orderkey",
        "t0"."l_partkey",
        "t0"."l_suppkey",
        "t0"."l_linenumber",
        CAST("t0"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
        CAST("t0"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
        CAST("t0"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
        CAST("t0"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
        "t0"."l_returnflag",
        "t0"."l_linestatus",
        "t0"."l_shipdate",
        "t0"."l_commitdate",
        "t0"."l_receiptdate",
        "t0"."l_shipinstruct",
        "t0"."l_shipmode",
        "t0"."l_comment"
      FROM "lineitem" AS "t0"
    ) AS "t4"
    INNER JOIN (
      SELECT
        "t1"."p_partkey",
        "t1"."p_name",
        "t1"."p_mfgr",
        "t1"."p_brand",
        "t1"."p_type",
        "t1"."p_size",
        "t1"."p_container",
        CAST("t1"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
        "t1"."p_comment"
      FROM "part" AS "t1"
    ) AS "t5"
      ON "t4"."l_partkey" = "t5"."p_partkey"
  ) AS "t7"
  WHERE
    "t7"."l_shipdate" >= FROM_ISO8601_DATE('1995-09-01')
    AND "t7"."l_shipdate" < FROM_ISO8601_DATE('1995-10-01')
) AS "t8"