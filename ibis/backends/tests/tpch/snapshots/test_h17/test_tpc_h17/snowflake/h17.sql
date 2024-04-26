WITH "t2" AS (
  SELECT
    "t0"."L_ORDERKEY" AS "l_orderkey",
    "t0"."L_PARTKEY" AS "l_partkey",
    "t0"."L_SUPPKEY" AS "l_suppkey",
    "t0"."L_LINENUMBER" AS "l_linenumber",
    "t0"."L_QUANTITY" AS "l_quantity",
    "t0"."L_EXTENDEDPRICE" AS "l_extendedprice",
    "t0"."L_DISCOUNT" AS "l_discount",
    "t0"."L_TAX" AS "l_tax",
    "t0"."L_RETURNFLAG" AS "l_returnflag",
    "t0"."L_LINESTATUS" AS "l_linestatus",
    "t0"."L_SHIPDATE" AS "l_shipdate",
    "t0"."L_COMMITDATE" AS "l_commitdate",
    "t0"."L_RECEIPTDATE" AS "l_receiptdate",
    "t0"."L_SHIPINSTRUCT" AS "l_shipinstruct",
    "t0"."L_SHIPMODE" AS "l_shipmode",
    "t0"."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t0"
)
SELECT
  SUM("t10"."l_extendedprice") / 7.0 AS "avg_yearly"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t6"."l_orderkey",
      "t6"."l_partkey",
      "t6"."l_suppkey",
      "t6"."l_linenumber",
      "t6"."l_quantity",
      "t6"."l_extendedprice",
      "t6"."l_discount",
      "t6"."l_tax",
      "t6"."l_returnflag",
      "t6"."l_linestatus",
      "t6"."l_shipdate",
      "t6"."l_commitdate",
      "t6"."l_receiptdate",
      "t6"."l_shipinstruct",
      "t6"."l_shipmode",
      "t6"."l_comment",
      "t5"."p_partkey",
      "t5"."p_name",
      "t5"."p_mfgr",
      "t5"."p_brand",
      "t5"."p_type",
      "t5"."p_size",
      "t5"."p_container",
      "t5"."p_retailprice",
      "t5"."p_comment"
    FROM "t2" AS "t6"
    INNER JOIN (
      SELECT
        "t1"."P_PARTKEY" AS "p_partkey",
        "t1"."P_NAME" AS "p_name",
        "t1"."P_MFGR" AS "p_mfgr",
        "t1"."P_BRAND" AS "p_brand",
        "t1"."P_TYPE" AS "p_type",
        "t1"."P_SIZE" AS "p_size",
        "t1"."P_CONTAINER" AS "p_container",
        "t1"."P_RETAILPRICE" AS "p_retailprice",
        "t1"."P_COMMENT" AS "p_comment"
      FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t1"
    ) AS "t5"
      ON "t5"."p_partkey" = "t6"."l_partkey"
  ) AS "t7"
  WHERE
    "t7"."p_brand" = 'Brand#23'
    AND "t7"."p_container" = 'MED BOX'
    AND "t7"."l_quantity" < (
      (
        SELECT
          AVG("t8"."l_quantity") AS "Mean(l_quantity)"
        FROM (
          SELECT
            *
          FROM "t2" AS "t4"
          WHERE
            "t4"."l_partkey" = "t7"."p_partkey"
        ) AS "t8"
      ) * 0.2
    )
) AS "t10"