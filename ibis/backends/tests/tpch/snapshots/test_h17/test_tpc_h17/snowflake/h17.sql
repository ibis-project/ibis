SELECT
  SUM("t10"."l_extendedprice") / 7.0 AS "avg_yearly"
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
    ) AS "t4"
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
      ON "t5"."p_partkey" = "t4"."l_partkey"
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
          WHERE
            "t0"."L_PARTKEY" = "t7"."p_partkey"
        ) AS "t8"
      ) * 0.2
    )
) AS "t10"