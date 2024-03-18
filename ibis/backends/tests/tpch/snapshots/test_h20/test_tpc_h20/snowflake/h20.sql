SELECT
  "t12"."s_name",
  "t12"."s_address"
FROM (
  SELECT
    "t8"."s_suppkey",
    "t8"."s_name",
    "t8"."s_address",
    "t8"."s_nationkey",
    "t8"."s_phone",
    "t8"."s_acctbal",
    "t8"."s_comment",
    "t9"."n_nationkey",
    "t9"."n_name",
    "t9"."n_regionkey",
    "t9"."n_comment"
  FROM (
    SELECT
      "t0"."S_SUPPKEY" AS "s_suppkey",
      "t0"."S_NAME" AS "s_name",
      "t0"."S_ADDRESS" AS "s_address",
      "t0"."S_NATIONKEY" AS "s_nationkey",
      "t0"."S_PHONE" AS "s_phone",
      "t0"."S_ACCTBAL" AS "s_acctbal",
      "t0"."S_COMMENT" AS "s_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t0"
  ) AS "t8"
  INNER JOIN (
    SELECT
      "t2"."N_NATIONKEY" AS "n_nationkey",
      "t2"."N_NAME" AS "n_name",
      "t2"."N_REGIONKEY" AS "n_regionkey",
      "t2"."N_COMMENT" AS "n_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t2"
  ) AS "t9"
    ON "t8"."s_nationkey" = "t9"."n_nationkey"
) AS "t12"
WHERE
  "t12"."n_name" = 'CANADA'
  AND "t12"."s_suppkey" IN (
    SELECT
      "t6"."ps_suppkey"
    FROM (
      SELECT
        "t1"."PS_PARTKEY" AS "ps_partkey",
        "t1"."PS_SUPPKEY" AS "ps_suppkey",
        "t1"."PS_AVAILQTY" AS "ps_availqty",
        "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
        "t1"."PS_COMMENT" AS "ps_comment"
      FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t1"
    ) AS "t6"
    WHERE
      "t6"."ps_partkey" IN (
        SELECT
          "t3"."P_PARTKEY" AS "p_partkey"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS "t3"
        WHERE
          "t3"."P_NAME" LIKE 'forest%'
      )
      AND "t6"."ps_availqty" > (
        (
          SELECT
            SUM("t11"."l_quantity") AS "Sum(l_quantity)"
          FROM (
            SELECT
              "t4"."L_ORDERKEY" AS "l_orderkey",
              "t4"."L_PARTKEY" AS "l_partkey",
              "t4"."L_SUPPKEY" AS "l_suppkey",
              "t4"."L_LINENUMBER" AS "l_linenumber",
              "t4"."L_QUANTITY" AS "l_quantity",
              "t4"."L_EXTENDEDPRICE" AS "l_extendedprice",
              "t4"."L_DISCOUNT" AS "l_discount",
              "t4"."L_TAX" AS "l_tax",
              "t4"."L_RETURNFLAG" AS "l_returnflag",
              "t4"."L_LINESTATUS" AS "l_linestatus",
              "t4"."L_SHIPDATE" AS "l_shipdate",
              "t4"."L_COMMITDATE" AS "l_commitdate",
              "t4"."L_RECEIPTDATE" AS "l_receiptdate",
              "t4"."L_SHIPINSTRUCT" AS "l_shipinstruct",
              "t4"."L_SHIPMODE" AS "l_shipmode",
              "t4"."L_COMMENT" AS "l_comment"
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t4"
            WHERE
              "t4"."L_PARTKEY" = "t6"."ps_partkey"
              AND "t4"."L_SUPPKEY" = "t6"."ps_suppkey"
              AND "t4"."L_SHIPDATE" >= DATE_FROM_PARTS(1994, 1, 1)
              AND "t4"."L_SHIPDATE" < DATE_FROM_PARTS(1995, 1, 1)
          ) AS "t11"
        ) * 0.5
      )
  )
ORDER BY
  "t12"."s_name" ASC