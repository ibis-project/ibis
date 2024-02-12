WITH "t7" AS (
  SELECT
    "t3"."L_ORDERKEY" AS "l_orderkey",
    "t3"."L_PARTKEY" AS "l_partkey",
    "t3"."L_SUPPKEY" AS "l_suppkey",
    "t3"."L_LINENUMBER" AS "l_linenumber",
    "t3"."L_QUANTITY" AS "l_quantity",
    "t3"."L_EXTENDEDPRICE" AS "l_extendedprice",
    "t3"."L_DISCOUNT" AS "l_discount",
    "t3"."L_TAX" AS "l_tax",
    "t3"."L_RETURNFLAG" AS "l_returnflag",
    "t3"."L_LINESTATUS" AS "l_linestatus",
    "t3"."L_SHIPDATE" AS "l_shipdate",
    "t3"."L_COMMITDATE" AS "l_commitdate",
    "t3"."L_RECEIPTDATE" AS "l_receiptdate",
    "t3"."L_SHIPINSTRUCT" AS "l_shipinstruct",
    "t3"."L_SHIPMODE" AS "l_shipmode",
    "t3"."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t3"
)
SELECT
  "t19"."s_name",
  "t19"."numwait"
FROM (
  SELECT
    "t18"."s_name",
    COUNT(*) AS "numwait"
  FROM (
    SELECT
      "t15"."l1_orderkey",
      "t15"."o_orderstatus",
      "t15"."l_receiptdate",
      "t15"."l_commitdate",
      "t15"."l1_suppkey",
      "t15"."s_name",
      "t15"."n_name"
    FROM (
      SELECT
        "t12"."l_orderkey" AS "l1_orderkey",
        "t9"."o_orderstatus",
        "t12"."l_receiptdate",
        "t12"."l_commitdate",
        "t12"."l_suppkey" AS "l1_suppkey",
        "t8"."s_name",
        "t10"."n_name"
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
      INNER JOIN "t7" AS "t12"
        ON "t8"."s_suppkey" = "t12"."l_suppkey"
      INNER JOIN (
        SELECT
          "t1"."O_ORDERKEY" AS "o_orderkey",
          "t1"."O_CUSTKEY" AS "o_custkey",
          "t1"."O_ORDERSTATUS" AS "o_orderstatus",
          "t1"."O_TOTALPRICE" AS "o_totalprice",
          "t1"."O_ORDERDATE" AS "o_orderdate",
          "t1"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t1"."O_CLERK" AS "o_clerk",
          "t1"."O_SHIPPRIORITY" AS "o_shippriority",
          "t1"."O_COMMENT" AS "o_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
      ) AS "t9"
        ON "t9"."o_orderkey" = "t12"."l_orderkey"
      INNER JOIN (
        SELECT
          "t2"."N_NATIONKEY" AS "n_nationkey",
          "t2"."N_NAME" AS "n_name",
          "t2"."N_REGIONKEY" AS "n_regionkey",
          "t2"."N_COMMENT" AS "n_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t2"
      ) AS "t10"
        ON "t8"."s_nationkey" = "t10"."n_nationkey"
    ) AS "t15"
    WHERE
      "t15"."o_orderstatus" = 'F'
      AND "t15"."l_receiptdate" > "t15"."l_commitdate"
      AND "t15"."n_name" = 'SAUDI ARABIA'
      AND EXISTS(
        SELECT
          1
        FROM "t7" AS "t13"
        WHERE
          (
            "t13"."l_orderkey" = "t15"."l1_orderkey"
          )
          AND (
            "t13"."l_suppkey" <> "t15"."l1_suppkey"
          )
      )
      AND NOT (
        EXISTS(
          SELECT
            1
          FROM "t7" AS "t14"
          WHERE
            (
              (
                "t14"."l_orderkey" = "t15"."l1_orderkey"
              )
              AND (
                "t14"."l_suppkey" <> "t15"."l1_suppkey"
              )
            )
            AND (
              "t14"."l_receiptdate" > "t14"."l_commitdate"
            )
        )
      )
  ) AS "t18"
  GROUP BY
    1
) AS "t19"
ORDER BY
  "t19"."numwait" DESC NULLS LAST,
  "t19"."s_name" ASC
LIMIT 100