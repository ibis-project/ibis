WITH "t5" AS (
  SELECT
    "t4"."n_nationkey",
    "t4"."n_name",
    "t4"."n_regionkey",
    "t4"."n_comment"
  FROM "hive"."ibis_sf1"."nation" AS "t4"
)
SELECT
  "t20"."supp_nation",
  "t20"."cust_nation",
  "t20"."l_year",
  "t20"."revenue"
FROM (
  SELECT
    "t19"."supp_nation",
    "t19"."cust_nation",
    "t19"."l_year",
    SUM("t19"."volume") AS "revenue"
  FROM (
    SELECT
      "t18"."supp_nation",
      "t18"."cust_nation",
      "t18"."l_shipdate",
      "t18"."l_extendedprice",
      "t18"."l_discount",
      "t18"."l_year",
      "t18"."volume"
    FROM (
      SELECT
        "t15"."n_name" AS "supp_nation",
        "t17"."n_name" AS "cust_nation",
        "t12"."l_shipdate",
        "t12"."l_extendedprice",
        "t12"."l_discount",
        EXTRACT(year FROM "t12"."l_shipdate") AS "l_year",
        "t12"."l_extendedprice" * (
          1 - "t12"."l_discount"
        ) AS "volume"
      FROM (
        SELECT
          "t0"."s_suppkey",
          "t0"."s_name",
          "t0"."s_address",
          "t0"."s_nationkey",
          "t0"."s_phone",
          CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t0"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t0"
      ) AS "t11"
      INNER JOIN (
        SELECT
          "t1"."l_orderkey",
          "t1"."l_partkey",
          "t1"."l_suppkey",
          "t1"."l_linenumber",
          CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t1"."l_returnflag",
          "t1"."l_linestatus",
          "t1"."l_shipdate",
          "t1"."l_commitdate",
          "t1"."l_receiptdate",
          "t1"."l_shipinstruct",
          "t1"."l_shipmode",
          "t1"."l_comment"
        FROM "hive"."ibis_sf1"."lineitem" AS "t1"
      ) AS "t12"
        ON "t11"."s_suppkey" = "t12"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."o_orderkey",
          "t2"."o_custkey",
          "t2"."o_orderstatus",
          CAST("t2"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t2"."o_orderdate",
          "t2"."o_orderpriority",
          "t2"."o_clerk",
          "t2"."o_shippriority",
          "t2"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t2"
      ) AS "t13"
        ON "t13"."o_orderkey" = "t12"."l_orderkey"
      INNER JOIN (
        SELECT
          "t3"."c_custkey",
          "t3"."c_name",
          "t3"."c_address",
          "t3"."c_nationkey",
          "t3"."c_phone",
          CAST("t3"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t3"."c_mktsegment",
          "t3"."c_comment"
        FROM "hive"."ibis_sf1"."customer" AS "t3"
      ) AS "t14"
        ON "t14"."c_custkey" = "t13"."o_custkey"
      INNER JOIN "t5" AS "t15"
        ON "t11"."s_nationkey" = "t15"."n_nationkey"
      INNER JOIN "t5" AS "t17"
        ON "t14"."c_nationkey" = "t17"."n_nationkey"
    ) AS "t18"
    WHERE
      (
        (
          (
            "t18"."cust_nation" = 'FRANCE'
          ) AND (
            "t18"."supp_nation" = 'GERMANY'
          )
        )
        OR (
          (
            "t18"."cust_nation" = 'GERMANY'
          ) AND (
            "t18"."supp_nation" = 'FRANCE'
          )
        )
      )
      AND "t18"."l_shipdate" BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
  ) AS "t19"
  GROUP BY
    1,
    2,
    3
) AS "t20"
ORDER BY
  "t20"."supp_nation" ASC,
  "t20"."cust_nation" ASC,
  "t20"."l_year" ASC