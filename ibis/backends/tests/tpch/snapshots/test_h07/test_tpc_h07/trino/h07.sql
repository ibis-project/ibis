SELECT
  "t24"."supp_nation",
  "t24"."cust_nation",
  "t24"."l_year",
  "t24"."revenue"
FROM (
  SELECT
    "t23"."supp_nation",
    "t23"."cust_nation",
    "t23"."l_year",
    SUM("t23"."volume") AS "revenue"
  FROM (
    SELECT
      "t22"."supp_nation",
      "t22"."cust_nation",
      "t22"."l_shipdate",
      "t22"."l_extendedprice",
      "t22"."l_discount",
      "t22"."l_year",
      "t22"."volume"
    FROM (
      SELECT
        "t10"."n_name" AS "supp_nation",
        "t16"."n_name" AS "cust_nation",
        "t13"."l_shipdate",
        "t13"."l_extendedprice",
        "t13"."l_discount",
        EXTRACT(year FROM "t13"."l_shipdate") AS "l_year",
        "t13"."l_extendedprice" * (
          1 - "t13"."l_discount"
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
        FROM "supplier" AS "t0"
      ) AS "t12"
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
        FROM "lineitem" AS "t1"
      ) AS "t13"
        ON "t12"."s_suppkey" = "t13"."l_suppkey"
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
        FROM "orders" AS "t2"
      ) AS "t14"
        ON "t14"."o_orderkey" = "t13"."l_orderkey"
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
        FROM "customer" AS "t3"
      ) AS "t15"
        ON "t15"."c_custkey" = "t14"."o_custkey"
      INNER JOIN (
        SELECT
          "t4"."n_nationkey",
          "t4"."n_name",
          "t4"."n_regionkey",
          "t4"."n_comment"
        FROM "nation" AS "t4"
      ) AS "t10"
        ON "t12"."s_nationkey" = "t10"."n_nationkey"
      INNER JOIN (
        SELECT
          "t4"."n_nationkey",
          "t4"."n_name",
          "t4"."n_regionkey",
          "t4"."n_comment"
        FROM "nation" AS "t4"
      ) AS "t16"
        ON "t15"."c_nationkey" = "t16"."n_nationkey"
    ) AS "t22"
    WHERE
      (
        (
          (
            "t22"."cust_nation" = 'FRANCE'
          ) AND (
            "t22"."supp_nation" = 'GERMANY'
          )
        )
        OR (
          (
            "t22"."cust_nation" = 'GERMANY'
          ) AND (
            "t22"."supp_nation" = 'FRANCE'
          )
        )
      )
      AND "t22"."l_shipdate" BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
  ) AS "t23"
  GROUP BY
    1,
    2,
    3
) AS "t24"
ORDER BY
  "t24"."supp_nation" ASC,
  "t24"."cust_nation" ASC,
  "t24"."l_year" ASC