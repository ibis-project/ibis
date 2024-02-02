SELECT
  SUM("t7"."l_extendedprice") / CAST(7.0 AS DOUBLE) AS "avg_yearly"
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
    "t4"."p_partkey",
    "t4"."p_name",
    "t4"."p_mfgr",
    "t4"."p_brand",
    "t4"."p_type",
    "t4"."p_size",
    "t4"."p_container",
    "t4"."p_retailprice",
    "t4"."p_comment"
  FROM (
    SELECT
      "t2"."l_orderkey",
      "t2"."l_partkey",
      "t2"."l_suppkey",
      "t2"."l_linenumber",
      "t2"."l_quantity",
      "t2"."l_extendedprice",
      "t2"."l_discount",
      "t2"."l_tax",
      "t2"."l_returnflag",
      "t2"."l_linestatus",
      "t2"."l_shipdate",
      "t2"."l_commitdate",
      "t2"."l_receiptdate",
      "t2"."l_shipinstruct",
      "t2"."l_shipmode",
      "t2"."l_comment",
      "t3"."p_partkey",
      "t3"."p_name",
      "t3"."p_mfgr",
      "t3"."p_brand",
      "t3"."p_type",
      "t3"."p_size",
      "t3"."p_container",
      "t3"."p_retailprice",
      "t3"."p_comment"
    FROM "lineitem" AS "t2"
    INNER JOIN "part" AS "t3"
      ON "t3"."p_partkey" = "t2"."l_partkey"
  ) AS "t4"
  WHERE
    "t4"."p_brand" = 'Brand#23'
    AND "t4"."p_container" = 'MED BOX'
    AND "t4"."l_quantity" < (
      (
        SELECT
          AVG("t5"."l_quantity") AS "Mean(l_quantity)"
        FROM (
          SELECT
            "t0"."l_orderkey",
            "t0"."l_partkey",
            "t0"."l_suppkey",
            "t0"."l_linenumber",
            "t0"."l_quantity",
            "t0"."l_extendedprice",
            "t0"."l_discount",
            "t0"."l_tax",
            "t0"."l_returnflag",
            "t0"."l_linestatus",
            "t0"."l_shipdate",
            "t0"."l_commitdate",
            "t0"."l_receiptdate",
            "t0"."l_shipinstruct",
            "t0"."l_shipmode",
            "t0"."l_comment"
          FROM "lineitem" AS "t0"
          WHERE
            "t0"."l_partkey" = "t4"."p_partkey"
        ) AS "t5"
      ) * CAST(0.2 AS DOUBLE)
    )
) AS "t7"