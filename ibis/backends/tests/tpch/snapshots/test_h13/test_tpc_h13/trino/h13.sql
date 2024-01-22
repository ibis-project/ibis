SELECT
  "t8"."c_count",
  "t8"."custdist"
FROM (
  SELECT
    "t7"."c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t6"."c_custkey",
      COUNT("t6"."o_orderkey") AS "c_count"
    FROM (
      SELECT
        "t4"."c_custkey",
        "t4"."c_name",
        "t4"."c_address",
        "t4"."c_nationkey",
        "t4"."c_phone",
        "t4"."c_acctbal",
        "t4"."c_mktsegment",
        "t4"."c_comment",
        "t5"."o_orderkey",
        "t5"."o_custkey",
        "t5"."o_orderstatus",
        "t5"."o_totalprice",
        "t5"."o_orderdate",
        "t5"."o_orderpriority",
        "t5"."o_clerk",
        "t5"."o_shippriority",
        "t5"."o_comment"
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
        FROM "hive"."ibis_sf1"."customer" AS "t0"
      ) AS "t4"
      LEFT OUTER JOIN (
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
        FROM "hive"."ibis_sf1"."orders" AS "t1"
      ) AS "t5"
        ON "t4"."c_custkey" = "t5"."o_custkey"
        AND NOT (
          "t5"."o_comment" LIKE '%special%requests%'
        )
    ) AS "t6"
    GROUP BY
      1
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."custdist" DESC,
  "t8"."c_count" DESC