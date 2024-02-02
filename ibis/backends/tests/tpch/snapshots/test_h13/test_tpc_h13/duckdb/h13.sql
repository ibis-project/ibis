SELECT
  "t6"."c_count",
  "t6"."custdist"
FROM (
  SELECT
    "t5"."c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t4"."c_custkey",
      COUNT("t4"."o_orderkey") AS "c_count"
    FROM (
      SELECT
        "t2"."c_custkey",
        "t2"."c_name",
        "t2"."c_address",
        "t2"."c_nationkey",
        "t2"."c_phone",
        "t2"."c_acctbal",
        "t2"."c_mktsegment",
        "t2"."c_comment",
        "t3"."o_orderkey",
        "t3"."o_custkey",
        "t3"."o_orderstatus",
        "t3"."o_totalprice",
        "t3"."o_orderdate",
        "t3"."o_orderpriority",
        "t3"."o_clerk",
        "t3"."o_shippriority",
        "t3"."o_comment"
      FROM "customer" AS "t2"
      LEFT OUTER JOIN "orders" AS "t3"
        ON "t2"."c_custkey" = "t3"."o_custkey"
        AND NOT (
          "t3"."o_comment" LIKE '%special%requests%'
        )
    ) AS "t4"
    GROUP BY
      1
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."custdist" DESC,
  "t6"."c_count" DESC