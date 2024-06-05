SELECT
  *
FROM (
  SELECT
    "t5"."c_count",
    COUNT(*) AS "custdist"
  FROM (
    SELECT
      "t5"."c_custkey",
      "t5"."c_count"
    FROM (
      SELECT
        "t4"."c_custkey",
        COUNT("t4"."o_orderkey") AS "c_count"
      FROM (
        SELECT
          "t4"."c_name",
          "t4"."c_address",
          "t4"."c_nationkey",
          "t4"."c_phone",
          "t4"."c_acctbal",
          "t4"."c_mktsegment",
          "t4"."c_comment",
          "t4"."o_orderkey",
          "t4"."o_custkey",
          "t4"."o_orderstatus",
          "t4"."o_totalprice",
          "t4"."o_orderdate",
          "t4"."o_orderpriority",
          "t4"."o_clerk",
          "t4"."o_shippriority",
          "t4"."o_comment",
          "t4"."c_custkey"
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
      ) AS t4
      GROUP BY
        "t4"."c_custkey"
    ) AS "t5"
  ) AS t5
  GROUP BY
    "t5"."c_count"
) AS "t6"
ORDER BY
  "t6"."custdist" DESC NULLS LAST,
  "t6"."c_count" DESC NULLS LAST