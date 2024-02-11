SELECT
  "t6"."ancestor_node_sort_order",
  "t6"."n"
FROM (
  SELECT
    "t5"."ancestor_node_sort_order",
    CAST(1 AS TINYINT) AS "n"
  FROM (
    SELECT
      "t2"."product_id",
      "t4"."ancestor_level_name",
      "t4"."ancestor_level_number",
      "t4"."ancestor_node_sort_order",
      "t4"."descendant_node_natural_key",
      "t4"."product_level_name"
    FROM "facts" AS "t2"
    INNER JOIN (
      SELECT
        "t1"."ancestor_level_name",
        "t1"."ancestor_level_number",
        "t1"."ancestor_node_sort_order",
        "t1"."descendant_node_natural_key",
        LPAD(
          '-',
          (
            "t1"."ancestor_level_number" - CAST(1 AS TINYINT)
          ) * CAST(7 AS TINYINT),
          '-'
        ) || "t1"."ancestor_level_name" AS "product_level_name"
      FROM "products" AS "t1"
    ) AS "t4"
      ON "t2"."product_id" = "t4"."descendant_node_natural_key"
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."ancestor_node_sort_order" ASC