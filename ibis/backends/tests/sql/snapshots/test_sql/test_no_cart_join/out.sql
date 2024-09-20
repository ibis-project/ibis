SELECT
  *
FROM (
  SELECT
    "t5"."ancestor_node_sort_order",
    1 AS "n"
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
        CASE
          WHEN (
            (
              "t1"."ancestor_level_number" - 1
            ) * 7
          ) <= LENGTH('-')
          THEN '-'
          ELSE CONCAT(
            REPEAT('-', (
              (
                "t1"."ancestor_level_number" - 1
              ) * 7
            ) - LENGTH('-')),
            '-'
          )
        END || "t1"."ancestor_level_name" AS "product_level_name"
      FROM "products" AS "t1"
    ) AS "t4"
      ON "t2"."product_id" = "t4"."descendant_node_natural_key"
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."ancestor_node_sort_order" ASC