SELECT
  *
FROM (
  SELECT
    t4.ancestor_node_sort_order AS ancestor_node_sort_order,
    CAST(1 AS TINYINT) AS n
  FROM (
    SELECT
      t0.product_id AS product_id,
      t2.ancestor_level_name AS ancestor_level_name,
      t2.ancestor_level_number AS ancestor_level_number,
      t2.ancestor_node_sort_order AS ancestor_node_sort_order,
      t2.descendant_node_natural_key AS descendant_node_natural_key,
      t2.product_level_name AS product_level_name
    FROM facts AS t0
    INNER JOIN (
      SELECT
        t1.ancestor_level_name AS ancestor_level_name,
        t1.ancestor_level_number AS ancestor_level_number,
        t1.ancestor_node_sort_order AS ancestor_node_sort_order,
        t1.descendant_node_natural_key AS descendant_node_natural_key,
        CONCAT(
          LPAD('-', (
            t1.ancestor_level_number - CAST(1 AS TINYINT)
          ) * CAST(7 AS TINYINT), '-'),
          t1.ancestor_level_name
        ) AS product_level_name
      FROM products AS t1
    ) AS t2
      ON t0.product_id = t2.descendant_node_natural_key
  ) AS t4
  GROUP BY
    1
) AS t5
ORDER BY
  t5.ancestor_node_sort_order ASC