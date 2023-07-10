SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      t2.foo_id AS foo_id,
      t2.total AS total,
      t1.value1 AS value1
    FROM (
      SELECT
        t0.foo_id AS foo_id,
        SUM(t0.f) AS total
      FROM star1 AS t0
      GROUP BY
        1
    ) AS t2
    INNER JOIN star2 AS t1
      ON t2.foo_id = t1.foo_id
  ) AS t4
  WHERE
    (
      t4.total > CAST(100 AS TINYINT)
    )
) AS t5
ORDER BY
  t5.total DESC