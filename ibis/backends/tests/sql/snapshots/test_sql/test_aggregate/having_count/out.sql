SELECT
  t2.foo_id AS foo_id,
  t2.total AS total
FROM (
  SELECT
    *
  FROM (
    SELECT
      t0.foo_id AS foo_id,
      SUM(t0.f) AS total,
      COUNT(*) AS "CountStar()"
    FROM star1 AS t0
    GROUP BY
      1
  ) AS t1
  WHERE
    (
      t1."CountStar()" > CAST(100 AS TINYINT)
    )
) AS t2