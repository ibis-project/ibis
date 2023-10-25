SELECT
  t0.x IN (
    SELECT
      t0.x
    FROM t AS t0
    WHERE
      t0.x > 2
  ) AS "InColumn(x, x)"
FROM t AS t0