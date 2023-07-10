SELECT
  t3.b AS b,
  t3.sum AS sum
FROM (
  SELECT
    *
  FROM (
    SELECT
      t1.b AS b,
      SUM(t1.a) AS sum,
      MAX(t1.a) AS "Max(a)"
    FROM (
      SELECT
        *
      FROM t AS t0
      WHERE
        (
          t0.b = 'm'
        )
    ) AS t1
    GROUP BY
      1
  ) AS t2
  WHERE
    (
      t2."Max(a)" = CAST(2 AS TINYINT)
    )
) AS t3