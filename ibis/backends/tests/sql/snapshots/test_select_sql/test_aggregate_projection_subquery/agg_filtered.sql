SELECT
  t3.g AS g,
  SUM(t3.foo) AS "foo total"
FROM (
  SELECT
    *
  FROM (
    SELECT
      t1.a AS a,
      t1.b AS b,
      t1.c AS c,
      t1.d AS d,
      t1.e AS e,
      t1.f AS f,
      t1.g AS g,
      t1.h AS h,
      t1.i AS i,
      t1.j AS j,
      t1.k AS k,
      (
        t1.a + t1.b
      ) AS foo
    FROM (
      SELECT
        *
      FROM alltypes AS t0
      WHERE
        (
          t0.f > CAST(0 AS TINYINT)
        )
    ) AS t1
  ) AS t2
  WHERE
    (
      t2.g = 'bar'
    )
) AS t3
GROUP BY
  1