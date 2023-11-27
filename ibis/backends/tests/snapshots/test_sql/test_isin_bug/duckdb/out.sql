SELECT
  t0.x IN ((
    SELECT
      t1.x AS x
    FROM (
      SELECT
        *
      FROM "t" AS t0
      WHERE
        (
          t0.x > CAST(2 AS TINYINT)
        )
    ) AS t1
  )) AS "InSubquery(x)"
FROM "t" AS t0