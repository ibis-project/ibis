SELECT `col`, `analytic`
FROM (
  SELECT `col`, count(1) OVER () AS `analytic`
  FROM (
    SELECT `col`, `filter`
    FROM (
      SELECT *
      FROM (
        SELECT `col`, NULL AS `filter`
        FROM x
      ) t3
      WHERE `filter` IS NULL
    ) t2
  ) t1
) t0