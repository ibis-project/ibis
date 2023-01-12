SELECT t0.`col`, t0.`analytic`
FROM (
  SELECT t1.`col`, count(1) OVER () AS `analytic`
  FROM (
    SELECT t2.`col`, t2.`filter`
    FROM (
      SELECT t3.*
      FROM (
        SELECT t4.`col`, NULL AS `filter`
        FROM x t4
      ) t3
      WHERE t3.`filter` IS NULL
    ) t2
  ) t1
) t0