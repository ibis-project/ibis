WITH t0 AS (
  SELECT t4.`col`, NULL AS `filter`
  FROM `x` t4
),
t1 AS (
  SELECT t0.*
  FROM t0
  WHERE t0.`filter` IS NULL
),
t2 AS (
  SELECT t1.`col`, t1.`filter`
  FROM t1
)
SELECT t3.`col`, t3.`analytic`
FROM (
  SELECT t2.`col`, count(1) OVER () AS `analytic`
  FROM t2
) t3