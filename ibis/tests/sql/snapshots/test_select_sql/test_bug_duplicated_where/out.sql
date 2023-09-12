WITH t0 AS (
  SELECT t3.`arrdelay`, t3.`dest`
  FROM airlines t3
),
t1 AS (
  SELECT t0.*, avg(t0.`arrdelay`) OVER (PARTITION BY t0.`dest`) AS `dest_avg`,
         t0.`arrdelay` - avg(t0.`arrdelay`) OVER (PARTITION BY t0.`dest`) AS `dev`
  FROM t0
)
SELECT t2.*
FROM (
  SELECT t1.*
  FROM t1
  WHERE t1.`dev` IS NOT NULL
) t2
ORDER BY t2.`dev` DESC
LIMIT 10