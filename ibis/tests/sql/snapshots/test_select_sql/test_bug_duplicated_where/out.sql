WITH t0 AS (
  SELECT t2.`arrdelay`, t2.`dest`,
         avg(t2.`arrdelay`) OVER (PARTITION BY t2.`dest`) AS `dest_avg`,
         t2.`arrdelay` - avg(t2.`arrdelay`) OVER (PARTITION BY t2.`dest`) AS `dev`
  FROM airlines t2
)
SELECT t1.*
FROM (
  SELECT t0.*
  FROM t0
  WHERE t0.`dev` IS NOT NULL
) t1
ORDER BY t1.`dev` DESC
LIMIT 10