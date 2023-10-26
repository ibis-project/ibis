SELECT t0.*
FROM (
  SELECT t1.`arrdelay`, t1.`dest`,
         avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dest_avg`,
         t1.`arrdelay` - avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dev`
  FROM airlines t1
) t0
WHERE t0.`dev` IS NOT NULL
ORDER BY t0.`dev` DESC
LIMIT 10