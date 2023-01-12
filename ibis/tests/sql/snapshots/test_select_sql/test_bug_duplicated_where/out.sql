SELECT t0.*
FROM (
  SELECT t1.*
  FROM (
    SELECT t2.*, avg(t2.`arrdelay`) OVER (PARTITION BY t2.`dest`) AS `dest_avg`,
           t2.`arrdelay` - avg(t2.`arrdelay`) OVER (PARTITION BY t2.`dest`) AS `dev`
    FROM (
      SELECT t3.`arrdelay`, t3.`dest`
      FROM airlines t3
    ) t2
  ) t1
  WHERE t1.`dev` IS NOT NULL
) t0
ORDER BY t0.`dev` DESC
LIMIT 10