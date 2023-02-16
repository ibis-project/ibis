SELECT t0.*
FROM (
  WITH t1 AS (
    SELECT t3.`arrdelay`, t3.`dest`
    FROM airlines t3
  ),
  t3 AS (
    SELECT t1.*, avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dest_avg`,
           t1.`arrdelay` - avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dev`
    FROM (
      SELECT t3.`arrdelay`, t3.`dest`
      FROM airlines t3
    ) t1
  )
  SELECT t3.*
  FROM (
    SELECT t1.*, avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dest_avg`,
           t1.`arrdelay` - avg(t1.`arrdelay`) OVER (PARTITION BY t1.`dest`) AS `dev`
    FROM (
      SELECT t3.`arrdelay`, t3.`dest`
      FROM airlines t3
    ) t1
  ) t3
  WHERE t3.`dev` IS NOT NULL
) t0
ORDER BY t0.`dev` DESC
LIMIT 10