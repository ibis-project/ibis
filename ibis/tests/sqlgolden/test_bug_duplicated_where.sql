SELECT *
FROM (
  SELECT t1.*
  FROM (
    SELECT *, avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dest_avg`,
           `arrdelay` - avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dev`
    FROM (
      SELECT `arrdelay`, `dest`
      FROM airlines
    ) t3
  ) t1
  WHERE t1.`dev` IS NOT NULL
) t0
ORDER BY `dev` DESC
LIMIT 10
