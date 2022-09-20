SELECT (t0.`mean` / t1.`mean`) - 1 AS `Subtract(Divide(mean, mean), 1)`
FROM (
  SELECT avg(`value`) AS `mean`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) AS `mean`
    FROM tbl
    WHERE `flag` = '0'
  ) t1
