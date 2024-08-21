SELECT
  NTILE(2) OVER (ORDER BY RAND() ASC NULLS LAST) - 1 AS `new_col`
FROM (
  SELECT
    `test`
  FROM (VALUES
    (1),
    (2),
    (3),
    (4),
    (5)) AS `test`(`test`)
) AS `t0`
LIMIT 10