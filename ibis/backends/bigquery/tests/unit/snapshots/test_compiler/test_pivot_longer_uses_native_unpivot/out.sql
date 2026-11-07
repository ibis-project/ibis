SELECT
  `artist`,
  `week`,
  `rank`
FROM (
  SELECT
    `t0`.*
    EXCEPT (`wk1`, `wk2`),
    CAST(`t0`.`wk1` AS INT64) AS `wk1`,
    CAST(`t0`.`wk2` AS INT64) AS `wk2`
  FROM `t` AS `t0`
)
UNPIVOT INCLUDE NULLS (`rank` FOR `week` IN (`wk1` AS 'wk1', `wk2` AS 'wk2'))