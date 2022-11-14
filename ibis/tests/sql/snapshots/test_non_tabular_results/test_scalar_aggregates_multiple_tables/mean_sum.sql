SELECT t0.`fv` - t1.`uv` AS `tmp`
FROM (
  SELECT avg(`value`) / sum(`value`) AS `fv`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) / sum(`value`) AS `uv`
    FROM tbl
    WHERE `flag` = '0'
  ) t1