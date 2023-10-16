SELECT t0.*
FROM `t0` t0
WHERE (t0.`a` < 100) AND
      (t0.`a` = (
  SELECT max(t0.`a`) AS `Max(a)`
  FROM `t0` t0
  WHERE t0.`a` < 100
)) AND
      (t0.`b` = 'a')