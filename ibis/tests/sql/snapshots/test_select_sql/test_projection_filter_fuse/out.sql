SELECT t0.`a`, t0.`b`, t0.`c`
FROM (
  SELECT t1.*
  FROM foo t1
  WHERE t1.`a` > 0
) t0