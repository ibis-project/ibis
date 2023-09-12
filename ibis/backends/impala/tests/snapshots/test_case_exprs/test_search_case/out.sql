CASE
  WHEN `f` > 0 THEN `d` * 2
  WHEN `c` < 0 THEN `a` * 2
  ELSE CAST(NULL AS bigint)
END