SELECT t0.`g`, sum(t0.`foo`) AS `foo total`
FROM (
  SELECT t1.*, t1.`a` + t1.`b` AS `foo`
  FROM alltypes t1
  WHERE t1.`f` > 0
) t0
WHERE t0.`foo` < 10
GROUP BY 1