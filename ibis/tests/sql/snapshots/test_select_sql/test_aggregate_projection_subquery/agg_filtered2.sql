SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE `f` > 0
) t0
WHERE `foo` < 10
GROUP BY 1