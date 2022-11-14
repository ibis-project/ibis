SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE (`f` > 0) AND
        (`g` = 'bar')
) t0
GROUP BY 1