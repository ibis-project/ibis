SELECT t0.`g`, sum(t0.`foo`) AS `foo total`
FROM (
  SELECT t1.*, t1.`a` + t1.`b` AS `foo`
  FROM alltypes t1
  WHERE (t1.`f` > 0) AND
        (t1.`g` = 'bar')
) t0
GROUP BY 1