SELECT *, `a` + `b` AS `foo`
FROM alltypes
WHERE (`f` > 0) AND
      (`g` = 'bar')