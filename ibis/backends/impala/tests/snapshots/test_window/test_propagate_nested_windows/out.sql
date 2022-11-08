SELECT lag(`f` - lag(`f`) OVER (PARTITION BY `g` ORDER BY `f` ASC)) OVER (PARTITION BY `g` ORDER BY `f` ASC) AS `foo`
FROM alltypes