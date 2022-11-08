SELECT lag(`f` - lag(`f`) OVER (ORDER BY `f` ASC)) OVER (ORDER BY `f` ASC) AS `foo`
FROM alltypes