def test_sum(alltypes, df):
    expr = alltypes.double_col.sum()
    result = expr.execute()
    assert result == df.double_col.sum()
