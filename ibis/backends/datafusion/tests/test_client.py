def test_list_tables(client):
    tables = set(client.list_tables())
    assert {'awards_players', 'batting', 'functional_alltypes'} <= tables


def test_register_csv(client, data_directory):
    path = data_directory / 'functional_alltypes.csv'
    client.register_csv("ft_csv", path)
    assert "ft_csv" in client.list_tables()


def test_register_parquet(client, data_directory):
    path = data_directory / 'functional_alltypes.parquet'
    client.register_parquet("ft_parquet", path)
    assert "ft_parquet" in client.list_tables()
