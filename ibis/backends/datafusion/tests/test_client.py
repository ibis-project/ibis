def test_list_tables(client):
    assert set(client.list_tables()) == {
        'awards_players',
        'batting',
        'functional_alltypes',
    }
