from __future__ import annotations  # noqa: INP001

import os

from algoliasearch.search_client import SearchClient

api_key = os.environ["ALGOLIA_WRITE_API_KEY"]
app_id = os.environ["ALGOLIA_APP_ID"]
index_name = os.environ["ALGOLIA_INDEX"]


# A list of search terms that have (historically) not returned results
# that we can map to existing search terms that we know are good
ONE_WAY_SYNONYMS = {
    "fetchdf": ["to_pandas", "to_polars", "to_pyarrow"],
    "md5": ["hashbytes"],
    "strptime": ["as_timestamp"],
    "unique": ["distinct"],
}


def main():
    client = SearchClient.create(app_id, api_key)
    index = client.init_index(index_name)

    # Core is a custom attribute set to denote whether a record is part
    # of the base expression API, we sort descending so those methods
    # show up first in search instead of backend-specific methods
    override_default_settings = {
        "ranking": [
            "typo",
            "words",
            "desc(core)",
            "filters",
            "proximity",
            "attribute",
            "exact",
        ]
    }

    index.set_settings(override_default_settings)

    for input_, synonyms in ONE_WAY_SYNONYMS.items():
        index.save_synonym(
            {
                "objectID": input_,
                "type": "oneWaySynonym",
                "input": input_,
                "synonyms": synonyms,
            }
        )


if __name__ == "__main__":
    main()
