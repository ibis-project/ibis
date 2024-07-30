from __future__ import annotations  # noqa: INP001

import glob
import json
import os
import re
from functools import partial

from algoliasearch.search_client import SearchClient

api_key = os.environ["ALGOLIA_WRITE_API_KEY"]
app_id = os.environ["ALGOLIA_APP_ID"]
index_name = os.environ["ALGOLIA_INDEX"]


# These are QMD files generated with help from Quartodoc.
API_QMDS = [
    "docs/reference/expression-collections.qmd",
    "docs/reference/expression-generic.qmd",
    "docs/reference/expression-geospatial.qmd",
    "docs/reference/expression-numeric.qmd",
    "docs/reference/expression-strings.qmd",
    "docs/reference/expression-tables.qmd",
    "docs/reference/expression-temporal.qmd",
]


HORRID_REGEX = re.compile(r"\|\s*\[(\w+)\]\((#[\w.]+)\)\s*\|\s*(.*?)\s*\|")
# Given | [method](some-anchor) | some multiword description |
# this regex extracts ("method", "some-anchor", "some multiword description")


def _grab_qmd_methods(lines):
    # All of the QMD files have  a section that looks like:
    #
    # ## Methods
    #
    # | [method](anchor-ref) | description |
    # ...
    #
    # ### method
    #
    # yes this is gross, but grab the lines between the `## Methods` and the
    # first `###` and then smash it into a list
    methods = lines[(fence := lines.find("## Methods")) : lines.find("###", fence)]
    methods = [entry for entry in methods.split("\n") if entry.startswith("| [")]

    # Now this in in the form:
    # | [method name](#anchor-name) | Top-level description |
    return methods


def _create_api_record_from_method_line(base_url, method):
    # for e.g. `reference/expression-collections.html` we want to grab "Collections"
    section = (
        base_url.removesuffix(".html")
        .removeprefix("reference/expression-")
        .capitalize()
    )
    name, anchor, desc = re.match(HORRID_REGEX, method).groups()
    record = {
        "objectID": f"{base_url}{anchor}",
        "href": f"{base_url}{anchor}",
        "title": name,
        "backend": "core",
        "core": 1,
        "crumbs": ["Expression API", "API", f"{section} expression"],
    }
    if desc:
        record["text"] = desc

    return record


def adjust_backend_custom_attributes(backend_records):
    """Adjusts attributes of the Algolia records.

    Two custom attribute changes:
        One is the name of the backend, which we can possibly use for grouping
        or filtering results.

        The other is a marker of whether the record is part of the core
        expression API, which we can use to sort results so that generic table
        expressions appear above backend-specific ones in the case of
        name-collisions.

    We also strip out the "text" attribute if it's empty
    """
    backend_name = backend_records[0]["title"].split(".", 1)[0]
    for record in backend_records:
        record["backend"] = backend_name
        record["core"] = 0
        if not record["text"]:
            record.pop("text")

    return backend_records


def main():
    client = SearchClient.create(app_id, api_key)
    index = client.init_index(index_name)

    records = []
    for qmd in API_QMDS:
        # For each QMD file, get the table-section of the methods, anchors, and descriptions
        print(f"Scraping {qmd} for API methods...")  # noqa:T201
        with open(qmd) as f:
            methods = _grab_qmd_methods(f.read())

        # Massage the QMD filename into the expected URL that prepends the anchor
        # so we end up eventually with something like
        # `reference/expression-collections.html#some-anchor`
        base_url = f"{qmd.removeprefix('docs/').removesuffix('.qmd')}.html"

        # Generate a dictionary for each row of the method table
        _creator = partial(_create_api_record_from_method_line, base_url)
        records += list(map(_creator, methods))

    # This saves the list of records to Algolia
    # If the object IDs are new (which typically should be) this adds a new
    # record to the Algolia index.  If the object ID already exists, it gets
    # updated with the new fields in the record dict
    print(f"Uploading {len(records)} records to {index.name=}")  # noqa:T201
    index.save_objects(records)

    # Methods documented on backend-specific docs pages aren't scraped by Quarto
    # since we construct them programmatically.
    # There is a hook in docs/backends/_templates/api.qmd that calls
    # `dump_methods_to_json_for_algolia` that serializes all the backend methods
    # to a backend-specific json file in docs/backends/
    # (Not Pandas and Impala because those backend pages don't use the template)
    #
    # Here, we load those records and upload them to the Algolia index
    records = []
    for record_json in glob.glob("docs/backends/*.json"):
        print(f"Loading {record_json} methods...")  # noqa:T201
        with open(record_json) as f:
            backend_records = json.load(f)
            backend_records = adjust_backend_custom_attributes(backend_records)
            records.extend(backend_records)
    print(f"Uploading {len(records)} records to {index.name=}")  # noqa:T201
    index.save_objects(records)


if __name__ == "__main__":
    main()
