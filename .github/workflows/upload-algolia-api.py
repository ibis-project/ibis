from __future__ import annotations  # noqa: INP001

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
        "text": desc,
        "crumbs": ["Expression API", "API", f"{section} expressions"],
    }

    return record


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


if __name__ == "__main__":
    main()
