# Ibis Examples

## Regeneration
In `ibis/examples`, run `pixi shell` to drop into a shell with the
required dependencies to run both `gen_registry.py` and `gen_examples.R`

After running `pixi shell`, move up to the root of the repo before
running

`python ibis/examples/gen_registry.py`

To test out new functions in `gen_registry.py`, you can pass the `-d` flag to
prevent the script from uploading results to the examples bucket.


## Adding new examples

Add a function in `gen_registry.py`.  This function should:
- download the raw data, process it, and place it in `ibis/examples/data/`
- add an entry for each processed file to the `metadata` dictionary. Even if
  this entry is empty, the key much match the name of the dataset (not the full
  filename, e.g. "nycflights13_airlines" not "nycflights13_airlines.parquet")
- add a descriptor file to `ibis/examples/descriptions` where the filename is
  the name of the example dataset and the contents are a short description
