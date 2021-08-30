import argparse
import re
import sys
from pathlib import Path

import black
import tomli
from poetry.core.factory import Factory
from poetry.core.masonry.builders.sdist import SdistBuilder

# Poetry inserts a double pipe for "OR" version constraints.
# We use this regular expression to turn those into a single pipe.
DOUBLE_PIPE_REGEX = re.compile(r"\s+\|\|\s+")


def main(args: argparse.Namespace) -> None:
    input_dir = args.input_directory
    # create poetry things
    poetry = Factory().create_poetry(input_dir)
    sdist_builder = SdistBuilder(poetry)

    # generate setup.py code
    code = sdist_builder.build_setup().decode("UTF-8")

    # pull out black config
    config = tomli.loads(input_dir.joinpath("pyproject.toml").read_text())
    black_config = config["tool"]["black"]
    black_config["string_normalization"] = black_config.pop(
        "skip_string_normalization", False
    )
    black_config.pop("exclude", None)
    out = black.format_file_contents(
        code, fast=False, mode=black.Mode(**black_config)
    )
    print(DOUBLE_PIPE_REGEX.sub("|", out), file=args.output_file, end="")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate a setup.py file from pyproject.toml"
    )
    p.add_argument(
        "-i",
        "--input-directory",
        type=Path,
        default=Path(__file__).parent.parent.resolve(),
        help="The input directory to use for poetry setup",
    )
    p.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType(mode="w"),
        default=sys.stdout,
        help="The file to which to write the generated setup.py output",
    )
    main(p.parse_args())
