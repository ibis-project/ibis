#!/usr/bin/env python3

import collections
import fnmatch
import json
import pathlib
import subprocess
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


def generate_dependency_graph(*args):
    command = ("pydeps", "--show-deps", *args)
    print(f"Running: {' '.join(command)}")  # noqa: T201
    result = subprocess.check_output(command, text=True)
    return json.loads(result)


def check_dependency_rules(dependency_graph, allowed_imports, disallowed_imports):
    prohibited_deps = collections.defaultdict(set)

    for module, module_data in dependency_graph.items():
        imports = module_data.get("imports", [])

        for pattern, disallow_rules in disallowed_imports.items():
            if fnmatch.fnmatch(module, pattern):
                for disallow_rule in disallow_rules:
                    for imported in imports:
                        if fnmatch.fnmatch(imported, disallow_rule):
                            if imported not in allowed_imports.get(module, []):
                                prohibited_deps[module].add(imported)

    return prohibited_deps


disallowed_imports = {
    "ibis.expr.*": ["numpy", "pandas", "pyarrow"],
}

allowed_imports = {
    "ibis.expr.*.test_*_pandas.py": ["numpy", "pandas"],
    "ibis.expr.*.test_*_pyarrow.py": ["pyarrow"],
}


if __name__ == '__main__':
    dependency_graph = generate_dependency_graph(*sys.argv[1:])
    prohibited_deps = check_dependency_rules(
        dependency_graph, allowed_imports, disallowed_imports
    )

    if prohibited_deps:
        print("\n")  # noqa: T201
        print("Prohibited dependencies:")  # noqa: T201
        print("------------------------")  # noqa: T201
        for module, deps in prohibited_deps.items():
            print(f"\n{module}:")  # noqa: T201
            for dep in deps:
                print(f"  <= {dep}")  # noqa: T201
        sys.exit(1)
    else:
        print("Good! No prohibited dependencies found.")  # noqa: T201
        sys.exit(0)
