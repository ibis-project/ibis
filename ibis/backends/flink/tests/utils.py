from __future__ import annotations


def download_jar_for_package(
    package_name: str,
    jar_name: str,
    jar_url: str,
):
    import os
    from importlib import metadata

    import requests

    # Find the path to package lib
    try:
        distribution = metadata.distribution(package_name)
        lib_path = distribution.locate_file("")
    except metadata.PackageNotFoundError:
        lib_path = None

    # Check if the JAR already exists
    jar_path = os.path.join(lib_path, "pyflink/lib", f"{jar_name}.jar")
    if os.path.exists(jar_path):
        return jar_path

    # Download the JAR
    response = requests.get(jar_url, stream=True)
    if response.status_code != 200:
        raise SystemError(
            f"Failed to download the JAR file \n"
            f"\t jar_url= {jar_url} \n"
            f"\t response.status_code= {response.status_code}"
        )

    # Save the JAR
    with open(jar_path, "wb") as jar_file:
        for chunk in response.iter_content(chunk_size=128):
            jar_file.write(chunk)

    return jar_path


# TODO (mehmet): Why does Flink backend not implement `list_catalogs()`?
def get_catalogs(con) -> list[str]:
    show_catalogs = con.raw_sql("show catalogs")
    catalog_list = []
    with show_catalogs.collect() as results:
        for result in results:
            catalog_list.append(result._values[0])

    return catalog_list
