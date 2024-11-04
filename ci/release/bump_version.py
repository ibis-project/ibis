from __future__ import annotations


def main():
    from dunamai import Version

    version = Version.from_git(latest_tag=True, pattern="default-unprefixed")
    if version.distance:
        version = version.bump(index=0)
        format = "{base}.dev{distance}"
    else:
        format = None

    print(version.serialize(format=format))  # noqa: T201


if __name__ == "__main__":
    main()
