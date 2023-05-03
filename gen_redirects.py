import pathlib

import mkdocs_gen_files

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <link rel="canonical" href="{url}">
    <meta name="robots" content="noindex">
    <script>var anchor=window.location.hash.substr(1);location.href="{url}"+(anchor?"#"+anchor:"")</script>
    <meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
Redirecting...
</body>
</html>
"""

# Versions for templated redirects
VERSIONS = ["latest", "dev", "4.1.0", "4.0.0", "3.2.0", "3.1.0"]

# Templated redirects
TEMPLATED_REDIRECTS = {
    "/docs/{version}/": "/",
    "/docs/{version}/install/": "/install/",
    "/docs/{version}/docs/": "/docs/",
    "/docs/{version}/backends/": "/backends/",
}

# Untemplated redirects
REDIRECTS = {
    "/backends/Pandas/": "/backends/pandas/",
}

# Fill in templates
REDIRECTS.update(
    {
        old.format(version=version): new
        for version in VERSIONS
        for old, new in TEMPLATED_REDIRECTS.items()
    }
)

# Write all redirect files
for old, new in REDIRECTS.items():
    if old.endswith("/"):
        old = old + "index.html"

    html = HTML_TEMPLATE.format(url=new)

    with mkdocs_gen_files.open(pathlib.Path(old.lstrip("/")), "w") as f:
        f.write(html)
