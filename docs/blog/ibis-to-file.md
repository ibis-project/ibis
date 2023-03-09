# Ibis Sneak Peek: Writing to Files

**by Kae Suarez**

Ibis 5.0 is coming soon and will offer new functionality and fixes to users. To enhance clarity around this process, we’re sharing a sneak peek into what we’re working on.

In Ibis 4.0, we added the ability to read CSVs and Parquet via the Ibis interface. We felt this was important because, well, the ability to read files is simply necessary, be it on a local scale, legacy data, data not yet in a database, and so on. However, for a user, the natural next question was “can I go ahead and write when I’m done?” The answer was no. We didn’t like that, especially since we do care about file-based use cases.

So, we’ve gone ahead and fixed that for Ibis 5.0.

## Files in, Files out

Before we can write a file, we need data — so let’s read in a file, to start this off:

```python
t = ibis.read_csv(
    "https://storage.googleapis.com/ibis-examples/data/penguins.csv.gz"
)
```

Of course, we could just write out, but let’s do an operation first — how about using selectors, which you can read more about [here](https://ibis-project.org/blog/selectors/)? Self-promotion aside, here’s an operation:

```python
expr = (
    t.group_by("species")
     .mutate(s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std()))
)
```

Now, finally, time to do the exciting part:

```python
expr.to_parquet("normalized.parquet")
```

Like many things in Ibis, this is as simple and plain-looking as it is important. Being able to create files from Ibis instead of redirecting into other libraries first enables operation at larger scales and fewer steps. Where desired, you can address a backend directly to use its native export functionality — we want to make sure you have the flexibility to use Ibis or the backend as you see fit.

## Wrapping Up

Ibis is an interface tool for analytical engines that can reach scales far beyond a laptop. Files are important to Ibis because:

- Ibis also supports local execution, where files are the standard unit of data — we want to support all our users.
- Files are useful for moving between platforms, and long-term storage that isn’t tied to a particular backend.
- Files can move more easily between our backends than database files, so we think this adds some convenience for the multi-backend use case.

We’re excited to release this functionality in Ibis 5.0.

Interested in Ibis? Docs are available on this very website, at:

- [Ibis Docs](https://ibis-project.org/)

and the repo is always at:

- [Ibis GitHub](https://github.com/ibis-project/ibis)

Please feel free to reach out on GitHub!
