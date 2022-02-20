# Working on the Codebase

## Find an issue to work on

All contributions are welcome! Code, docs, and constructive feedback are all
great contributions to the project.

If you don't have a particular issue in mind head over to the GitHub issue
tracker for Ibis and look for open issues with the label [`good first issue`](https://github.com/ibis-project/ibis/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

Feel free to help with other issues that aren't labeled as such, but they may be more challenging.

Once you find an issue you want to work on, write a comment with the text
`/take` on the issue. GitHub will then assign the issue to you.

This lets people know you're working on the issue. If you find an issue that
has an assignee, comment on the issue and ask whether the assignee is still
working on the issue.

## Make a branch

The first thing you want to do is make a branch. Let's call it `useful-bugfix`:

```sh
git checkout -b useful-bugfix
```

## Make the desired change

Let's say you've made a change to `ibis/expr/types.py` to fix a bug reported in issue #424242 (not actually an issue).

Running `git status` should give output similar to this:

```sh
On branch useful-bugfix
Your branch is up to date with 'origin/useful-bugfix'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   ibis/expr/types.py

no changes added to commit (use "git add" and/or "git commit -a")
```

## Run the test suite

Next, you'll want to run a subset of the test suite.

### Required Dependencies

!!! warning "You need a development environment before running tests"

    Make sure you've set up a [development environment](01_environment.md)
    before proceeding

Run the test suite:

```sh
pytest -m core
```

!!! tip "Each backend has a `pytest` marker"

    You can run the tests for a specific backend using

    ```sh
    pytest -m $the_backend_name
    ```

    For example, to run SQLite tests:

    ```sh
    pytest -m sqlite
    ```

## Commit your changes

### Required Dependencies

- `git`
- [`cz`](https://commitizen-tools.github.io/commitizen/)

!!! tip

    `cz` is already installed in your environment if you followed the [setup
    instructions](01_environment.md)

Next, you'll want to commit your changes.

Ibis's commit message structure follows the [`semantic-release`
conventions](https://github.com/semantic-release/semantic-release).

!!! warning

    It isn't necessary to use `cz commit` to make commits, but it is necessary
    to follow the instructions outlined in [this
    table](https://github.com/semantic-release/semantic-release#commit-message-format).

Stage your changes and run `cz commit`:

```sh
git add .
cz commit
```

You should see a series of prompts about actions to take next:

1. Select the type of change you're committing. In this case, we're committing a bug fix, so we'll select fix:

   ```console
   ? Select the type of change you are committing (Use arrow keys)
    » fix: A bug fix. Correlates with PATCH in SemVer
      feat: A new feature. Correlates with MINOR in SemVer
      docs: Documentation only changes
      style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
      refactor: A code change that neither fixes a bug nor adds a feature
      perf: A code change that improves performance
      test: Adding missing or correcting existing tests
      build: Changes that affect the build system or external dependencies (example scopes: pip, docker, npm)
      ci: Changes to our CI configuration files and scripts (example scopes: GitLabCI)
   ```

   Generally you don't need to think too hard about what category to select, but note that:

   - `feat` will cause a minor version bump
   - `fix` will cause a patch version bump
   - everything else will **not** cause a version bump, **unless it's a breaking
     change** (continue reading these instructions for more info on that)

2. Next, you're asked what the scope of this change is:

   ```console
   ? What is the scope of this change? (class or file name): (press [enter] to skip)
   ```

   This is optional, but if there's a clear component or single file that is
   modified you should put it. In our case, let's assume the bug fixed a type
   inference problem, so we'd type in `type-inference` at this prompt.

3. You'll then be asked to type in a short description of the change which will be the commit message title:

   ```console
   ? Write a short and imperative summary of the code changes: (lower case and no period)
    fix a type inference issue where floats were incorrectly cast to ints
   ```

   Let's say there was a problem with spurious casting of float to integers, so
   we type in the message above. That number on the left (here `(69)`) is the
   length of description you've typed in.

4. Next you'll be asked for a longer description, which is entirely optional
   **unless the change is a breaking change**, or you feel like a bit of prose

   ```console
   ? Provide additional contextual information about the code changes: (press [enter] to skip)
    A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.
   ```

   For non breaking changes, this isn't strictly necessary but it can be very
   helpful when a change is large, obscure, or complex. For this example let's just reiterate
   most of what the commit title says.

5. Next you're asked about breaking changes:

   ```console
   ? Is this a BREAKING CHANGE? Correlates with MAJOR in SemVer (y/N)
   ```

   If you answer `y`, then you'll get an additional prompt asking you to
   describe the breaking changes. This description will ultimately make its way
   into the user-facing release notes. If there aren't any breaking changes, press enter.
   Let's say this bug fix does **not** introduce a breaking change.

6. Finally, you're asked whether this change affects any open issues (ignore
   the bit about breaking changes) and if yes then to reference them:

   ```console
   ? Footer. Information about Breaking Changes and reference issues that this commit closes: (press [enter] to skip)
    fixes #424242
   ```

   Here we typed `fixes #424242` to indicate that we fixed issue #9000.

Whew! Seems like a lot, but it's rather quick once you get used to it. After
that you should have a commit that looks roughly like this, ready to be automatically rolled into the next release:

```console
commit 4049adbd66b0df48e37ca105da0b9139101a1318 (HEAD -> useful-bugfix)
Author: Phillip Cloud <417981+cpcloud@users.noreply.github.com>
Date:   Tue Dec 21 10:30:50 2021 -0500

    fix(type-inference): fix a type inference issue where floats were incorrectly cast to ints

    A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.

    fixes #424242
```

### Push your changes

Now that you've got a commit, you're ready to push your changes and make a pull request!

```sh
gh pr create
```

Follow the prompts, and `gh` will print a link to your PR upon successfuly submission.
