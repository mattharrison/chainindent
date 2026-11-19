# Chainindent

Chainindent formats pandas and Polars chains in Python scripts, Markdown, and
Jupyter notebooks.

```python
result = (vehicles
    .sort('year', descending=True)
    .with_columns(
        city_mpg=pl.col('city08'),
        highway_mpg=pl.col('highway08'))
    .select('year', 'make', 'city_mpg', 'highway_mpg')
)
```

Each DataFrame step gets its own line. Named columns in `.assign()`,
`.with_columns()`, and `.select()` start on separate lines. A single named
aggregation can stay inline. Positional arguments, list items, and nested
column expression calls share a line when they fit. Long indivisible terms
can exceed the preferred line width.

A standalone `.select()` can stay inline when its arguments are string
literals. Keyword arguments, expressions, and other values trigger the
indented chain layout.

## Run it

From the checkout:

```sh
uv run chainindent example.py
uv run chainindent --diff --line-length 73 chapter.ipynb
uv run chainindent --check notes.md analysis.py
```

Install the checkout as an editable uv tool to run it from any directory.
Edits to the checkout's Python source take effect on the next invocation.

```sh
uv tool install --editable /path/to/chainindent
chainindent example.py
```

uv exposes the command in its tool bin directory. Use `uv tool dir --bin`
to see that directory and `uv tool update-shell` to add it to your PATH.
For a one-off run from a checkout, use
`uvx --from /path/to/chainindent chainindent example.py`.

The default command writes files in place. Pass `--diff` to preview changes
or `--check` to exit with status 1 if files need formatting. Paths can be
files or directories. Pass `-` to read from standard input and write the
formatted result to standard output. `--stdin-filename chapter.md` selects
the format and project settings for that input.

## Configure it

Set defaults in your project's `pyproject.toml` or override them with
`--line-length`, `--indent-width`, `--align-continuations`, and
`--closing-paren-on-own-line`.

```toml
[tool.chainindent]
line-length = 80
indent-width = 4
align-continuations = false
closing-paren-on-own-line = false
```

Use `# chainindent: skip` immediately above a statement to preserve its
formatting. This also works inside a Markdown code block or a notebook cell.

```python
# chainindent: skip
result = vehicles.sort('year').head(5)
```

Chainindent tracks DataFrame, LazyFrame, and Series bindings from recognized
constructors, annotations, assignments, and earlier notebook cells. On a
known receiver it can format a chain containing unfamiliar or custom methods.
For standalone snippets whose receiver cannot be inferred, it looks for
pandas and Polars method patterns. It supports fenced
Python and PyCon blocks, Jupytext percent-format Markdown cells, and plain or
prompted Python cells in notebooks. Python cell magics such as `%%timeit` can
contain formatted Python, with the magic line preserved. Notebook outputs
and metadata are preserved. Incomplete examples are left as written.

Run the tests and docstring examples with `uv run pytest`. Check types with
`uv run ty check`.
