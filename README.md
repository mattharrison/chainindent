# Chainindent

Chainindent formats pandas and Polars chains in Python, Markdown, and Jupyter
notebooks. In a chain, each DataFrame operation starts on its own line.

**Before**

```python
result = vehicles.sort('year').with_columns(city_mpg=pl.col('city08'), highway_mpg=pl.col('highway08')).select('year', 'city_mpg', 'highway_mpg')
```

**After**

```python
result = (vehicles
    .sort('year')
    .with_columns(
        city_mpg=pl.col('city08'),
        highway_mpg=pl.col('highway08'))
    .select('year', 'city_mpg', 'highway_mpg')
)
```

## Quickstart

Chainindent requires Python 3.14 or newer and has no runtime dependencies.
The commands below use uv.

To preview the changes to a local Python file, run Chainindent from GitHub
with `--diff`. The default command writes files in place.

```sh
uvx --from git+https://github.com/mattharrison/chainindent chainindent --diff analysis.py
```

### Install a local checkout

To run `chainindent` from any directory and pick up edits to your local
checkout, install it in editable mode.

```sh
git clone https://github.com/mattharrison/chainindent.git
cd chainindent
uv tool install --editable .
chainindent --help
```

If your shell cannot find the command, run `uv tool update-shell` and open a
new terminal. From the checkout, you can preview changes with
`uv run chainindent --diff analysis.py`.

## What gets formatted

A standalone `.select` call with string literals stays inline. Expressions
and keyword arguments put the call in the chain layout. The `.assign` method
puts each named column on a separate line.

```python
vehicles.select('year', 'city08')

(vehicles
    .select(pl.col('city08').mean())
)

(df
    .assign(
        revenue=lambda df: df['price'] * df['quantity'],
        margin=lambda df: df['revenue'] - df['cost'])
)
```

Positional arguments, list items, and nested expression calls can share a
line when they fit. A single named aggregation can stay inline. An
indivisible token such as a long string can exceed the preferred width.

| File | Where Chainindent formats |
| --- | --- |
| Python (`.py`) | Python source and Python examples in Jupytext percent-format Markdown cells |
| Markdown (`.md`, `.markdown`) | Fenced Python and PyCon examples |
| Jupyter (`.ipynb`) | Python code cells, including PyCon input and supported cell magics such as `%%timeit`, plus Python and PyCon fences in Markdown cells |

Chainindent uses assignments, type annotations, and earlier notebook cells
to recognize DataFrames, LazyFrames, and Series. It also recognizes method
patterns in standalone examples. It preserves notebook outputs and metadata,
console output, and incomplete examples.

## Configure line width and indentation

The defaults are 80 columns and four spaces. For a book project that uses
73 columns, add this to its `pyproject.toml`.

```toml
[tool.chainindent]
line-length = 73
```

Chainindent uses the nearest `[tool.chainindent]` table above each input.
A command-line flag overrides the project setting for that run.

```sh
chainindent --diff --line-length 80 chapter.ipynb
```

The other settings are `indent-width`, `align-continuations`, and
`closing-paren-on-own-line`. The last two default to `false` and have
matching command-line flags.

## Skip a teaching example

Put `# chainindent: skip` immediately above a statement to preserve its
layout. The directive also works inside a notebook cell or a fenced example.

```python
# chainindent: skip
result = vehicles.sort('year').head(5)
```

## Use Chainindent with Ruff or Black

When you want Chainindent's layout in the final file, run it after your
general-purpose formatter.

```sh
uvx ruff format analysis.py
chainindent analysis.py
```

## Check formatting in CI

Chainindent accepts files and recursively scans directories for supported
files. Use `--check` to test whether anything needs formatting.

```sh
chainindent --check chapters/
```

| Exit status | Meaning |
| --- | --- |
| `0` | The check passed |
| `1` | At least one file would change |
| `2` | The command encountered an error |

For standard input, pass `-`. Use `--stdin-filename` to select the file format
and locate the project configuration.

## Development

```sh
uv run pytest
uv run ty check
```

Pytest runs the regression suite and the docstring examples.
