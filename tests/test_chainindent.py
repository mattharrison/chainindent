import ast
import json
from textwrap import dedent

import pytest

from chainindent.cli import main
from chainindent.documents import format_document
from chainindent.formatter import Settings, format_python


@pytest.fixture
def assert_format():
    def check(before, after, **settings):
        style = Settings(**settings)
        expected = dedent(after).lstrip("\n")
        formatted = format_python(dedent(before).lstrip("\n"), style)
        assert formatted == expected
        assert format_python(formatted, style) == formatted

    return check


def test_single_assignment_and_mixed_named_select(assert_format):
    assert_format(
        """\
        result = df.with_columns(total=pl.col('price') * pl.col('quantity'))
        report = df.select('customer', amount=pl.col('total')).sort('amount')
        """,
        """\
        result = (df
            .with_columns(
                total=pl.col('price') * pl.col('quantity'))
        )
        report = (df
            .select('customer',
                amount=pl.col('total'))
            .sort('amount')
        )
        """,
    )

def test_only_data_frame_chains_and_skip_markers(assert_format):
    assert_format(
        """\
        def example():
            # chainindent: skip
            first = autos.sort('year').head(5)
            second = autos.sort('year').head(5)
            third = autos.sort('year')
            path = Path('data').resolve().as_posix()
            x = 'word'.upper().strip()
        """,
        """\
        def example():
            # chainindent: skip
            first = autos.sort('year').head(5)
            second = (autos
                .sort('year')
                .head(5)
            )
            third = autos.sort('year')
            path = Path('data').resolve().as_posix()
            x = 'word'.upper().strip()
        """,
    )

def test_select_with_a_positional_polars_expression_chain(assert_format):
    assert_format(
        """\
        print(autos.select(pl.col('city08', 'highway08')
            .name.replace(pattern='08$', value='_mpg')))
        print(autos.select('city08', 'highway08'))
        """,
        """\
        print(autos
            .select(pl.col('city08', 'highway08')
                .name.replace(pattern='08$', value='_mpg'))
        )
        print(autos.select('city08', 'highway08'))
        """,
    )


@pytest.mark.parametrize("method", ["select", "select_seq"])
def test_standalone_select_layout_depends_on_arguments(assert_format, method):
    assert_format(
        f"""\
        df.{method}()
        df.{method}('year', 'make')
        df.{method}(pl.col('year'))
        df.{method}(columns)
        df.{method}(['year', 'make'])
        df.{method}('year', make_name=pl.col('make'))
        """,
        f"""\
        df.{method}()
        df.{method}('year', 'make')
        (df
            .{method}(pl.col('year'))
        )
        (df
            .{method}(columns)
        )
        (df
            .{method}(['year', 'make'])
        )
        (df
            .{method}('year',
                make_name=pl.col('make'))
        )
        """,
    )


def test_polars_glimpse_formats_inside_print(assert_format):
    assert_format(
        """\
        print(
            df.select('make', 'model', 'year')
            .glimpse(max_items_per_column=2, return_type='frame')
        )
        """,
        """\
        print(df
            .select('make', 'model', 'year')
            .glimpse(max_items_per_column=2, return_type='frame')
        )
        """,
    )


@pytest.mark.parametrize(
    "source, method",
    [
        ("df.sort('key').merge_sorted(other, key='key')", "merge_sorted"),
        ("df.select('key').rechunk()", "rechunk"),
        ("df.select('key').write_ipc('output.arrow')", "write_ipc"),
        ("df.select(pl.col('key').max()).item()", "item"),
        ("df.select(pl.col('key')).custom_method()", "custom_method"),
        ("df.update(other, on='key').update(third, on='key')", "update"),
    ],
)
def test_polars_chain_signals(source, method):
    formatted = format_python(source + "\n", Settings())
    assert f"\n    .{method}(" in formatted
    assert format_python(formatted, Settings()) == formatted


def test_ambiguous_non_frame_chains_remain_unchanged(assert_format):
    source = """\
        soup.select('button').item(0)
        mapping.update({'first': 1}).update({'second': 2})
        """
    assert_format(source, source)


def test_nested_dataframe_steps_and_expression_steps(assert_format):
    assert_format(
        """\
        print(autos.join(autos.filter(ford).select('year').unique(), how='cross')
            .with_columns(result=pl.col('mpg').abs().sqrt()).sort('year'))
        """,
        """\
        print(autos
            .join(autos
                .filter(ford)
                .select('year')
                .unique(),
                how='cross')
            .with_columns(
                result=pl.col('mpg').abs().sqrt())
            .sort('year')
        )
        """,
    )

def test_long_list_and_sort_arguments_pack_to_width(assert_format):
    assert_format(
        """\
        df.sort(['year', 'make', 'model'], descending=[True, False, False]).select(['year', 'make', 'model', 'displ', 'cylinders', 'drive', 'VClass', 'fuelType', 'highway08'])
        """,
        """\
        (df
            .sort(['year', 'make', 'model'],
                descending=[True, False, False])
            .select(['year', 'make', 'model', 'displ', 'cylinders',
                'drive', 'VClass', 'fuelType', 'highway08'])
        )
        """,
        line_length=60,
    )

def test_comment_relocation_preserves_order(assert_format):
    assert_format(
        """\
        (autos
         .with_columns(
          pl.col('year').cast(pl.Int16),
          pl.col('createdOn').str.to_date('%b %d %Y'),  # Convert the ISO date to a value Polars can work with
         )  # Method comment
         .select(cols)
        )
        """,
        """\
        (autos
            .with_columns(pl.col('year').cast(pl.Int16),
                # Convert the ISO date to a value Polars can work with
                pl.col('createdOn').str.to_date('%b %d %Y')
            )  # Method comment
            .select(cols)
        )
        """,
        line_length=65,
    )

def test_indentation_and_method_closing_settings(assert_format):
    assert_format(
        "answer = df.assign(a=1, b=2).head(5)\n",
        """\
        answer = (df
          .assign(
            a=1,
            b=2
          )
          .head(5)
        )
        """,
        indent_width=2,
        closing_paren_on_own_line=True,
    )

def test_single_and_multiple_named_aggregations(assert_format):
    assert_format(
        """\
        df.group_by('year').agg(avg=pl.col('mpg').mean())
        df.group_by('year').agg(avg=pl.col('mpg').mean(), num=pl.len())
        """,
        """\
        (df
            .group_by('year')
            .agg(avg=pl.col('mpg').mean())
        )
        (df
            .group_by('year')
            .agg(
                avg=pl.col('mpg').mean(),
                num=pl.len())
        )
        """,
    )


@pytest.fixture
def notebook():
    return {
        "cells": [
            {"cell_type": "code", "source": [">>> autos.sort('year').head(5)"],
             "metadata": {"tags": ["example"]}, "outputs": [{"text": ["example"]}]},
            {"cell_type": "code", "source": "autos.sort("},
            {"cell_type": "markdown",
             "source": "```pycon\n>>> autos.sort('year').head(5)\n5\n```"},
        ],
        "metadata": {"language_info": {"name": "python"}},
    }


def test_notebook_edits_only_sources_and_leaves_incomplete_code(notebook):
    text = json.dumps(notebook, indent=1)
    updated = format_document(text, ".ipynb", Settings())
    parsed = json.loads(updated)
    prompted_chain = ">>> (autos\n...     .sort('year')\n...     .head(5)\n... )"
    assert "".join(parsed["cells"][0]["source"]) == prompted_chain
    assert parsed["cells"][2]["source"] == f"```pycon\n{prompted_chain}\n5\n```"
    assert '"outputs": [\n    {\n     "text": [\n      "example"' in updated
    for actual, original in zip(parsed["cells"], notebook["cells"], strict=True):
        actual.pop("source")
        original.pop("source")
        assert actual == original
    assert json.loads(updated)["cells"][1]["source"] == "autos.sort("
    assert format_document(updated, ".ipynb", Settings()) == updated

def test_jupytext_markdown_fences_and_python_cells():
    source = (
        "# %% [markdown]\n# Prose.\n#\n# ```{.python}\n"
        "# autos.sort('year').head(5)\n# ```\n\n# %%\nautos.sort('year').head(5)\n"
    )
    formatted = format_document(source, ".py", Settings())
    assert "# (autos\n#     .sort('year')\n#     .head(5)\n# )" in formatted
    assert formatted.endswith("\n(autos\n    .sort('year')\n    .head(5)\n)\n")
    assert format_document(formatted, ".py", Settings()) == formatted


@pytest.mark.parametrize("align, padding", [(False, 8), (True, 13)])
def test_list_continuation_alignment(assert_format, align, padding):
    original = "df.select(['a_segment', 'b_segment', 'c_segment']).head(5)\n"
    expected = (
        "(df\n"
        "    .select(['a_segment', 'b_segment',\n"
        + " " * padding + "'c_segment'])\n"
        "    .head(5)\n"
        ")\n"
    )
    assert_format(original, expected, line_length=40, align_continuations=align)


def test_pycon_incomplete_example_with_a_complete_neighbor():
    source = (
        "```pycon\n"
        ">>> autos.sort(\n"
        ">>> autos.sort('year').head(5)\n"
        "<table output>\n"
        "```\n"
    )
    formatted = format_document(source, ".md", Settings())
    assert formatted == (
        "```pycon\n"
        ">>> autos.sort(\n"
        ">>> (autos\n"
        "...     .sort('year')\n"
        "...     .head(5)\n"
        "... )\n"
        "<table output>\n"
        "```\n"
    )


@pytest.fixture
def configured_python_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.chainindent]\nindent-width = 2\n")
    path = tmp_path / "analysis.py"
    path.write_text("autos.sort('year').head(5)\n")
    return path


def test_cli_preview_check_and_write(configured_python_file, capsys):
    path = configured_python_file
    original = path.read_text()
    assert main(["--diff", str(path)]) == 0
    assert "+  .sort('year')" in capsys.readouterr().out
    assert path.read_text() == original
    assert main(["--check", str(path)]) == 1
    assert path.read_text() == original
    assert main([str(path)]) == 0
    assert path.read_text() == "(autos\n  .sort('year')\n  .head(5)\n)\n"
    assert main(["--check", str(path)]) == 0


@pytest.fixture
def check_width():
    def check(source, width):
        settings = Settings(line_length=width)
        before = dedent(source)
        after = format_python(before, settings)
        assert ast.dump(ast.parse(after)) == ast.dump(ast.parse(before))
        assert max(map(len, after.splitlines())) <= width
        assert format_python(after, settings) == after
        return after

    return check


def test_polars_expression_with_grouped_receiver(check_width):
    result = check_width(
        """\
        result = (df
            .with_columns(
                timestamp=(pl.col('date') + ' ' + pl.col('offset')
                    .str.replace('MST', '-0700').str.replace('MDT', '-0600'))
                    .str.to_datetime('%Y-%m-%d %H:%M %z', time_zone='UTC')
                    .dt.convert_time_zone('America/Denver'))
        )
        """,
        73,
    )
    assert "timestamp=(pl.col('date')" in result
    assert "\n            .str.to_datetime(" in result


def test_polars_expression_referenced_by_name(check_width):
    result = check_width(
        r"""
        mentions = (df
            .select(
                count=tweet_col.str.extract_all(r'\S+')
                    .list.eval(pl.element().str.starts_with('@'))
                    .list.sum())
        )
        """,
        73,
    )
    assert "\n            .list.eval(" in result
    assert ".list.sum())" in result


def test_punctuation_counts_toward_line_width(check_width):
    result = check_width(
        r"""
        (df
            .with_columns(
                num_gears=pl.col('trany').str.extract(r'(\d+)')
                    .cast(pl.UInt8))
        )
        """,
        70,
    )
    assert "\n            .cast(pl.UInt8))" in result


def test_assignment_prefix_and_dictionary_entries_count_toward_width(check_width):
    source = """\
        new_vehicle = pl.DataFrame({'model': ['F-150'], 'year': [1982], 'displ': [5.0], 'fuel': ['Gas']}).select('model').head(5)
        """
    assert "'year': [1982]," in check_width(source, 73)


def test_pandas_loc_can_wrap_column_list(check_width):
    result = check_width(
        """\
        presidents.sort_values('Year').loc[:, ['Party', 'Domestic_accomplishments', 'Foreign_policy_accomplishments']]
        """,
        80,
    )
    assert ".loc[:," in result
    assert "'Foreign_policy_accomplishments']]" in result


def test_moving_select_argument_comment_also_indents_nested_chain(assert_format):
    assert_format(
        """\
        draw(ashby.select(cs.numeric().median()),  # alternative: calculate it from the original group
             width=60)
        """,
        """\
        draw(
            # alternative: calculate it from the original group
            ashby
                .select(cs.numeric().median()),
            width=60)
        """,
        line_length=73,
    )


@pytest.mark.parametrize("index", ["name,", "(name,)", ":,", "row, (first, second)"])
def test_pandas_subscript_preserves_tuple_shape(check_width, index):
    check_width(f"df.sort_values('year').loc[{index}].head(5)\n", 40)


@pytest.mark.parametrize("prompt", ["", ">>> "])
def test_notebook_python_magic_formats_body_and_preserves_shell_magic(notebook, prompt):
    source = "df.sort_values('year').head(5)"
    notebook["cells"] = [
        {"cell_type": "code", "source": f"{prompt}%%timeit -n 1\n{prompt}{source}"},
        {"cell_type": "code", "source": f"{prompt}%%bash\n{prompt}{source}"},
    ]
    before = json.dumps(notebook)
    after = format_document(before, ".ipynb")
    timed, shell = json.loads(after)["cells"]
    assert timed["source"].splitlines()[0] == f"{prompt}%%timeit -n 1"
    assert "    .sort_values('year')" in timed["source"]
    assert shell == notebook["cells"][1]
    assert format_document(after, ".ipynb") == after


def test_fstring_literal_braces_do_not_interfere_with_call_delimiters(assert_format):
    assert_format(
        "print(df.assign(label=f'{{name {name}').head(5))\n",
        """\
        print(df
            .assign(
                label=f'{{name {name}')
            .head(5)
        )
        """,
    )


def test_inferred_frames_allow_methods_outside_fallback(assert_format):
    assert_format(
        """\
        import polars as tables
        from polars import LazyFrame as Lazy
        raw = tables.read_csv("data.csv")
        aliased = raw.clone()
        aliased.clone().clear()
        output = Lazy()
        output.clone().clear()
        """,
        """\
        import polars as tables
        from polars import LazyFrame as Lazy
        raw = tables.read_csv("data.csv")
        aliased = raw.clone()
        (aliased
            .clone()
            .clear()
        )
        output = Lazy()
        (output
            .clone()
            .clear()
        )
        """,
    )


def test_frame_annotation_and_rebinding_are_scoped(assert_format):
    assert_format(
        """\
        df = pl.DataFrame()
        def example(data: pl.DataFrame):
            data.clone().clear()
            df = {"key": 1}
            df.copy().clear()
        df.clone().clear()
        df = [3, 2, 1]
        df.reverse().sort()
        """,
        """\
        df = pl.DataFrame()
        def example(data: pl.DataFrame):
            (data
                .clone()
                .clear()
            )
            df = {"key": 1}
            df.copy().clear()
        (df
            .clone()
            .clear()
        )
        df = [3, 2, 1]
        df.reverse().sort()
        """,
    )


@pytest.mark.parametrize("prompt", ["", ">>> "])
def test_notebook_remembers_series_bindings_across_cells(notebook, prompt):
    sources = [
        'import polars as tables',
        'df = tables.read_csv("data.csv")',
        "column = df['height']",
        "column.cast(tables.Int32).clip(upper_bound=100).mean()",
        "rows = df.rows()",
        "rows.copy().sort()",
    ]
    notebook["cells"] = [{"cell_type": "code", "source": prompt + line} for line in sources]
    updated = format_document(json.dumps(notebook), ".ipynb", Settings())
    cells = json.loads(updated)["cells"]
    assert "    .clip(upper_bound=100)" in cells[3]["source"]
    assert cells[-1]["source"] == prompt + sources[-1]
    assert format_document(updated, ".ipynb", Settings()) == updated


def test_single_ordinary_method_stays_inline_on_a_known_frame(assert_format):
    source = 'df = pl.DataFrame()\ndf.clone()\n'
    assert_format(source, source)


def test_jupytext_tracks_bindings_without_using_markdown_examples():
    source = (
        '# %%\ndf = pl.DataFrame()\n'
        '# %% [markdown]\n# ```python\n# df = []\n# ```\n'
        '# %%\ndf.clone().clear()\n'
    )
    updated = format_document(source, ".py")
    assert "(df\n    .clone()\n    .clear()\n)" in updated
    assert format_document(updated, ".py") == updated
