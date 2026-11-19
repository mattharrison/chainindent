"""Apply Python formatting to source spans in teaching and notebook formats."""

import ast
import io
import json
import re
import tokenize
from collections.abc import Iterator
from difflib import SequenceMatcher
from typing import Any

from .bindings import Bindings
from .edits import apply_edits
from .formatter import Settings, format_python


PROMPT = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>>>>|\.\.\.)(?: |$)")
FENCE = re.compile(r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")
CELL = re.compile(r"^#\s*%%(?:\s+\[(?P<kind>[^\]]+)\])?(?:\s.*)?$")
PYTHON_LANGUAGES = {"python", "python3", "py", "ipython"}
PROMPT_LANGUAGES = {"pycon", "python-console", "doctest"}
PERCENT_COMMENT = re.compile(r"^#(?: |$)")
CELL_MAGIC = re.compile(r"\A(?:[ \t]*\n)*[ \t]*(?:>>> )?%%(?P<name>\w+)\b[^\n]*(?:\n|$)")
PYTHON_CELL_MAGICS = {"capture", "prun", "time", "timeit"}


def prompted(code: str) -> str:
    """Add Python console prompts based on each statement’s source span."""
    tree = ast.parse(code)
    continuations = set()
    for statement in tree.body:
        start = min([statement.lineno, *(d.lineno for d in getattr(statement, "decorator_list", []))])
        assert statement.end_lineno is not None
        continuations.update(range(start + 1, statement.end_lineno + 1))
    result = []
    for index, line in enumerate(code.splitlines(keepends=True), 1):
        content = line.removesuffix("\n")
        marker = "... " if index in continuations else ">>> "
        result.append((marker + content if content.strip() or index in continuations else "") + (
            "\n" if line.endswith("\n") else ""
        ))
    return "".join(result)


def format_prompt_group(
    lines: list[str], code_lines: list[str], indent: str, settings: Settings, bindings: Bindings
) -> list[str]:
    """Reformat a contiguous input block and retain its explicit console terminator."""
    terminator = lines[-1].strip() == "..."
    plain = "".join(code_lines[:-1] if terminator else code_lines)
    updated = format_python(plain, settings, bindings=bindings)
    if updated == plain:
        return lines
    result = [indent + line if line.strip() else line for line in (
        prompted(updated).splitlines(keepends=True)
    )]
    if terminator:
        result.append(lines[-1])
    return result


def format_transcript(text: str, settings: Settings, bindings: Bindings | None = None) -> str:
    """Format prompted input and leave output lines exactly as written."""
    lines = text.splitlines(keepends=True)
    context = bindings if bindings is not None else Bindings()
    result = []
    index = 0
    while index < len(lines):
        match = PROMPT.match(lines[index])
        if not match or match["marker"] != ">>>":
            result.append(lines[index])
            index += 1
            continue
        indent = match["indent"]
        start = index
        starts = []
        code_lines = []
        pending_comment = False
        while index < len(lines):
            match = PROMPT.match(lines[index])
            if not match or match["indent"] != indent:
                break
            code_line = lines[index][match.end():]
            if match["marker"] == ">>>":
                if not pending_comment:
                    starts.append(index - start)
                pending_comment = code_line.startswith("#")
            code_lines.append(code_line)
            index += 1
        prompt_lines = lines[start:index]
        updated = format_prompt_group(prompt_lines, code_lines, indent, settings, context)
        if updated is prompt_lines and len(starts) > 1:
            try:
                ast.parse("".join(code_lines))
            except SyntaxError:
                # An incomplete example should not stop adjacent prompted examples.
                ends = [*starts[1:], len(prompt_lines)]
                updated = [line for left, right in zip(starts, ends, strict=True)
                           for line in format_prompt_group(
                               prompt_lines[left:right], code_lines[left:right], indent, settings, context
                           )]
        result.extend(updated)
    return "".join(result)


def format_code(code: str, settings: Settings, bindings: Bindings | None = None) -> str:
    """Format either standalone Python or Python console input."""
    if magic := CELL_MAGIC.match(code):
        if magic["name"] not in PYTHON_CELL_MAGICS:
            return code
        # Timed and captured examples can execute in a different namespace.
        return code[:magic.end()] + format_code(code[magic.end():], settings)
    if re.match(r"\s*>>> ", code):
        return format_transcript(code, settings, bindings)
    return format_python(code, settings, bindings=bindings)


def format_markdown(text: str, settings: Settings) -> str:
    r"""Rewrite Python and PyCon fences while preserving surrounding Markdown.

    >>> sample = "```pycon\n>>> df.sort('year').head(5)\n5 rows\n```"
    >>> lines = format_markdown(sample, Settings()).splitlines()
    >>> lines[1:4]
    ['>>> (df', "...     .sort('year')", '...     .head(5)']
    >>> lines[-2]
    '5 rows'
    """
    lines = text.splitlines(keepends=True)
    result = []
    index = 0
    while index < len(lines):
        line = lines[index]
        result.append(line)
        index += 1
        opening = FENCE.match(line.removesuffix("\n"))
        if not opening:
            continue
        fence = opening["fence"]
        info = opening["info"].strip()
        if fence[0] == "`" and "`" in info:
            continue
        end = index
        closing = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
        while end < len(lines) and not closing.match(lines[end].removesuffix("\n")):
            end += 1
        body = lines[index:end]
        language = re.split(r"[\s,}]", info.lstrip("{."), maxsplit=1)[0].lower()
        if end < len(lines) and language in PYTHON_LANGUAGES | PROMPT_LANGUAGES:
            indent = opening["indent"]
            code = "".join(line[len(indent):] if line.startswith(indent) else line for line in body)
            if language in PROMPT_LANGUAGES:
                formatted = format_code(code, settings) if CELL_MAGIC.match(code) else format_transcript(code, settings)
            else:
                formatted = format_code(code, settings)
            if formatted != code:
                body = [indent + line if line.strip() else line for line in (
                    formatted.splitlines(keepends=True)
                )]
        result.extend(body)
        index = end
        if index < len(lines):
            result.append(lines[index])
            index += 1
    return "".join(result)


def format_percent_markdown(cell: str, settings: Settings) -> str:
    """Format fences inside a Jupytext percent-format Markdown cell."""
    lines = cell.splitlines(keepends=True)
    if any(line.strip() and not PERCENT_COMMENT.match(line) for line in lines):
        return cell
    text = "".join(PERCENT_COMMENT.sub("", line, count=1) for line in lines)
    formatted = format_markdown(text, settings)
    if formatted == text:
        return cell
    formatted_lines = formatted.splitlines(keepends=True)
    # Keep the original spelling of unchanged comment lines, including blank separators.
    result = []
    for tag, start, end, new_start, new_end in SequenceMatcher(
        None, text.splitlines(keepends=True), formatted_lines, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            result.extend(lines[start:end])
        else:
            result.extend("# " + line if line.strip() else "#" + line
                          for line in formatted_lines[new_start:new_end])
    return "".join(result)


def format_script(text: str, settings: Settings) -> str:
    """Handle both ordinary Python and Jupytext percent-format scripts."""
    if not re.search(r"(?m)^#\s*%%", text):
        return format_python(text, settings)
    lines = text.splitlines(keepends=True)
    try:
        markers = [
            (token.start[0] - 1, match["kind"])
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT and token.start[1] == 0
            if (match := CELL.match(token.string))
        ]
    except (SyntaxError, tokenize.TokenError):
        # An incomplete teaching cell can prevent tokenizing the remaining cells.
        markers = [(index, match["kind"]) for index, line in enumerate(lines)
                   if (match := CELL.match(line.removesuffix("\n")))]
    result = []
    bindings = Bindings()
    start = 0
    kind = None
    for end, next_kind in [*markers, (len(lines), None)]:
        cell = "".join(lines[start:end])
        if kind == "markdown":
            result.append(format_percent_markdown(cell, settings))
        elif kind in {None, "code"}:
            result.append(format_code(cell, settings, bindings))
        else:
            result.append(cell)
        if end < len(lines):
            result.append(lines[end])
        start = end + 1
        kind = next_kind
    return "".join(result)


def json_entries(text: str, start: int, *, mapping: bool) -> Iterator[tuple[str | None, int, int, Any]]:
    """Yield the key and value spans of one JSON object or array."""
    decoder = json.JSONDecoder()
    end_symbol = "}" if mapping else "]"
    index = start + 1
    whitespace = re.compile(r"\s*")

    def skip(offset: int) -> int:
        gap = whitespace.match(text, offset)
        assert gap is not None
        return gap.end()

    index = skip(index)
    while text[index] != end_symbol:
        key = None
        if mapping:
            key, index = decoder.raw_decode(text, index)
            index = skip(skip(index) + 1)
        value, end = decoder.raw_decode(text, index)
        yield key, index, end, value
        index = skip(end)
        if text[index] == ",":
            index = skip(index + 1)


def json_source(text: str, start: int, end: int, previous: str | list[str], formatted: str) -> str:
    """Serialize a replacement using the existing notebook source representation."""
    if isinstance(previous, str):
        return json.dumps(formatted, ensure_ascii=False)
    lines = formatted.splitlines(keepends=True)
    existing = text[start:end]
    layout = re.match(r"\[\n([ \t]*)", existing)
    closing = re.search(r"\n([ \t]*)\]$", existing)
    encoded = [json.dumps(line, ensure_ascii=False) for line in lines]
    if layout and closing and encoded:
        return "[\n" + ",\n".join(layout[1] + line for line in encoded) + "\n" + closing[1] + "]"
    return "[" + ", ".join(encoded) + "]"


def format_notebook(text: str, settings: Settings) -> str:
    """Update only code and Markdown source spans in a notebook JSON document."""
    notebook = json.loads(text)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("expected a Jupyter notebook with a cells array")
    metadata = notebook.get("metadata", {})
    language = metadata.get("language_info", {}).get("name") or metadata.get("kernelspec", {}).get("language")
    python_notebook = not language or language.lower() in PYTHON_LANGUAGES
    bindings = Bindings()
    array_start = next(start for key, start, _, _ in json_entries(text, len(text) - len(text.lstrip()),
                                                                 mapping=True) if key == "cells")
    edits = []
    for _, start, _, cell in json_entries(text, array_start, mapping=False):
        if not isinstance(cell, dict) or "source" not in cell:
            continue
        kind = cell.get("cell_type")
        if kind != "markdown" and (kind != "code" or not python_notebook):
            continue
        previous = cell["source"]
        if isinstance(previous, str):
            plain = previous
        elif isinstance(previous, list) and all(isinstance(line, str) for line in previous):
            plain = "".join(previous)
        else:
            continue
        formatted = format_code(plain, settings, bindings) if kind == "code" else format_markdown(plain, settings)
        if formatted != plain:
            source_start, source_end = next((left, right) for key, left, right, _ in (
                json_entries(text, start, mapping=True)
            ) if key == "source")
            edits.append((source_start, source_end, json_source(
                text, source_start, source_end, previous, formatted
            )))
    return apply_edits(text, reversed(edits))


def format_document(text: str, suffix: str, settings: Settings = Settings()) -> str:
    """Format a supported document using its filename suffix.

    >>> format_document("x = 1", ".py")
    'x = 1'
    """
    if "\r\n" in text and "\n" not in text.replace("\r\n", ""):
        return format_document(text.replace("\r\n", "\n"), suffix, settings).replace("\n", "\r\n")
    action = {".py": format_script, ".md": format_markdown, ".markdown": format_markdown,
              ".ipynb": format_notebook}[suffix]
    return action(text, settings)
