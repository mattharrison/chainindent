"""Command line for formatting files or reading Python from standard input."""

import argparse
import difflib
import io
import os
import stat
import sys
import tempfile
import tokenize
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

from .documents import format_document
from .formatter import Settings


SUFFIXES = {".py", ".md", ".markdown", ".ipynb"}
EXCLUDED_DIRS = {"__pycache__", "node_modules", "build", "dist", "venv"}


def config_for(path: Path) -> tuple[dict[str, object], Path | None]:
    """Find the nearest project settings for the input path."""
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        candidate = directory / "pyproject.toml"
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            config = tomllib.load(handle).get("tool", {}).get("chainindent")
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError(f"{candidate}: expected a [tool.chainindent] table")
            return config, candidate
    return {}, None


def settings_for(path: Path, args: argparse.Namespace) -> Settings:
    """Apply project settings and then explicit command-line overrides."""
    values = asdict(Settings())
    config, source = config_for(path)
    for raw_key, value in config.items():
        key = raw_key.replace("-", "_")
        if key not in values:
            raise ValueError(f"{source}: unknown Chainindent setting {raw_key!r}")
        expected = type(values[key])
        if type(value) is not expected or (expected is int and isinstance(value, int) and value < 1):
            raise ValueError(f"{source}: invalid value for {raw_key}: {value!r}")
        values[key] = value
    values.update({key: getattr(args, key) for key in values if getattr(args, key) is not None})
    return Settings(**values)


def eligible_paths(path: Path) -> Iterator[Path]:
    """Yield supported files for an explicit path or recursive directory scan."""
    if path.is_file() and path.suffix.lower() in SUFFIXES:
        yield path
    elif path.is_dir():
        for directory, dirs, files in os.walk(path):
            dirs[:] = sorted(name for name in dirs if not name.startswith(".") and name not in EXCLUDED_DIRS)
            for name in sorted(files):
                candidate = Path(directory, name)
                if not name.startswith(".") and candidate.suffix.lower() in SUFFIXES:
                    yield candidate
    else:
        raise ValueError(f"expected a Python, Markdown, or notebook file or a directory: {path}")


def write_file(path: Path, data: bytes) -> None:
    """Replace a file in the same directory after a complete write."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".chainindent-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.close()
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def print_diff(before: str, after: str, name: str) -> None:
    """Print a unified diff with markers for missing terminal newlines."""
    for line in difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                     fromfile=name, tofile=name):
        sys.stdout.write(line)
        if not line.endswith("\n"):
            sys.stdout.write("\n\\ No newline at end of file\n")


def positive_int(value: str) -> int:
    """Parse a positive integer for a numeric command-line option.

    >>> positive_int("73")
    73
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return 2 for errors or 1 for formatting needed in check mode."""
    parser = argparse.ArgumentParser(description="Format pandas and Polars chains.")
    parser.add_argument("paths", nargs="+", help="Files, directories, or - for stdin")
    parser.add_argument("--stdin-filename", type=Path, help="Choose stdin format and config by filename")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diff", action="store_true", help="Show changes without writing")
    mode.add_argument("--check", action="store_true", help="Exit 1 if changes are needed")
    parser.add_argument("--line-length", type=positive_int, help="Preferred line width (default 80)")
    parser.add_argument("--indent-width", type=positive_int, help="Spaces per additional level (default 4)")
    parser.add_argument("--align-continuations", action=argparse.BooleanOptionalAction, default=None,
                        help="Align wrapped arguments and list items with the first item")
    parser.add_argument("--closing-paren-on-own-line", action=argparse.BooleanOptionalAction, default=None,
                        help="Put the closing parenthesis of multiline method calls on its own line")
    args = parser.parse_args(argv)
    if "-" in args.paths and args.paths != ["-"]:
        parser.error("stdin must be the only input")
    if args.stdin_filename and args.paths != ["-"]:
        parser.error("--stdin-filename applies only to stdin")

    changed = False
    errors = False
    paths = []
    seen = set()
    for name in args.paths:
        if name == "-":
            paths.append((args.stdin_filename or Path("stdin.py"), True))
            continue
        try:
            for candidate in eligible_paths(Path(name)):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append((candidate, False))
        except (OSError, ValueError) as error:
            print(f"chainindent: {error}", file=sys.stderr)
            errors = True

    for path, stdin in paths:
        try:
            suffix = path.suffix.lower()
            if suffix not in SUFFIXES:
                raise ValueError(f"unsupported suffix {suffix!r}")
            settings = settings_for(path.absolute(), args)
            raw = sys.stdin.buffer.read() if stdin else path.read_bytes()
            if suffix == ".py":
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            else:
                encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
            before = raw.decode(encoding)
            after = format_document(before, suffix, settings)
            if before != after:
                changed = True
                if args.diff:
                    print_diff(before, after, str(path))
                elif args.check:
                    print(f"Would format {path}", file=sys.stderr)
                elif not stdin:
                    write_file(path.resolve(), after.encode(encoding))
            if stdin and not (args.diff or args.check):
                sys.stdout.buffer.write(after.encode(encoding))
        except (OSError, ValueError, SyntaxError) as error:
            print(f"chainindent: {path}: {error}", file=sys.stderr)
            errors = True
    return 2 if errors else int(args.check and changed)


if __name__ == "__main__":
    raise SystemExit(main())
