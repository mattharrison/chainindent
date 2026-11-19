"""Find chains in Python syntax and reflow only their source spans."""

import ast
import bisect
import io
import re
import tokenize
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from .bindings import BindingAnalysis, Bindings, LibraryMember
from .edits import Edit, apply_edits


@dataclass(frozen=True)
class Settings:
    """Layout preferences shared by all supported document formats."""

    line_length: int = 80
    indent_width: int = 4
    align_continuations: bool = False
    closing_paren_on_own_line: bool = False


COLUMN_METHODS = frozenset({"assign", "with_columns", "with_columns_seq", "select", "select_seq"})
# These operations are distinctive enough to recognize a frame when a
# notebook cell has no context about where its receiver was created.
POLARS_FRAME_SIGNALS = frozenset("""
    collect_schema drop_nans estimated_size get_column glimpse insert_column
    is_duplicated merge_sorted n_unique null_count rechunk to_arrow to_dicts
    to_dummies to_struct top_k upsample vstack write_database write_excel
    write_ipc write_json write_ndjson
""".split())
FRAME_METHODS = COLUMN_METHODS | POLARS_FRAME_SIGNALS | frozenset("""
    agg aggregate apply asfreq astype cache cast collect columns convert_dtypes corr
    drop drop_duplicates drop_nulls dropna dtypes explain explode fill_nan fill_null fillna filter
    gather_every group_by group_by_dynamic groupby head insert interpolate join join_asof
    join_where lazy limit melt merge n_chunks nunique order_by partition_by pct_change
    pipe pivot pivot_table query reindex rename replace reset_index resample rolling
    rolling_mean round sample schema schema_names select_dtypes set_index shift sink_csv
    sink_parquet sort sort_index sort_values stack tail take to_csv to_dict to_numpy
    to_pandas to_parquet to_polars to_series transpose unique unnest unpivot unstack
    value_counts with_row_count with_row_index write_csv write_parquet
""".split())
FRAME_SIGNALS = (COLUMN_METHODS | POLARS_FRAME_SIGNALS | frozenset("""
    collect drop_duplicates drop_nulls dropna fill_null fillna group_by group_by_dynamic
    groupby join_asof join_where lazy pivot_table reset_index select_dtypes set_index
    sort_index sort_values to_pandas to_polars unnest unpivot with_row_index
""".split())) - {"select"}
EXPR_FACTORIES = frozenset("""
    all any_horizontal all_horizontal coalesce col concat_list concat_str corr cov cum_count
    cum_fold date duration element exclude field first fold format int_range int_ranges last len
    lit map_batches max max_horizontal mean mean_horizontal median min min_horizontal nth
    reduce repeat rolling_corr rolling_cov
    row_index struct sum sum_horizontal when
""".split())
ACCESSORS = frozenset({"arr", "bin", "cat", "dt", "list", "meta", "name", "str", "struct"})
INDEXERS = frozenset({"at", "iat", "iloc", "loc"})
OPENERS = {"(": ")", "[": "]", "{": "}"}
LAYOUT_DIRECTIVE = re.compile(
    r"#\s*(?:chainindent\s*:\s*skip\b|fmt\s*:\s*skip\b|type:\s*ignore\b|noqa\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Token:
    """Significant token or comment positioned in the original source."""

    text: str
    start: int
    end: int
    comment: bool
    operator: bool


@dataclass(frozen=True)
class Step:
    """One dotted call, property, or indexer in a chain."""

    name: str
    start: int
    node: ast.expr
    accessor: bool


def wrap_single_step(step: Step) -> bool:
    """Apply the column layout rules to a chain with just one step."""
    if step.name in {"assign", "with_columns", "with_columns_seq"}:
        return True
    if not isinstance(step.node, ast.Call):
        return False
    if step.name in {"select", "select_seq"}:
        return bool(step.node.keywords) or any(
            not (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
            for arg in step.node.args
        )
    return step.name in {"agg", "aggregate"} and len(step.node.keywords) > 1


@dataclass(frozen=True)
class DictItem:
    """Dictionary entry and its span, including the key or unpacking marker."""

    value: ast.expr
    start: int
    end: int


type RenderNode = ast.expr | ast.keyword | DictItem
type ArgumentContainer = ast.Call | ast.List | ast.Set | ast.Tuple | ast.Dict | ast.Subscript
type Span = tuple[int, int]


@dataclass(frozen=True)
class ArgumentPart:
    """The value, punctuation, and comments attached to one argument."""

    node: RenderNode
    span: Span
    leading_comments: list[Token]
    trailing_comments: list[Token]
    had_comma: bool


class Source:
    """Translate parser and tokenizer positions into offsets in the same string."""
    def __init__(self, code: str) -> None:
        self.code = code
        self.lines = io.StringIO(code).readlines() + [""]
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.tokens = [
            Token(item.string, self.offsets[item.start[0] - 1] + item.start[1],
                  self.offsets[item.end[0] - 1] + item.end[1],
                  item.type == tokenize.COMMENT, item.type == tokenize.OP)
            for item in tokenize.generate_tokens(io.StringIO(code).readline)
            if item.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                 tokenize.DEDENT, tokenize.ENDMARKER}
        ]
        self.starts = [token.start for token in self.tokens]
        self.significant = [token for token in self.tokens if not token.comment]
        self.significant_starts = [token.start for token in self.significant]
        self.pairs: dict[int, Token] = {}
        stack = []
        for token in self.significant:
            if not token.operator:
                continue
            if token.text in OPENERS:
                stack.append(token)
            elif token.text in OPENERS.values() and stack:
                self.pairs[stack.pop().start] = token

    def offset(self, line: int, column: int) -> int:
        prefix = self.lines[line - 1].encode()[:column].decode()
        return self.offsets[line - 1] + len(prefix)

    def span(self, node: RenderNode | ast.stmt) -> Span:
        if isinstance(node, DictItem):
            return node.start, node.end
        assert node.end_lineno is not None and node.end_col_offset is not None
        return (self.offset(node.lineno, node.col_offset),
                self.offset(node.end_lineno, node.end_col_offset))

    def take(self, start: int, end: int) -> list[Token]:
        index = bisect.bisect_left(self.starts, start)
        stop = bisect.bisect_left(self.starts, end)
        return self.tokens[index:stop]

    def after(self, offset: int) -> Token | None:
        index = bisect.bisect_left(self.significant_starts, offset)
        if index < len(self.significant):
            return self.significant[index]
        return None

    def before(self, offset: int) -> Token | None:
        index = bisect.bisect_left(self.significant_starts, offset) - 1
        return self.significant[index] if index >= 0 else None

    def line_start(self, offset: int) -> int:
        return self.code.rfind("\n", 0, offset) + 1

    def indent(self, offset: int) -> str:
        start = self.line_start(offset)
        line = self.code[start:]
        return line[:len(line) - len(line.lstrip(" \t"))]

    def grouped(self, start: int, end: int) -> tuple[Token, Token] | None:
        left, right = self.before(start), self.after(end)
        if left and right and left.text == "(" and self.pairs.get(left.start) == right:
            return left, right
        return None

    def enclosed_span(self, node: RenderNode, lower: int, upper: int) -> Span:
        start, end = self.span(node)
        """Include explicit parentheses that belong to this expression within the bounds."""
        while group := self.grouped(start, end):
            left, right = group
            if left.start < lower or right.end > upper:
                break
            start, end = left.start, right.end
        return start, end


class Formatter:
    """Find and render edits for one parsed Python source unit."""
    def __init__(
        self, source: Source, tree: ast.Module, settings: Settings, bindings: Bindings | None = None
    ) -> None:
        self.source = source
        self.settings = settings
        self.space = " " * settings.indent_width
        self.parents = {child: parent for parent in ast.walk(tree)
                        for child in ast.iter_child_nodes(parent)}
        context = bindings if bindings is not None else Bindings()
        self.analysis = BindingAnalysis(context, EXPR_FACTORIES)
        self.analysis.visit(tree)

    def factory(self, node: ast.AST) -> LibraryMember | None:
        return self.analysis.factories.get(node) if isinstance(node, ast.Call) else None

    def chain(self, node: ast.expr) -> tuple[ast.expr, list[Step]]:
        """Split dotted operations from their receiver without splitting accessors like .str."""
        steps: list[Step] = []
        current = node
        while True:
            attr = current.func if isinstance(current, ast.Call) else current
            if self.factory(current):
                break
            if isinstance(current, ast.Subscript):
                attr = current.value
                if not isinstance(attr, ast.Attribute) or attr.attr not in INDEXERS:
                    break
            if not isinstance(attr, ast.Attribute):
                break
            receiver = attr.value
            if current is attr and not isinstance(receiver, ast.Call):
                break
            accessor = isinstance(receiver, ast.Attribute) and receiver.attr in ACCESSORS
            if accessor:
                receiver = receiver.value
            start, _ = self.source.span(current)
            _, base_end = self.source.span(receiver)
            # A parenthesized receiver can end before its closing delimiter.
            dot = next((token.start for token in self.source.take(base_end, self.source.span(attr)[1])
                        if token.text == "."), None)
            if dot is None or self.source.span(receiver)[0] < start:
                break
            steps.append(Step(attr.attr, dot, current, accessor))
            current = receiver
        return current, list(reversed(steps))

    def expression_factory(self, node: ast.AST) -> bool:
        factory = self.factory(node)
        return factory is not None and (factory == ("pandas", "col") or (
            factory[0] == "polars" and factory[1] in EXPR_FACTORIES
        ))

    def polars_frame_call(self, step: Step) -> bool:
        """Recognize frame operations with otherwise ambiguous method names."""
        if not isinstance(step.node, ast.Call):
            return False
        if step.name == "update":
            keywords = {keyword.arg for keyword in step.node.keywords}
            return bool(keywords & {"on", "left_on", "right_on", "include_nulls"})
        if step.name in {"select", "filter"}:
            args = [*step.node.args, *(keyword.value for keyword in step.node.keywords)]
            return any(self.expression_factory(part) for arg in args for part in ast.walk(arg))
        return False

    def kind(self, node: RenderNode) -> Literal["frame", "expression"] | None:
        if not isinstance(node, ast.expr):
            return None
        base, steps = self.chain(node)
        if not steps:
            return None
        factory = self.factory(base)
        receiver = self.analysis.facts.get(base)
        if receiver == "other":
            return None
        if receiver == "expression" or self.expression_factory(base) or (
            isinstance(base, (ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp))
            and any(self.expression_factory(part) for part in ast.walk(base))
        ):
            return "expression"
        names = [step.name for step in steps]
        accessor = next((i for i, step in enumerate(steps) if step.accessor), None)
        if accessor is not None and not any(
            name in FRAME_SIGNALS or name == "select" for name in names[:accessor]
        ):
            return "expression"
        if len(steps) == 1:
            return "frame" if wrap_single_step(steps[0]) else None
        if isinstance(base, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)):
            return None
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
            if base.func.id in {"Path", "PurePath", "PurePosixPath", "PureWindowsPath"}:
                return None
        if (
            receiver == "frame"
            or factory
            or FRAME_SIGNALS.intersection(names)
            or sum(name in FRAME_METHODS or name in INDEXERS for name in names) >= 2
            or any(self.polars_frame_call(step) for step in steps)
        ):
            return "frame"
        return None

    def fits(self, text: str, reserve: int = 0) -> bool:
        return len(text.rsplit("\n", 1)[-1].expandtabs(8)) + reserve <= self.settings.line_length

    def fits_all_lines(self, text: str, reserve: int) -> bool:
        lines = text.split("\n")
        return self.fits(lines[-1], reserve) and all(self.fits(line) for line in lines[:-1])

    def comments(self, start: int, end: int) -> list[Token]:
        return [token for token in self.source.take(start, end) if token.comment]

    def standalone_comments(self, comments: Iterable[Token], indent: str) -> str:
        return "".join("\n" + indent + comment.text for comment in comments)

    def inline(self, start: int, end: int) -> str | None:
        tokens = self.source.take(start, end)
        if any(token.comment or "\n" in token.text for token in tokens):
            return None
        parts = []
        last = start
        for token in tokens:
            gap = self.source.code[last:token.start]
            if "\n" in gap:
                previous = parts[-1] if parts else ""
                gap = "" if token.text in {".", ")", "]", "}", ","} or previous in {
                    ".", "(", "[", "{",
                } else " "
            parts.extend((gap, token.text))
            last = token.end
        return "".join(parts).strip()

    def original(self, start: int, end: int, prefix: str) -> str:
        raw = self.source.code[start:end]
        if "\n" not in raw:
            return prefix + raw
        tokens = self.source.take(start, end)
        if any("\n" in token.text for token in tokens):
            return prefix + raw
        line = prefix.rsplit("\n", 1)[-1]
        indent = line[:len(line) - len(line.lstrip(" \t"))]
        before = self.source.indent(start)
        first, *rest = raw.split("\n")
        return prefix + first + "".join(
            "\n" + (indent + line[len(before):] if line.startswith(before) else line)
            if line.strip() else "\n" for line in rest
        )

    def delimiters(self, node: ast.Call | ast.Subscript) -> tuple[Token, Token] | None:
        """Locate a call's parentheses or a subscript's brackets."""
        target, left = (node.func, "(") if isinstance(node, ast.Call) else (node.value, "[")
        _, target_end = self.source.span(target)
        opener = self.source.after(target_end)
        while opener and opener.text == ")":
            opener = self.source.after(opener.end)
        closer = self.source.pairs.get(opener.start) if opener else None
        if opener and opener.text == left and closer and closer.end == self.source.span(node)[1]:
            return opener, closer
        return None

    def receiver(self, base: ast.expr, dot: int, prefix: str, indent: str) -> tuple[str, int]:
        span = self.source.enclosed_span(base, 0, dot)
        return self.value(base, prefix, indent, span=span), span[1]

    def end_comment(self, text: str, comment: Token, start: int, indent: str) -> str:
        if self.fits(text + "  " + comment.text):
            return text + "  " + comment.text
        # Moving a comment above a multiline call can place it before comments
        # inside that call. In that case, give its closing token a separate line.
        if text.endswith((")", "]", "}")) and any(
            line.lstrip().startswith("#") or "  #" in line for line in text[start:].split("\n")
        ):
            return text[:-1] + "\n" + indent + text[-1] + "  " + comment.text
        line_start = text.rfind("\n", 0, start) + 1
        if text[line_start:start].strip():
            padding = "\n" + indent
            return text[:start].rstrip() + padding + comment.text + padding + text[start:]
        return text[:line_start] + indent + comment.text + "\n" + text[line_start:]

    def render_chain(
        self, node: ast.expr, prefix: str, indent: str, *, expression: bool = False, reserve: int = 0
    ) -> str:
        """Lay out steps, allowing nested expressions to share a line."""
        base, steps = self.chain(node)
        text, end = self.receiver(base, steps[0].start, prefix, indent)
        item_start = len(prefix)
        for index, step in enumerate(steps):
            tail = reserve if index == len(steps) - 1 else 0
            for comment in self.comments(end, step.start):
                if "\n" not in self.source.code[end:comment.start]:
                    text = self.end_comment(text, comment, item_start, indent)
                else:
                    text += "\n" + indent + comment.text
            end = self.source.span(step.node)[1]
            inline = self.inline(step.start, end)
            if (expression and inline and self.fits(text + inline, tail)
                    and not self.comments(self.source.span(base)[1], step.start)):
                text += inline
                continue
            item_start = len(text) + 1 + len(indent)
            if isinstance(step.node, (ast.Call, ast.Subscript)) and (bounds := self.delimiters(step.node)):
                opener, closer = bounds
                head = self.inline(step.start, opener.end)
                if head:
                    text += "\n" + self.arguments(
                        step.node, opener, closer, indent + head, indent, step.name, reserve=tail
                    )
                    continue
            text += "\n" + self.original(step.start, end, indent)
        return text

    def argument_parts(
        self, container: ArgumentContainer, opener: Token, closer: Token
    ) -> tuple[list[ArgumentPart], list[Token], bool] | None:
        """Associate arguments with comments and commas from their source spans."""
        nodes: Sequence[RenderNode]
        tuple_items = isinstance(container, ast.Tuple)
        if isinstance(container, ast.Call):
            nodes = sorted([*container.args, *container.keywords], key=lambda arg: self.source.span(arg)[0])
        elif isinstance(container, ast.Dict):
            nodes = list(self.dictionary_items(container, opener, closer))
        elif isinstance(container, ast.Subscript):
            first = self.source.after(opener.end)
            tuple_items = isinstance(container.slice, ast.Tuple) and bool(
                first and self.source.pairs.get(first.start) != self.source.before(closer.start)
            )
            nodes = container.slice.elts if tuple_items and isinstance(container.slice, ast.Tuple) else [container.slice]
        else:
            nodes = container.elts
        result: list[ArgumentPart] = []
        end = opener.end
        for index, node in enumerate(nodes):
            start, stop = self.source.enclosed_span(node, end, closer.start)
            if start < opener.end or stop > closer.start:
                return None
            following = self.source.after(stop)
            comma = following if following and following.text == "," else None
            limit = comma.end if comma else stop
            next_start = self.source.span(nodes[index + 1])[0] if index + 1 < len(nodes) else closer.start
            trailing = []
            for comment in self.comments(limit, next_start):
                if "\n" in self.source.code[limit:comment.start]:
                    break
                trailing.append(comment)
                limit = comment.end
            result.append(ArgumentPart(node, (start, stop), self.comments(end, start), trailing, bool(comma)))
            end = limit
        return result, self.comments(end, closer.start), tuple_items

    def infix(
        self, node: ast.BinOp | ast.BoolOp | ast.Compare, prefix: str, indent: str, reserve: int
    ) -> str:
        if isinstance(node, ast.BinOp):
            operands = [node.left, node.right]
        elif isinstance(node, ast.BoolOp):
            operands = node.values
        else:
            operands = [node.left, *node.comparators]
        start, end = self.source.span(node)
        spans = [self.source.enclosed_span(part, start, end) for part in operands]
        operators = [self.inline(left[1], right[0]) for left, right in zip(spans, spans[1:])]
        if not all(operators):
            return self.original(start, end, prefix)
        text = self.value(operands[0], prefix, indent, span=spans[0])
        for index, (operand, span, op) in enumerate(zip(operands[1:], spans[1:], operators, strict=True), 1):
            assert op is not None
            tail = reserve if index == len(operands) - 1 else 0
            inline = self.value(operand, text + " " + op + " ", indent, span=span, reserve=tail)
            if self.fits_all_lines(inline, tail):
                text = inline
            else:
                continuation = indent + self.space
                text += "\n" + self.value(
                    operand, continuation + op + " ", continuation, span=span, reserve=tail
                )
        return text

    def value(
        self, node: RenderNode, prefix: str, indent: str, *, span: Span | None = None, reserve: int = 0
    ) -> str:
        start, end = span or self.source.span(node)
        inner_start, inner_end = self.source.span(node)
        grouped = start != inner_start or end != inner_end
        if isinstance(node, (ast.keyword, DictItem, ast.Lambda)) and not grouped:
            subject = node.body if isinstance(node, ast.Lambda) else node.value
            val_start, val_end = self.source.enclosed_span(subject, start, end)
            between = self.source.code[start:val_start]
            if "\n" not in between and not self.comments(start, val_start):
                return self.value(subject, prefix + between, indent,
                                  span=(val_start, val_end), reserve=reserve)
        kind = self.kind(node)
        if isinstance(node, ast.expr) and kind == "frame":
            if not grouped or not (
                self.comments(start, inner_start) or self.comments(inner_end, end)
            ):
                return self.render_chain(node, prefix, indent + self.space, reserve=reserve)
        inline = self.inline(start, end)
        if inline is not None and self.fits(prefix + inline, reserve):
            return prefix + inline
        original = self.original(start, end, prefix)
        if grouped:
            opening = self.inline(start, inner_start)
            closing = self.inline(inner_end, end)
            if opening is None or closing is None:
                return original
            result = self.value(node, prefix + opening, indent, reserve=reserve + len(closing)) + closing
        elif isinstance(node, ast.expr) and kind == "expression":
            result = self.render_chain(node, prefix, indent + self.space, expression=True, reserve=reserve)
        elif isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare)):
            result = self.infix(node, prefix, indent, reserve)
        elif isinstance(node, (ast.Call, ast.Subscript)):
            bounds = self.delimiters(node)
            if bounds is None:
                return original
            opener, closer = bounds
            head = self.inline(start, opener.end)
            if not head:
                return original
            if not self.fits(prefix + head):
                target = node.func if isinstance(node, ast.Call) else node.value
                target_span = self.source.enclosed_span(target, start, opener.start)
                head = self.value(target, prefix, indent, span=target_span, reserve=1) + opener.text
            else:
                head = prefix + head
            result = self.arguments(node, opener, closer, head, indent, reserve=reserve)
        elif isinstance(node, (ast.List, ast.Set, ast.Tuple, ast.Dict)):
            opener, closer = self.source.after(start), self.source.before(end)
            if opener and closer and self.source.pairs.get(opener.start) == closer:
                result = self.arguments(node, opener, closer, prefix + opener.text, indent, reserve=reserve)
            else:
                return original
        else:
            return original
        if not self.fits_all_lines(result, reserve) and self.fits_all_lines(original, reserve):
            return original
        return result

    def dictionary_items(self, node: ast.Dict, opener: Token, closer: Token) -> Iterator[DictItem]:
        lower = opener.end
        for key, value in zip(node.keys, node.values, strict=True):
            val_start, end = self.source.enclosed_span(value, lower, closer.start)
            if key is None:
                marker = self.source.before(val_start)
                assert marker is not None and marker.text == "**"
                start = marker.start
            else:
                start, _ = self.source.enclosed_span(key, lower, val_start)
            yield DictItem(value, start, end)
            lower = end

    def arguments(
        self,
        node: ArgumentContainer,
        opener: Token,
        closer: Token,
        prefix: str,
        indent: str,
        method: str | None = None,
        *,
        reserve: int = 0,
    ) -> str:
        """Pack arguments while budgeting space for punctuation added by callers."""
        parts = self.argument_parts(node, opener, closer)
        if parts is None:
            return prefix + self.source.code[opener.end:closer.end]
        args, ending, tuple_items = parts
        keywords = sum(isinstance(part.node, ast.keyword) for part in args)
        column_layout = (method in COLUMN_METHODS and keywords > 0) or (
            method in {"agg", "aggregate"} and keywords > 1
        )
        item_indent = indent + self.space
        if self.settings.align_continuations and not column_layout:
            item_indent = " " * len(prefix.rsplit("\n", 1)[-1].expandtabs(8))
        host_chain = len(args) == 1 and self.kind(args[0].node) == "frame"
        closer_width = len(closer.text) + reserve
        text = prefix
        wrapped = False
        force_break = False
        for index, part in enumerate(args):
            first = index == 0
            last = index == len(args) - 1
            arg = part.node
            trailing = part.trailing_comments
            for comment in part.leading_comments:
                if first and "\n" not in self.source.code[opener.end:comment.start]:
                    text += "  " + comment.text
                else:
                    text += "\n" + item_indent + comment.text
                force_break = True
            named = column_layout and isinstance(arg, ast.keyword)
            comma = not last or (part.had_comma and bool(trailing)) or (tuple_items and first)
            suffix = "," if comma else ""
            subject = arg.value if isinstance(arg, ast.keyword) else arg
            frame = self.kind(subject) == "frame"
            tail = len(suffix) + (closer_width if last and not (ending or frame) else 0)

            may_share_line = not (named or force_break)
            separator = "" if first else " "
            inline = self.inline(*part.span) if may_share_line and not frame else None
            hanging = None
            if inline is not None and self.fits(text + separator + inline, tail):
                item_start = len(text) + len(separator)
                text += separator + inline
            else:
                item_start = len(text)
                if first and may_share_line:
                    candidate = self.value(arg, text, indent, span=part.span, reserve=tail)
                    if frame or "\n" in candidate[item_start:]:
                        hanging = candidate
                if (hanging is not None and frame and trailing and item_start == len(prefix)
                        and not self.fits(hanging + suffix + "  " + trailing[0].text)
                        and not self.comments(*part.span)):
                    text += "\n" + item_indent + trailing[0].text
                    trailing = trailing[1:]
                    hanging = None
                if hanging is not None:
                    text = hanging
                    wrapped |= "\n" in hanging[item_start:]
                else:
                    item_start = len(text) + 1 + len(item_indent)
                    text += "\n" + self.value(arg, item_indent, item_indent, span=part.span, reserve=tail)
                    wrapped = True
            text += suffix
            force_break = frame
            for comment in trailing:
                text = self.end_comment(text, comment, item_start, item_indent)
                force_break |= text.endswith(comment.text)
            if trailing and not force_break and last and not tuple_items:
                text = text.removesuffix(",")
        if ending:
            text += self.standalone_comments(ending, item_indent)
        if ending or (force_break and not host_chain) or (host_chain and wrapped and method is None) or (
            wrapped and self.settings.closing_paren_on_own_line
        ):
            if args and args[-1].had_comma and not force_break and not ending and not text.endswith(","):
                text += ","
            return text + "\n" + indent + closer.text
        return text + closer.text

    def protected(self, node: ast.AST) -> bool:
        while not isinstance(node, ast.stmt):
            parent = self.parents.get(node)
            if parent is None:
                return False
            node = parent
        start, end = self.source.span(node)
        line_start = self.source.line_start(start)
        previous_start = self.source.line_start(max(0, line_start - 1))
        line_end = self.source.code.find("\n", end)
        if line_end == -1:
            line_end = len(self.source.code)
        return any(
            LAYOUT_DIRECTIVE.match(token.text) and (
                token.start >= line_start or not self.source.code[previous_start:token.start].strip()
            )
            for token in self.comments(previous_start, line_end)
        )

    def edit(self, node: ast.AST) -> Edit | None:
        if not isinstance(node, ast.expr):
            return None
        start, end = self.source.span(node)
        if self.kind(node) == "frame":
            group = self.source.grouped(start, end)
            if group:
                parent = self.parents.get(node)
                bounds = self.delimiters(parent) if isinstance(parent, ast.Call) else None
                if bounds and bounds[0] == group[0]:
                    return None
            region_start = group[0].start if group else start
            indent = self.source.indent(region_start)
            context = self.source.code[self.source.line_start(region_start):region_start]
            prefix = context + "("
            if group and (leading := self.comments(group[0].end, start)):
                prefix += self.standalone_comments(leading, indent + self.space)
                prefix += "\n" + indent + self.space
            text = self.render_chain(node, prefix, indent + self.space)
            if group:
                left, right = group
                text += self.standalone_comments(self.comments(end, right.start), indent + self.space)
                start, end = left.start, right.end
            return start, end, text[len(context):] + "\n" + indent + ")"
        if isinstance(node, ast.Call) and any(
            self.kind(arg) == "frame" for arg in [*node.args, *(arg.value for arg in node.keywords)]
        ):
            if bounds := self.delimiters(node):
                opener, closer = bounds
                lead = self.inline(start, opener.end)
                if lead:
                    indent = self.source.indent(start)
                    context = self.source.code[self.source.line_start(start):start]
                    text = self.arguments(node, opener, closer, context + lead, indent)
                    return start, end, text[len(context):]
        return None

    def edits(self, tree: ast.Module) -> Iterator[Edit]:
        def walk(node: ast.AST) -> Iterator[Edit]:
            edit = self.edit(node)
            if edit and not self.protected(node):
                yield edit
                return
            for child in ast.iter_child_nodes(node):
                yield from walk(child)

        yield from walk(tree)


def format_python(code: str, settings: Settings = Settings(), *, bindings: Bindings | None = None) -> str:
    r"""Reformat Python chains while preserving syntax and comment order.

    >>> print(format_python("df.sort('year').head(5)"))
    (df
        .sort('year')
        .head(5)
    )
    >>> format_python("# chainindent: skip\ndf.sort('year').head(5)").splitlines()[-1]
    "df.sort('year').head(5)"
    """
    if "\r\n" in code and "\n" not in code.replace("\r\n", ""):
        plain = code.replace("\r\n", "\n")
        return format_python(plain, settings, bindings=bindings).replace("\n", "\r\n")
    try:
        tree = ast.parse(code, type_comments=True)
        source = Source(code)
    except (SyntaxError, tokenize.TokenError):
        return code
    formatter = Formatter(source, tree, settings, bindings)
    edits = sorted(set(formatter.edits(tree)), reverse=True)
    if not edits:
        return code

    def structure(parsed: ast.Module) -> str:
        for directive in parsed.type_ignores:
            directive.lineno = 0
        return ast.dump(parsed)

    original_ast = structure(tree)
    old_comments = [token.text for token in source.tokens if token.comment]

    def valid(candidate: str) -> bool:
        try:
            parsed = ast.parse(candidate, type_comments=True)
            if structure(parsed) != original_ast:
                return False
            return [token.text for token in Source(candidate).tokens if token.comment] == old_comments
        except (SyntaxError, tokenize.TokenError):
            return False

    candidate = apply_edits(code, edits)
    if candidate == code or valid(candidate):
        return candidate
    # A complex surrounding construct can block one rewrite without blocking the file.
    candidate = code
    last = len(code)
    for start, end, replacement in edits:
        if end <= last and code[start:end] != replacement:
            trial = candidate[:start] + replacement + candidate[end:]
            if valid(trial):
                candidate = trial
                last = start
    return candidate
