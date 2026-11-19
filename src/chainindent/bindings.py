"""Infer frame and expression receivers from earlier Python bindings."""

import ast
from dataclasses import dataclass, field
from typing import Literal


type Kind = Literal["frame", "expression", "other"]
type LibraryMember = tuple[str, str]

FRAME_TYPES = {"DataFrame", "LazyFrame", "Series"}
FRAME_FACTORIES = {
    *FRAME_TYPES, "concat", "concat_strict", "crosstab", "from_arrow", "from_dataframe",
    "from_dict", "from_dicts", "from_numpy", "from_pandas", "from_records", "merge",
    "merge_asof", "merge_ordered", "pivot", "pivot_table",
}
FRAME_RESULTS = frozenset("""
    agg aggregate alias asfreq astype cache cast clear clone collect convert_dtypes
    copy describe drop drop_duplicates drop_nans drop_nulls dropna explode extend
    fill_nan fill_null fillna filter first gather_every get_column group_by group_by_dynamic
    groupby head hstack insert_column interpolate join join_asof join_where last lazy limit
    map_batches melt merge merge_sorted order_by pivot pivot_table query rechunk reindex
    rename replace reset_index reshape rolling round sample select select_dtypes select_seq
    set_index set_sorted shift slice sort sort_index sort_values stack tail take
    to_dummies to_frame to_pandas to_polars to_series transpose unique unnest unpivot
    unstack update upsample vstack with_columns with_columns_seq with_row_count with_row_index
""".split())
NON_FRAME_RESULTS = frozenset("""
    collect_schema count_elements estimated_size explain get_column_index is_empty item
    iter_columns iter_rows n_chunks named_rows n_unique row rows rows_by_key serialize
    to_arrow to_csv to_dict to_dicts to_html to_json to_list to_markdown to_numpy
    to_pylist to_sql to_string write_csv write_excel write_ipc write_json write_ndjson write_parquet
""".split())
OTHER_BUILDERS = {
    "Path", "PurePath", "PurePosixPath", "PureWindowsPath", "bytes", "dict", "float", "int",
    "list", "range", "set", "str", "tuple",
}


@dataclass
class Bindings:
    """Carry top-level bindings to the next code cell in the same document."""

    values: dict[str, Kind | None] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=lambda: {
        "pd": "pandas", "pandas": "pandas", "pl": "polars", "polars": "polars",
    })
    imports: dict[str, LibraryMember] = field(default_factory=dict)
    functions: dict[str, Kind] = field(default_factory=dict)

    def copy(self) -> "Bindings":
        """Copy the visible state before visiting a nested scope."""
        return Bindings(self.values.copy(), self.modules.copy(), self.imports.copy(), self.functions.copy())

    def forget(self, name: str) -> None:
        """Remove previous facts when a name is rebound or shadowed."""
        self.values.pop(name, None)
        self.modules.pop(name, None)
        self.imports.pop(name, None)
        self.functions.pop(name, None)

    def keep_shared(self, first: "Bindings", second: "Bindings") -> None:
        """Retain facts that agree across two possible paths."""
        def shared[T](left: dict[str, T], right: dict[str, T]) -> dict[str, T]:
            return {key: value for key, value in left.items() if key in right and right[key] == value}

        self.values = shared(first.values, second.values)
        self.modules = shared(first.modules, second.modules)
        self.imports = shared(first.imports, second.imports)
        self.functions = shared(first.functions, second.functions)


class BindingAnalysis(ast.NodeVisitor):
    """Track simple assignments in source order and isolate function-local names."""

    def __init__(self, bindings: Bindings, expression_factories: frozenset[str]) -> None:
        self.bindings = bindings
        self.expression_factories = expression_factories
        self.facts: dict[ast.expr, Kind | None] = {}
        self.factories: dict[ast.Call, LibraryMember] = {}

    def visit(self, node: ast.AST) -> None:
        super().visit(node)
        if isinstance(node, ast.expr):
            self.facts[node] = self.infer(node)

    def member(self, node: ast.expr) -> LibraryMember | None:
        if isinstance(node, ast.Name):
            return self.bindings.imports.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if module := self.bindings.modules.get(node.value.id):
                return module, node.attr
        return None

    def annotation(self, node: ast.expr | None) -> Kind | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
        if node is not None and (member := self.member(node)):
            if member[1] in FRAME_TYPES:
                return "frame"
            if member == ("polars", "Expr"):
                return "expression"
        return None

    def infer(self, node: ast.expr) -> Kind | None:
        """Infer the kind of a value without executing user code."""
        if isinstance(node, ast.Name):
            return self.bindings.values.get(node.id)
        if isinstance(node, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple, ast.JoinedStr,
                             ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return "other"
        if isinstance(node, ast.BinOp):
            if "expression" in {self.facts.get(node.left), self.facts.get(node.right)}:
                return "expression"
        if isinstance(node, ast.Subscript):
            is_indexer = isinstance(node.value, ast.Attribute) and node.value.attr in {"loc", "iloc"}
            receiver = node.value.value if is_indexer and isinstance(node.value, ast.Attribute) else node.value
            if self.facts.get(receiver) == "frame":
                selectors = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if any(isinstance(index, (ast.Slice, ast.List)) for index in selectors):
                    return "frame"
                if not is_indexer and not isinstance(node.slice, ast.Tuple):
                    return "frame"
        if isinstance(node, ast.Call):
            func = node.func
            if member := self.member(func):
                self.factories[node] = member
                module, name = member
                if (module == "polars" and name in self.expression_factories) or member == ("pandas", "col"):
                    return "expression"
                options = {arg.arg: arg.value for arg in node.keywords}
                if name in {"read_csv_batched", "read_excel", "read_pickle"} or (
                    module == "pandas" and options.keys() & {"chunksize", "iterator"}
                ):
                    return None
                if name in FRAME_FACTORIES or name.startswith(("read_", "scan_")):
                    return "frame"
            if isinstance(func, ast.Name):
                if func.id in OTHER_BUILDERS:
                    return "other"
                return self.bindings.functions.get(func.id)
            if isinstance(func, ast.Attribute):
                receiver = self.facts.get(func.value)
                if isinstance(func.value, ast.Attribute) and receiver is None:
                    if func.value.attr in {"arr", "bin", "dt", "list", "name", "str", "struct"}:
                        receiver = self.facts.get(func.value.value)
                if receiver == "expression":
                    return "expression"
                if receiver == "frame":
                    if func.attr in NON_FRAME_RESULTS:
                        return "other"
                    if func.attr in FRAME_RESULTS:
                        return "frame"
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.bindings.forget(name)
            self.bindings.values[name] = "other" if alias.name in {"pandas", "polars"} else None
            if alias.name in {"pandas", "polars"}:
                self.bindings.modules[name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self.bindings.forget(name)
            if node.module in {"pandas", "polars"}:
                self.bindings.imports[name] = (node.module, alias.name)

    def bind(self, target: ast.expr, kind: Kind | None) -> None:
        if isinstance(target, ast.Name):
            self.bindings.forget(target.id)
            self.bindings.values[target.id] = kind
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item, None)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, None)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        kind = self.facts[node.value]
        for target in node.targets:
            self.visit(target)
            self.bind(target, kind)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        kind = self.facts[node.value] if node.value is not None else None
        self.bind(node.target, kind if kind is not None else self.annotation(node.annotation))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.bind(node.target, self.facts[node.value])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self.bind(node.target, None)

    def visit_Delete(self, node: ast.Delete) -> None:
        self.generic_visit(node)
        for target in node.targets:
            self.bind(target, None)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        args = [arg for arg in ast.iter_child_nodes(node.args) if isinstance(arg, ast.arg)]
        parameters = [(arg.arg, self.annotation(arg.annotation)) for arg in args]
        enclosing = self.bindings
        kind = self.annotation(node.returns)
        enclosing.forget(node.name)
        enclosing.values[node.name] = "other"
        if kind:
            enclosing.functions[node.name] = kind
        local = enclosing.copy()
        # A local assignment shadows the outer name even before it runs.
        pending: list[ast.AST] = list(node.body)
        while pending:
            statement = pending.pop()
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.forget(statement.name)
                continue
            if isinstance(statement, ast.Name) and isinstance(statement.ctx, (ast.Store, ast.Del)):
                local.forget(statement.id)
            if not isinstance(statement, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                pending.extend(ast.iter_child_nodes(statement))
        self.bindings = local
        try:
            for name, kind in parameters:
                self.bindings.forget(name)
                self.bindings.values[name] = kind
            for statement in node.body:
                self.visit(statement)
        finally:
            self.bindings = enclosing

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        parent = self.bindings
        local = parent.copy()
        for arg in ast.iter_child_nodes(node.args):
            if isinstance(arg, ast.arg):
                local.forget(arg.arg)
        self.bindings = local
        try:
            self.visit(node.body)
        finally:
            self.bindings = parent

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for header in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(header)
        enclosing = self.bindings
        enclosing.forget(node.name)
        enclosing.values[node.name] = "other"
        self.bindings = enclosing.copy()
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.bindings = enclosing

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        start = self.bindings
        self.bindings = start.copy()
        for statement in node.body:
            self.visit(statement)
        first = self.bindings
        self.bindings = start.copy()
        for statement in node.orelse:
            self.visit(statement)
        start.keep_shared(first, self.bindings)
        self.bindings = start

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        start = self.bindings
        self.bindings = start.copy()
        self.bind(node.target, None)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)
        start.keep_shared(start, self.bindings)
        self.bindings = start

    visit_AsyncFor = visit_For
