import ast
import copy
import keyword
import os
from typing import Callable, Dict, Iterable, List, Sequence

import libcst as cst

from name_obfuscator import AdvancedNameObfuscator


STRICT_MUTATION_ENV = "ROBUST_TRANSFORM_STRICT"


def _preserve_trailing_newline(original: str, transformed: str) -> str:
    if original.endswith("\n") and not transformed.endswith("\n"):
        return transformed + "\n"
    return transformed


def _apply_python_ast_mutation(
    code: str,
    mutator: Callable[[ast.Module], bool],
) -> str:
    try:
        tree = ast.parse(code)
        changed = mutator(tree)
        if not changed:
            return code
        ast.fix_missing_locations(tree)
        transformed = ast.unparse(tree)
        return _preserve_trailing_newline(code, transformed) if transformed.strip() else code
    except Exception:
        if os.getenv(STRICT_MUTATION_ENV) == "1":
            raise
        return code


def _apply_python_cst_mutation(
    code: str,
    transformer_factory: Callable[[], cst.CSTTransformer],
    changed_attr: str = "counter",
) -> str:
    try:
        module = cst.parse_module(code)
        transformer = transformer_factory()
        updated_module = module.visit(transformer)
        changed = bool(getattr(transformer, changed_attr, 0))
        if not changed:
            return code
        transformed = updated_module.code
        return _preserve_trailing_newline(code, transformed) if transformed.strip() else code
    except Exception:
        if os.getenv(STRICT_MUTATION_ENV) == "1":
            raise
        return code


def _parameter_names(args: ast.arguments) -> List[str]:
    names: List[str] = []
    names.extend(arg.arg for arg in args.posonlyargs)
    names.extend(arg.arg for arg in args.args)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _collect_target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _collect_target_names(elt)
        return
    if isinstance(target, ast.Starred):
        yield from _collect_target_names(target.value)


class _FunctionLocalCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.locals: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.locals.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self.locals.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _eligible_python_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        if not name:
            continue
        if keyword.iskeyword(name):
            continue
        if name in {"self", "cls"}:
            continue
        if name.startswith("__"):
            continue
        if name.isupper():
            continue
        out.append(name)
    return out


def _fresh_python_name(
    obfuscator: AdvancedNameObfuscator,
    reserved: set[str],
) -> str:
    while True:
        candidate = obfuscator.get_name()
        if not candidate.isidentifier():
            continue
        if keyword.iskeyword(candidate):
            continue
        if candidate in reserved:
            continue
        reserved.add(candidate)
        return candidate


class _ScopeRenameTransformer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str]) -> None:
        self.rename_map = rename_map

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.rename_map:
            node.arg = self.rename_map[node.arg]
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if isinstance(node.name, str) and node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        return self.generic_visit(node)


class _PythonFunctionRenamer(ast.NodeTransformer):
    def __init__(self, target: str, rng) -> None:
        self.target = target
        self.rng = rng
        self.changed = False
        self.obfuscator = AdvancedNameObfuscator(mode="chaos", lang="python", rng=self.rng)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._process_function(node)

    def _process_function(self, node):
        node = copy.deepcopy(node)
        param_names = _parameter_names(node.args)
        collector = _FunctionLocalCollector()
        for stmt in node.body:
            collector.visit(stmt)

        local_names = collector.locals - set(param_names) - collector.globals - collector.nonlocals

        target_names: List[str] = []
        if self.target in {"params", "all"}:
            target_names.extend(param_names)
        if self.target in {"locals", "all"}:
            target_names.extend(sorted(local_names))

        eligible = _eligible_python_names(target_names)
        if eligible:
            pick_count = self.rng.randint(1, min(4, len(eligible)))
            chosen = self.rng.sample(eligible, pick_count)
            reserved = set(target_names)
            rename_map = {name: _fresh_python_name(self.obfuscator, reserved) for name in chosen}
            scope_transformer = _ScopeRenameTransformer(rename_map)
            node.args = scope_transformer.visit(node.args)
            node.body = [scope_transformer.visit(stmt) for stmt in node.body]
            self.changed = True

        node.decorator_list = [self.visit(dec) for dec in node.decorator_list]
        if getattr(node, "returns", None) is not None:
            node.returns = self.visit(node.returns)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node


def obfuscate_local_variables(code: str, rng) -> str:
    return _apply_python_ast_mutation(
        code,
        _RenameMutation("locals", rng),
    )


def obfuscate_parameters(code: str, rng) -> str:
    return _apply_python_ast_mutation(
        code,
        _RenameMutation("params", rng),
    )


def obfuscate_local_identifiers(code: str, rng) -> str:
    return _apply_python_ast_mutation(
        code,
        _RenameMutation("all", rng),
    )


class _RenameMutation:
    def __init__(self, target: str, rng) -> None:
        self.target = target
        self.rng = rng

    def __call__(self, tree: ast.Module) -> bool:
        renamer = _PythonFunctionRenamer(self.target, self.rng)
        updated = renamer.visit(tree)
        tree.body = updated.body
        tree.type_ignores = updated.type_ignores
        return renamer.changed


class _SwapSymmetricComparisonsTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison):
        if len(updated_node.comparisons) != 1:
            return updated_node
        if self.rng.random() >= 0.5:
            return updated_node

        target = updated_node.comparisons[0]
        if not isinstance(target.operator, (cst.Equal, cst.NotEqual)):
            return updated_node

        self.counter += 1
        return updated_node.with_changes(
            left=target.comparator,
            comparisons=[target.with_changes(comparator=updated_node.left)],
        )


def swap_symmetric_comparisons(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _SwapSymmetricComparisonsTransformer(rng))


class _AppendIdentityConditionTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0

    def _wrap_test(self, test: cst.BaseExpression) -> cst.BaseExpression:
        if self.rng.random() < 0.5:
            operator = cst.Or(
                whitespace_before=cst.SimpleWhitespace(" "),
                whitespace_after=cst.SimpleWhitespace(" "),
            )
            right = cst.Name("False")
        else:
            operator = cst.And(
                whitespace_before=cst.SimpleWhitespace(" "),
                whitespace_after=cst.SimpleWhitespace(" "),
            )
            right = cst.Name("True")

        self.counter += 1
        return cst.BooleanOperation(
            left=test,
            operator=operator,
            right=right,
            lpar=[cst.LeftParen()],
            rpar=[cst.RightParen()],
        )

    def leave_If(self, original_node: cst.If, updated_node: cst.If):
        return updated_node.with_changes(test=self._wrap_test(updated_node.test))

    def leave_While(self, original_node: cst.While, updated_node: cst.While):
        return updated_node.with_changes(test=self._wrap_test(updated_node.test))


def append_identity_condition(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _AppendIdentityConditionTransformer(rng))


class _WrapNumericIdentityTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0

    def _wrap(self, node):
        strategy = self.rng.choice(["add_zero", "sub_zero", "mul_one"])
        if strategy == "add_zero":
            operator = cst.Add()
            right = cst.Integer("0")
        elif strategy == "sub_zero":
            operator = cst.Subtract()
            right = cst.Integer("0")
        else:
            operator = cst.Multiply()
            right = cst.Integer("1")

        self.counter += 1
        return cst.BinaryOperation(
            left=node,
            operator=operator,
            right=right,
            lpar=[cst.LeftParen()],
            rpar=[cst.RightParen()],
        )

    def leave_Integer(self, original_node: cst.Integer, updated_node: cst.Integer):
        return self._wrap(updated_node)

    def leave_Float(self, original_node: cst.Float, updated_node: cst.Float):
        return self._wrap(updated_node)


def wrap_numeric_literals_identity(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _WrapNumericIdentityTransformer(rng))


class _WrapNumericParenthesesTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.counter = 0

    def _wrap(self, node):
        self.counter += 1
        return node.with_changes(
            lpar=node.lpar + (cst.LeftParen(),),
            rpar=node.rpar + (cst.RightParen(),),
        )

    def leave_Integer(self, original_node: cst.Integer, updated_node: cst.Integer):
        return self._wrap(updated_node)

    def leave_Float(self, original_node: cst.Float, updated_node: cst.Float):
        return self._wrap(updated_node)


def wrap_numeric_literals_parentheses(code: str, rng) -> str:
    del rng
    return _apply_python_cst_mutation(code, _WrapNumericParenthesesTransformer)


class _ForToWhileTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.counter = 0

    def leave_For(self, original_node: cst.For, updated_node: cst.For):
        self.counter += 1
        iter_name = f"_iterator_{self.counter}"
        init_stmt = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(iter_name))],
                    value=cst.Call(func=cst.Name("iter"), args=[cst.Arg(value=updated_node.iter)]),
                )
            ]
        )

        try_stmt = cst.Try(
            body=cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[
                            cst.Assign(
                                targets=[cst.AssignTarget(target=updated_node.target)],
                                value=cst.Call(func=cst.Name("next"), args=[cst.Arg(value=cst.Name(iter_name))]),
                            )
                        ]
                    )
                ]
            ),
            handlers=[
                cst.ExceptHandler(
                    type=cst.Name("StopIteration"),
                    body=cst.IndentedBlock(
                        body=list(updated_node.orelse.body.body) + [cst.SimpleStatementLine(body=[cst.Break()])]
                        if updated_node.orelse is not None
                        else [cst.SimpleStatementLine(body=[cst.Break()])]
                    ),
                )
            ],
        )

        while_stmt = cst.While(
            test=cst.Name("True"),
            body=cst.IndentedBlock(body=[try_stmt] + list(updated_node.body.body)),
            whitespace_after_while=cst.SimpleWhitespace(" "),
        )
        return cst.FlattenSentinel([init_stmt, while_stmt])


def convert_for_to_while(code: str, rng) -> str:
    del rng
    return _apply_python_cst_mutation(code, _ForToWhileTransformer)


class _StatementWrappingTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        if self.rng.random() >= 0.5:
            return updated_node
        self.counter += 1
        wrapper_stmt = cst.If(
            test=cst.Name("True"),
            body=updated_node.body,
            whitespace_before_test=cst.SimpleWhitespace(" "),
        )
        return updated_node.with_changes(body=cst.IndentedBlock(body=[wrapper_stmt]))


def wrap_statements_if_true(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _StatementWrappingTransformer(rng))


_DEAD_CODE_TEMPLATES = [
    """
_debug_mode = True
if _debug_mode:
    print(f"Debug: initializing module with params {locals()}")
""",
    """
_temp_x = 1024
_temp_y = 256
_temp_res = (_temp_x ** 2) + (_temp_y // 2)
_temp_x = _temp_res % 10
""",
    """
_data_cache = [x * 2 for x in range(10) if x % 2 == 0]
_data_cache.append(999)
_data_cache.clear()
""",
    """
try:
    raise ValueError("Invalid configuration state")
except ValueError:
    pass
""",
]


class _DeadCodeInjectionTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0

    def _complex_body(self) -> cst.IndentedBlock:
        code = self.rng.choice(_DEAD_CODE_TEMPLATES).strip()
        return cst.IndentedBlock(body=cst.parse_module(code).body)

    def _dead_stmt(self) -> cst.BaseStatement:
        mode = self.rng.choice(["if_false", "while_false", "if_impossible"])
        body = self._complex_body()
        if mode == "if_false":
            return cst.If(
                test=cst.Name("False"),
                body=body,
                whitespace_before_test=cst.SimpleWhitespace(" "),
            )
        if mode == "while_false":
            return cst.While(
                test=cst.Name("False"),
                body=body,
                whitespace_after_while=cst.SimpleWhitespace(" "),
            )
        return cst.If(
            test=cst.Comparison(
                left=cst.Integer("0"),
                comparisons=[
                    cst.ComparisonTarget(operator=cst.GreaterThan(), comparator=cst.Integer("1"))
                ],
            ),
            body=body,
            whitespace_before_test=cst.SimpleWhitespace(" "),
        )

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        self.counter += 1
        new_body = [self._dead_stmt()] + list(updated_node.body.body)
        return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))


def inject_dead_code(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _DeadCodeInjectionTransformer(rng))


class _ExtractCallArgumentLiteralsTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0
        self.in_safe_context = False
        self.pending_assignments: List[tuple[str, cst.BaseExpression]] = []
        self.obfuscator = AdvancedNameObfuscator(mode="chaos", lang="python", rng=self.rng)

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool:
        self.in_safe_context = True
        return True

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine):
        self.in_safe_context = False
        if not self.pending_assignments:
            return updated_node

        new_statements = []
        for var_name, value_node in self.pending_assignments:
            new_statements.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[
                                cst.AssignTarget(
                                    target=cst.Name(var_name),
                                    whitespace_before_equal=cst.SimpleWhitespace(" "),
                                    whitespace_after_equal=cst.SimpleWhitespace(" "),
                                )
                            ],
                            value=value_node,
                        )
                    ]
                )
            )

        self.pending_assignments = []
        return cst.FlattenSentinel(new_statements + [updated_node])

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call):
        if not self.in_safe_context:
            return updated_node

        new_args = []
        modified = False
        for arg in updated_node.args:
            if isinstance(arg.value, (cst.SimpleString, cst.Integer, cst.Float)):
                self.counter += 1
                new_name = _fresh_python_name(self.obfuscator, set())
                self.pending_assignments.append((new_name, arg.value))
                new_args.append(arg.with_changes(value=cst.Name(new_name)))
                modified = True
            else:
                new_args.append(arg)

        return updated_node.with_changes(args=new_args) if modified else updated_node


def extract_call_argument_literals(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _ExtractCallArgumentLiteralsTransformer(rng))


class _SplitAssignmentTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.counter = 0

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine):
        new_body = []
        modified = False

        for stmt in updated_node.body:
            if isinstance(stmt, cst.AnnAssign) and stmt.value is not None and isinstance(stmt.target, cst.Name):
                self.counter += 1
                modified = True
                new_body.append(stmt.with_changes(value=None, equal=None))
                new_body.append(
                    cst.Assign(
                        targets=[
                            cst.AssignTarget(
                                target=stmt.target,
                                whitespace_before_equal=cst.SimpleWhitespace(" "),
                                whitespace_after_equal=cst.SimpleWhitespace(" "),
                            )
                        ],
                        value=stmt.value,
                    )
                )
                continue

            if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0].target, cst.Name):
                self.counter += 1
                modified = True
                new_body.append(
                    cst.Assign(
                        targets=[
                            cst.AssignTarget(
                                target=stmt.targets[0].target,
                                whitespace_before_equal=cst.SimpleWhitespace(" "),
                                whitespace_after_equal=cst.SimpleWhitespace(" "),
                            )
                        ],
                        value=cst.Name("None"),
                    )
                )
                new_body.append(stmt)
                continue

            new_body.append(stmt)

        if not modified:
            return updated_node
        return cst.FlattenSentinel([cst.SimpleStatementLine(body=[stmt]) for stmt in new_body])


def split_declaration_and_initialization(code: str, rng) -> str:
    del rng
    return _apply_python_cst_mutation(code, _SplitAssignmentTransformer)


class _EncapsulateLiteralsTransformer(cst.CSTTransformer):
    def __init__(self, rng) -> None:
        self.rng = rng
        self.counter = 0
        self.generated_functions: List[cst.FunctionDef] = []

    def _should_transform(self, node: cst.CSTNode) -> bool:
        return isinstance(node, (cst.SimpleString, cst.Integer, cst.Float))

    def _helper_name(self) -> cst.Name:
        self.counter += 1
        return cst.Name(f"_get_const_{self.counter}")

    def _make_helper(self, literal_node: cst.CSTNode) -> cst.Name:
        func_name = self._helper_name()
        self.generated_functions.append(
            cst.FunctionDef(
                name=func_name,
                params=cst.Parameters(),
                body=cst.IndentedBlock(
                    body=[cst.SimpleStatementLine(body=[cst.Return(value=literal_node)])]
                ),
            )
        )
        return func_name

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call):
        new_args = []
        modified = False
        for arg in updated_node.args:
            if self._should_transform(arg.value):
                helper_call = cst.Call(func=self._make_helper(arg.value))
                new_args.append(arg.with_changes(value=helper_call))
                modified = True
            else:
                new_args.append(arg)
        return updated_node.with_changes(args=new_args) if modified else updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign):
        if self._should_transform(updated_node.value):
            return updated_node.with_changes(value=cst.Call(func=self._make_helper(updated_node.value)))
        return updated_node

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison):
        left = updated_node.left
        if self._should_transform(left):
            left = cst.Call(func=self._make_helper(left))

        comparisons = []
        for comp in updated_node.comparisons:
            if self._should_transform(comp.comparator):
                comparisons.append(comp.with_changes(comparator=cst.Call(func=self._make_helper(comp.comparator))))
            else:
                comparisons.append(comp)
        return updated_node.with_changes(left=left, comparisons=comparisons)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module):
        if not self.generated_functions:
            return updated_node
        return updated_node.with_changes(body=list(self.generated_functions) + list(updated_node.body))


def encapsulate_literals_with_helpers(code: str, rng) -> str:
    return _apply_python_cst_mutation(code, lambda: _EncapsulateLiteralsTransformer(rng))


LEXICAL_TRANSFORMS: Dict[str, Callable[[str, object], str]] = {
    "obfuscate_local_variables": obfuscate_local_variables,
    "obfuscate_parameters": obfuscate_parameters,
    "obfuscate_local_identifiers": obfuscate_local_identifiers,
}


EXPRESSION_TRANSFORMS: Dict[str, Callable[[str, object], str]] = {
    "swap_symmetric_comparisons": swap_symmetric_comparisons,
    "append_identity_condition": append_identity_condition,
    "wrap_numeric_literals_identity": wrap_numeric_literals_identity,
    "wrap_numeric_literals_parentheses": wrap_numeric_literals_parentheses,
}


CONTROL_FLOW_TRANSFORMS: Dict[str, Callable[[str, object], str]] = {
    "convert_for_to_while": convert_for_to_while,
    "wrap_statements_if_true": wrap_statements_if_true,
    "inject_dead_code": inject_dead_code,
}


DATAFLOW_TRANSFORMS: Dict[str, Callable[[str, object], str]] = {
    "extract_call_argument_literals": extract_call_argument_literals,
    "split_declaration_and_initialization": split_declaration_and_initialization,
    "encapsulate_literals_with_helpers": encapsulate_literals_with_helpers,
}


ALL_TRANSFORMS: Dict[str, Callable[[str, object], str]] = {}
ALL_TRANSFORMS.update(LEXICAL_TRANSFORMS)
ALL_TRANSFORMS.update(EXPRESSION_TRANSFORMS)
ALL_TRANSFORMS.update(CONTROL_FLOW_TRANSFORMS)
ALL_TRANSFORMS.update(DATAFLOW_TRANSFORMS)
