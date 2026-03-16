import random
from typing import Callable, Dict, List
from xml.etree import ElementTree as ET

from name_obfuscator import AdvancedNameObfuscator
from transform_utils import (
    NS,
    ancestors,
    apply_srcml_mutation,
    build_parent_map,
    clear_tail,
    copy_node,
    direct_child,
    first_function_name,
    get_decl_name_node,
    has_ancestor,
    insert_after,
    insert_before,
    local_name,
    make_block,
    make_call_node,
    make_decl_stmt,
    make_expr_stmt,
    make_if_stmt,
    make_literal_node,
    make_name_node,
    make_return_stmt,
    make_while_stmt,
    next_sibling,
    node_text,
    previous_sibling,
    replace_child,
    src_tag,
)


C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "_Complex",
    "_Imaginary",
    "true",
    "false",
    "NULL",
}


def _is_cpp_ancestor(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> bool:
    for ancestor_node in ancestors(node, parent_map):
        if str(ancestor_node.tag).startswith(f"{{{NS['cpp']}}}"):
            return True
    return False


def _get_c_type(literal_type: str | None, value: str) -> str:
    if literal_type == "string":
        return "const char*"
    if literal_type == "char":
        return "char"
    if literal_type == "number":
        value_lower = value.lower()
        if "." in value or "e" in value_lower:
            return "double"
        if value_lower.endswith("l"):
            return "long long"
        return "int"
    return "int"


def _is_safe_rename_site(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> bool:
    if node.text is None:
        return False
    if _is_cpp_ancestor(node, parent_map):
        return False

    parent = parent_map.get(node)
    if parent is None:
        return False

    parent_name = local_name(parent)
    if parent_name in {"type", "label", "goto"}:
        return False
    if parent_name == "call":
        return False

    prev_node = previous_sibling(node, parent_map)
    if prev_node is not None and local_name(prev_node) == "operator":
        if (prev_node.text or "").strip() in {".", "->", "::"}:
            return False

    return True


def _global_reserved_names(root: ET.Element) -> set[str]:
    reserved: set[str] = set()
    for child in list(root):
        child_name = local_name(child)
        if child_name == "function":
            name = first_function_name(child)
            if name:
                reserved.add(name)
        elif child_name == "decl_stmt":
            for decl in child.findall("src:decl", NS):
                name_node = get_decl_name_node(decl)
                if name_node is not None and name_node.text:
                    reserved.add(name_node.text)
    return reserved


def _collect_identifier_candidates(function_node: ET.Element, target: str) -> List[ET.Element]:
    function_parent_map = build_parent_map(function_node)
    candidates: List[ET.Element] = []

    if target in {"params", "all"}:
        for param in function_node.findall(".//src:parameter", NS):
            decl = param.find(".//src:decl", NS)
            if decl is None:
                continue
            name_node = get_decl_name_node(decl)
            if name_node is not None:
                candidates.append(name_node)

    if target in {"locals", "all"}:
        for decl in function_node.findall(".//src:block//src:decl_stmt/src:decl", NS):
            name_node = get_decl_name_node(decl)
            if name_node is None:
                continue
            if has_ancestor(name_node, function_parent_map, ["parameter", "struct", "union", "enum"]):
                continue
            candidates.append(name_node)

    return candidates


def _rename_identifiers_mutation(root: ET.Element, rng: random.Random, target: str) -> bool:
    parent_map = build_parent_map(root)
    reserved = _global_reserved_names(root)
    changed = False

    for function_node in root.findall(".//src:function", NS):
        raw_candidates = _collect_identifier_candidates(function_node, target)
        if not raw_candidates:
            continue

        candidate_names = {
            (node.text or "")
            for node in raw_candidates
            if (node.text or "")
        }
        eligible = [
            name
            for name in candidate_names
            if name not in reserved
            and name not in C_KEYWORDS
            and not name.startswith("__")
            and not name.isupper()
            and name != "main"
        ]
        if not eligible:
            continue

        obfuscator = AdvancedNameObfuscator(mode="chaos", lang="c", rng=rng)
        pick_count = rng.randint(1, min(4, len(eligible)))
        chosen = rng.sample(eligible, pick_count)
        rename_map = {name: obfuscator.get_name() for name in chosen}

        for name_node in function_node.iter(src_tag("name")):
            current_text = name_node.text or ""
            if current_text not in rename_map:
                continue
            if not _is_safe_rename_site(name_node, parent_map):
                continue
            name_node.text = rename_map[current_text]
            changed = True

    return changed


def obfuscate_local_variables(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, lambda root, inner_rng: _rename_identifiers_mutation(root, inner_rng, "locals"))


def obfuscate_parameters(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, lambda root, inner_rng: _rename_identifiers_mutation(root, inner_rng, "params"))


def obfuscate_local_identifiers(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, lambda root, inner_rng: _rename_identifiers_mutation(root, inner_rng, "all"))


def _swap_symmetric_comparisons_mutation(root: ET.Element, rng: random.Random) -> bool:
    parent_map = build_parent_map(root)
    changed = False

    for operator_node in list(root.iter(src_tag("operator"))):
        op_text = (operator_node.text or "").strip()
        if op_text not in {"==", "!="}:
            continue

        parent = parent_map.get(operator_node)
        if parent is None or local_name(parent) != "expr":
            continue

        lhs = previous_sibling(operator_node, parent_map)
        rhs = next_sibling(operator_node, parent_map)
        if lhs is None or rhs is None:
            continue

        prev_of_lhs = previous_sibling(lhs, parent_map)
        if prev_of_lhs is not None:
            if local_name(prev_of_lhs) == "operator":
                continue
            if local_name(prev_of_lhs) == "cast":
                continue

        next_of_rhs = next_sibling(rhs, parent_map)
        if next_of_rhs is not None and local_name(next_of_rhs) in {"operator", "index"}:
            continue

        if local_name(lhs) not in {"name", "literal", "expr", "call"}:
            continue
        if local_name(rhs) not in {"name", "literal", "expr", "call"}:
            continue
        if rng.random() >= 0.5:
            continue

        lhs_copy = copy_node(lhs)
        rhs_copy = copy_node(rhs)
        lhs_copy.tail = rhs.tail
        rhs_copy.tail = lhs.tail or ""

        parent.remove(lhs)
        parent.remove(rhs)
        updated_children = list(parent)
        op_index = updated_children.index(operator_node)
        parent.insert(op_index, rhs_copy)
        parent.insert(op_index + 2, lhs_copy)
        operator_node.tail = " "
        changed = True

    return changed


def swap_symmetric_comparisons(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _swap_symmetric_comparisons_mutation)


def _append_identity_condition_mutation(root: ET.Element, rng: random.Random) -> bool:
    changed = False
    expr_nodes = root.findall(".//src:if_stmt/src:if/src:condition/src:expr", NS)
    expr_nodes.extend(root.findall(".//src:while/src:condition/src:expr", NS))

    for expr_node in expr_nodes:
        assignment_found = False
        for operator_node in expr_node.iter(src_tag("operator")):
            if (operator_node.text or "").strip() in {"=", "+=", "-=", "*=", "/=", "%="}:
                assignment_found = True
                break
        if assignment_found:
            continue

        choice = rng.choice([("||", "0"), ("&&", "1")])
        children = list(expr_node)
        if children:
            last_child = children[-1]
            last_child.tail = (last_child.tail or "") + " "

        new_operator = ET.Element(src_tag("operator"))
        new_operator.text = choice[0]
        new_operator.tail = " "
        expr_node.append(new_operator)
        expr_node.append(make_literal_node(choice[1], "number"))
        changed = True

    return changed


def append_identity_condition(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _append_identity_condition_mutation)


def _wrap_numeric_literals_identity_mutation(root: ET.Element, rng: random.Random) -> bool:
    parent_map = build_parent_map(root)
    targets: List[ET.Element] = []

    for literal_node in root.findall(".//src:literal[@type='number']", NS):
        if _is_cpp_ancestor(literal_node, parent_map):
            continue
        targets.append(literal_node)

    if not targets:
        return False

    changed = False
    for literal_node in targets:
        if rng.random() >= 0.75:
            continue

        parent = parent_map.get(literal_node)
        if parent is None:
            continue

        op_text = rng.choice(["+", "-", "*"])
        identity_operand = "1" if op_text == "*" else "0"

        expr_node = ET.Element(src_tag("expr"))
        expr_node.text = "("

        left_literal = copy_node(literal_node)
        left_literal.tail = " "
        expr_node.append(left_literal)

        operator_node = ET.Element(src_tag("operator"))
        operator_node.text = op_text
        operator_node.tail = " "
        expr_node.append(operator_node)

        right_literal = make_literal_node(identity_operand, "number", ")")
        expr_node.append(right_literal)
        expr_node.tail = literal_node.tail

        replace_child(parent, literal_node, expr_node)
        changed = True

    return changed


def wrap_numeric_literals_identity(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _wrap_numeric_literals_identity_mutation)


def _wrap_numeric_literals_parentheses_mutation(root: ET.Element, rng: random.Random) -> bool:
    parent_map = build_parent_map(root)
    targets: List[ET.Element] = []

    for literal_node in root.findall(".//src:literal[@type='number']", NS):
        if _is_cpp_ancestor(literal_node, parent_map):
            continue
        prev_node = previous_sibling(literal_node, parent_map)
        next_node = next_sibling(literal_node, parent_map)
        if prev_node is not None and next_node is not None:
            if local_name(prev_node) == "operator" and (prev_node.text or "").strip() == "(":
                if local_name(next_node) == "operator" and (next_node.text or "").strip() == ")":
                    continue
        targets.append(literal_node)

    changed = False
    for literal_node in targets:
        if rng.random() > 0.8:
            continue

        parent = parent_map.get(literal_node)
        if parent is None:
            continue

        expr_node = ET.Element(src_tag("expr"))
        expr_node.text = "("
        copied = copy_node(literal_node)
        copied.tail = ")"
        expr_node.append(copied)
        expr_node.tail = literal_node.tail
        replace_child(parent, literal_node, expr_node)
        changed = True

    return changed


def wrap_numeric_literals_parentheses(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _wrap_numeric_literals_parentheses_mutation)


def _build_init_statement(init_node: ET.Element) -> ET.Element | None:
    for child in list(init_node):
        child_name = local_name(child)
        if child_name == "expr":
            stmt = ET.Element(src_tag("expr_stmt"))
            copied = clear_tail(copy_node(child))
            stmt.append(copied)
            copied.tail = ";"
            stmt.tail = "\n"
            return stmt
        if child_name == "decl":
            stmt = ET.Element(src_tag("decl_stmt"))
            copied = clear_tail(copy_node(child))
            stmt.append(copied)
            copied.tail = ";"
            stmt.tail = "\n"
            return stmt
    return None


def _condition_expr_from_for(condition_node: ET.Element | None) -> ET.Element:
    if condition_node is not None:
        for child in list(condition_node):
            if local_name(child) == "expr":
                return clear_tail(copy_node(child))
    expr_node = ET.Element(src_tag("expr"))
    expr_node.append(make_literal_node("1", "number"))
    return expr_node


def _incr_statement(incr_node: ET.Element | None) -> ET.Element | None:
    if incr_node is None:
        return None
    for child in list(incr_node):
        if local_name(child) == "expr":
            stmt = ET.Element(src_tag("expr_stmt"))
            copied = clear_tail(copy_node(child))
            stmt.append(copied)
            copied.tail = ";"
            stmt.tail = "\n"
            return stmt
    return None


def _for_to_while_mutation(root: ET.Element, rng: random.Random) -> bool:
    del rng
    parent_map = build_parent_map(root)
    changed = False

    for for_node in root.findall(".//src:for", NS):
        if list(for_node.iter(src_tag("continue"))):
            continue

        control = direct_child(for_node, "control")
        if control is None:
            continue

        init_node = direct_child(control, "init")
        condition_node = direct_child(control, "condition")
        incr_node = direct_child(control, "incr")

        init_stmt = _build_init_statement(init_node) if init_node is not None else None
        condition_expr = _condition_expr_from_for(condition_node)
        incr_stmt = _incr_statement(incr_node)

        statements: List[ET.Element] = []
        if init_stmt is not None:
            statements.append(init_stmt)

        loop_body_stmts: List[ET.Element] = []
        body_block = direct_child(for_node, "block")
        if body_block is not None:
            body_content = direct_child(body_block, "block_content")
            if body_content is not None:
                for child in list(body_content):
                    loop_body_stmts.append(copy_node(child))
        else:
            for child in list(for_node):
                if local_name(child) != "control":
                    loop_body_stmts.append(copy_node(child))
                    break

        if incr_stmt is not None:
            loop_body_stmts.append(incr_stmt)

        while_node = make_while_stmt(condition_expr, loop_body_stmts, tail="\n")
        statements.append(while_node)

        outer_block = make_block(statements, tail=for_node.tail)
        parent = parent_map.get(for_node)
        if parent is None:
            continue
        replace_child(parent, for_node, outer_block)
        changed = True

    return changed


def convert_for_to_while(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _for_to_while_mutation)


def _wrap_statements_if_true_mutation(root: ET.Element, rng: random.Random) -> bool:
    targets: List[ET.Element] = []
    for block_content in root.findall(".//src:block_content", NS):
        for child in list(block_content):
            if local_name(child) in {"expr_stmt", "return", "break", "continue"} and rng.random() < 0.5:
                targets.append(child)

    if not targets:
        return False

    parent_map = build_parent_map(root)
    changed = False
    for stmt in targets:
        parent = parent_map.get(stmt)
        if parent is None:
            continue
        wrapped_stmt = copy_node(stmt)
        if wrapped_stmt.tail is None:
            wrapped_stmt.tail = "\n"
        if_stmt = make_if_stmt(make_literal_node("1", "number"), [wrapped_stmt], tail=stmt.tail or "\n")
        replace_child(parent, stmt, if_stmt)
        changed = True

    return changed


def wrap_statements_if_true(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _wrap_statements_if_true_mutation)


def _create_dead_code_if_stmt(counter: int) -> ET.Element:
    dead_var = f"dead_{counter}"
    decl_stmt = make_decl_stmt("int", dead_var, make_literal_node("100", "number"), tail="\n")

    expr_node = ET.Element(src_tag("expr"))
    expr_node.append(make_name_node(dead_var))
    op_node = ET.SubElement(expr_node, src_tag("operator"))
    op_node.text = "++"
    expr_stmt = make_expr_stmt(expr_node, tail="\n")
    return make_if_stmt(make_literal_node("0", "number"), [decl_stmt, expr_stmt], tail="\n")


def _inject_dead_code_mutation(root: ET.Element, rng: random.Random) -> bool:
    block_contents = root.findall(".//src:function//src:block/src:block_content", NS)
    if not block_contents:
        return False

    guaranteed_idx = rng.randrange(len(block_contents))
    changed = False
    counter = 0

    for idx, block_content in enumerate(block_contents):
        should_inject = idx == guaranteed_idx or rng.random() <= 0.3
        if not should_inject:
            continue

        counter += 1
        dead_code = _create_dead_code_if_stmt(counter)
        children = list(block_content)
        insert_at = rng.randint(0, len(children))
        block_content.insert(insert_at, dead_code)
        changed = True

    return changed


def inject_dead_code(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _inject_dead_code_mutation)


def _extract_call_argument_literals_mutation(root: ET.Element, rng: random.Random) -> bool:
    parent_map = build_parent_map(root)
    obfuscator = AdvancedNameObfuscator(mode="chaos", lang="c", rng=rng)
    modifications = []

    for literal_node in root.findall(".//src:call//src:argument//src:literal", NS):
        literal_value = literal_node.text or ""
        if not literal_value.strip():
            continue
        if has_ancestor(literal_node, parent_map, ["condition"]):
            continue

        stmt_node = None
        for ancestor_node in ancestors(literal_node, parent_map):
            if local_name(ancestor_node) in {"expr_stmt", "decl_stmt"}:
                stmt_node = ancestor_node
                break
        if stmt_node is None:
            continue
        if not _statement_supports_prefix_decl(stmt_node, parent_map):
            continue

        modifications.append(
            {
                "literal_node": literal_node,
                "stmt_node": stmt_node,
                "value": literal_value,
                "type": literal_node.get("type"),
            }
        )

    if not modifications:
        return False

    changed = False
    for mod in reversed(modifications):
        literal_node = mod["literal_node"]
        stmt_node = mod["stmt_node"]
        var_name = obfuscator.get_name()
        init_expr = ET.Element(src_tag("expr"))
        init_expr.append(make_literal_node(mod["value"], mod["type"]))
        decl_stmt = make_decl_stmt(_get_c_type(mod["type"], mod["value"]), var_name, init_expr, tail="\n")

        stmt_parent = parent_map.get(stmt_node)
        literal_parent = parent_map.get(literal_node)
        if stmt_parent is None or literal_parent is None:
            continue

        insert_before(stmt_parent, stmt_node, decl_stmt)
        replacement = make_name_node(var_name, literal_node.tail)
        replace_child(literal_parent, literal_node, replacement)
        changed = True

    return changed


def extract_call_argument_literals(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _extract_call_argument_literals_mutation)


def _statement_supports_prefix_decl(
    stmt_node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> bool:
    stmt_parent = parent_map.get(stmt_node)
    if stmt_parent is None or local_name(stmt_parent) != "block_content":
        return False

    block_node = parent_map.get(stmt_parent)
    if block_node is None or local_name(block_node) != "block":
        return False

    return block_node.get("type") != "pseudo"


def _split_decl_init_mutation(root: ET.Element, rng: random.Random) -> bool:
    del rng
    parent_map = build_parent_map(root)
    modifications = []

    for decl_stmt in root.findall(".//src:decl_stmt", NS):
        decls = decl_stmt.findall("src:decl", NS)
        if len(decls) != 1:
            continue

        decl = decls[0]
        parent_of_stmt = parent_map.get(decl_stmt)
        if parent_of_stmt is None:
            continue
        parent_tag = local_name(parent_of_stmt)
        if parent_tag in {"unit", "init", "condition"}:
            continue

        init_node = direct_child(decl, "init")
        type_node = direct_child(decl, "type")
        name_node = get_decl_name_node(decl)
        if init_node is None or type_node is None or name_node is None:
            continue

        type_text = node_text(type_node)
        if "const" in type_text or "static" in type_text or "auto" in type_text or "&" in type_text:
            continue

        expr_node = direct_child(init_node, "expr")
        if expr_node is None:
            continue

        modifications.append(
            {
                "decl": decl,
                "init": init_node,
                "expr": expr_node,
                "var_name": name_node.text or "",
                "decl_stmt": decl_stmt,
            }
        )

    if not modifications:
        return False

    changed = False
    for mod in reversed(modifications):
        decl = mod["decl"]
        init_node = mod["init"]
        expr_node = mod["expr"]
        decl_stmt = mod["decl_stmt"]
        stmt_parent = parent_map.get(decl_stmt)
        if stmt_parent is None:
            continue

        if init_node in list(decl):
            decl.remove(init_node)

        assignment_expr = ET.Element(src_tag("expr"))
        assignment_expr.append(make_name_node(mod["var_name"], " "))
        op_node = ET.SubElement(assignment_expr, src_tag("operator"))
        op_node.text = "="
        op_node.tail = " "

        copied_expr = clear_tail(copy_node(expr_node))
        if copied_expr.text:
            assignment_expr.text = copied_expr.text
        for child in list(copied_expr):
            assignment_expr.append(child)

        assignment_stmt = make_expr_stmt(assignment_expr, tail="\n")
        insert_after(stmt_parent, decl_stmt, assignment_stmt)
        changed = True

    return changed


def split_declaration_and_initialization(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _split_decl_init_mutation)


def _make_helper_function(func_name: str, return_type: str, literal_value: str, literal_type: str | None) -> ET.Element:
    function = ET.Element(src_tag("function"))
    type_node = ET.SubElement(function, src_tag("type"))
    type_name = ET.SubElement(type_node, src_tag("name"))
    type_name.text = return_type
    type_name.tail = " "

    name_node = ET.SubElement(function, src_tag("name"))
    name_node.text = func_name

    params = ET.SubElement(function, src_tag("parameter_list"))
    params.text = "()"

    expr = ET.Element(src_tag("expr"))
    expr.append(make_literal_node(literal_value, literal_type))
    body = make_block([make_return_stmt(expr, tail="\n")], tail=None)
    function.append(body)
    function.tail = "\n\n"
    return function


def _encapsulate_literals_with_helpers_mutation(root: ET.Element, rng: random.Random) -> bool:
    del rng
    parent_map = build_parent_map(root)
    replacements = []
    reserved_names = _global_reserved_names(root)
    helper_counter = 1

    for literal_node in root.findall(".//src:function//src:literal", NS):
        literal_value = literal_node.text or ""
        if not literal_value:
            continue
        if _is_cpp_ancestor(literal_node, parent_map):
            continue
        if has_ancestor(literal_node, parent_map, ["case", "index"]):
            continue

        replacements.append(
            {
                "node": literal_node,
                "value": literal_value,
                "type": literal_node.get("type"),
            }
        )

    if not replacements:
        return False

    helper_functions: List[ET.Element] = []
    changed = False

    for replacement in replacements:
        literal_node = replacement["node"]
        literal_parent = parent_map.get(literal_node)
        if literal_parent is None:
            continue

        while True:
            func_name = f"get_hardcode_{helper_counter}"
            helper_counter += 1
            if func_name not in reserved_names:
                reserved_names.add(func_name)
                break
        return_type = _get_c_type(replacement["type"], replacement["value"])
        helper_functions.append(
            _make_helper_function(func_name, return_type, replacement["value"], replacement["type"])
        )

        call_node = make_call_node(func_name, literal_node.tail)
        replace_child(literal_parent, literal_node, call_node)
        changed = True

    if not changed:
        return False

    insert_index = 0
    root_children = list(root)
    for idx, child in enumerate(root_children):
        if str(child.tag).startswith(f"{{{NS['cpp']}}}") or local_name(child) == "using":
            insert_index = idx + 1
            continue
        if local_name(child) in {"function", "struct", "class", "enum", "decl_stmt"}:
            if insert_index == 0:
                insert_index = idx
            break

    for helper_function in reversed(helper_functions):
        root.insert(insert_index, helper_function)

    return True


def encapsulate_literals_with_helpers(code: str, rng: random.Random) -> str:
    return apply_srcml_mutation(code, rng, _encapsulate_literals_with_helpers_mutation)


LEXICAL_TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {
    "obfuscate_local_variables": obfuscate_local_variables,
    "obfuscate_parameters": obfuscate_parameters,
    "obfuscate_local_identifiers": obfuscate_local_identifiers,
}


EXPRESSION_TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {
    "swap_symmetric_comparisons": swap_symmetric_comparisons,
    "append_identity_condition": append_identity_condition,
    "wrap_numeric_literals_identity": wrap_numeric_literals_identity,
    "wrap_numeric_literals_parentheses": wrap_numeric_literals_parentheses,
}


CONTROL_FLOW_TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {
    "convert_for_to_while": convert_for_to_while,
    "wrap_statements_if_true": wrap_statements_if_true,
    "inject_dead_code": inject_dead_code,
}


DATAFLOW_TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {
    "extract_call_argument_literals": extract_call_argument_literals,
    "split_declaration_and_initialization": split_declaration_and_initialization,
    "encapsulate_literals_with_helpers": encapsulate_literals_with_helpers,
}


ALL_TRANSFORMS: Dict[str, Callable[[str, random.Random], str]] = {}
ALL_TRANSFORMS.update(LEXICAL_TRANSFORMS)
ALL_TRANSFORMS.update(EXPRESSION_TRANSFORMS)
ALL_TRANSFORMS.update(CONTROL_FLOW_TRANSFORMS)
ALL_TRANSFORMS.update(DATAFLOW_TRANSFORMS)
