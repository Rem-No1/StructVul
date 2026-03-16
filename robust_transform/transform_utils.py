import argparse
import copy
import json
import os
import random
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET


NS = {
    "src": "http://www.srcML.org/srcML/src",
    "cpp": "http://www.srcML.org/srcML/cpp",
}

ET.register_namespace("", NS["src"])
ET.register_namespace("cpp", NS["cpp"])


STRICT_MUTATION_ENV = "ROBUST_TRANSFORM_STRICT"


def src_tag(name: str) -> str:
    return f"{{{NS['src']}}}{name}"


def cpp_tag(name: str) -> str:
    return f"{{{NS['cpp']}}}{name}"


def local_name(tag_or_elem) -> str:
    tag = tag_or_elem.tag if hasattr(tag_or_elem, "tag") else str(tag_or_elem)
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def ancestors(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Iterable[ET.Element]:
    current = parent_map.get(node)
    while current is not None:
        yield current
        current = parent_map.get(current)


def has_ancestor(
    node: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
    names: Sequence[str],
    *,
    namespace: str | None = None,
) -> bool:
    name_set = set(names)
    for ancestor_node in ancestors(node, parent_map):
        if namespace is not None:
            if not str(ancestor_node.tag).startswith(f"{{{namespace}}}"):
                continue
        if local_name(ancestor_node) in name_set:
            return True
    return False


def previous_sibling(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> ET.Element | None:
    parent = parent_map.get(node)
    if parent is None:
        return None
    children = list(parent)
    idx = children.index(node)
    if idx == 0:
        return None
    return children[idx - 1]


def next_sibling(node: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> ET.Element | None:
    parent = parent_map.get(node)
    if parent is None:
        return None
    children = list(parent)
    idx = children.index(node)
    if idx + 1 >= len(children):
        return None
    return children[idx + 1]


def insert_before(parent: ET.Element, ref_node: ET.Element, new_node: ET.Element) -> None:
    children = list(parent)
    parent.insert(children.index(ref_node), new_node)


def insert_after(parent: ET.Element, ref_node: ET.Element, new_node: ET.Element) -> None:
    children = list(parent)
    parent.insert(children.index(ref_node) + 1, new_node)


def replace_child(parent: ET.Element, old_node: ET.Element, new_node: ET.Element) -> None:
    children = list(parent)
    idx = children.index(old_node)
    parent.remove(old_node)
    parent.insert(idx, new_node)


def direct_child(node: ET.Element, child_name: str) -> ET.Element | None:
    wanted = src_tag(child_name)
    for child in list(node):
        if child.tag == wanted:
            return child
    return None


def get_decl_name_node(decl_node: ET.Element) -> ET.Element | None:
    for child in list(decl_node):
        if child.tag == src_tag("name"):
            return child
    return None


def first_function_name(function_node: ET.Element) -> str:
    for child in list(function_node):
        if child.tag == src_tag("name"):
            return child.text or ""
    return ""


def node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def copy_node(node: ET.Element) -> ET.Element:
    return copy.deepcopy(node)


def clear_tail(node: ET.Element) -> ET.Element:
    node.tail = None
    return node


def make_name_node(value: str, tail: str | None = None) -> ET.Element:
    node = ET.Element(src_tag("name"))
    node.text = value
    node.tail = tail
    return node


def make_literal_node(value: str, literal_type: str | None = None, tail: str | None = None) -> ET.Element:
    node = ET.Element(src_tag("literal"))
    node.text = value
    if literal_type:
        node.set("type", literal_type)
    node.tail = tail
    return node


def make_call_node(func_name: str, tail: str | None = None) -> ET.Element:
    call = ET.Element(src_tag("call"))
    name_node = ET.SubElement(call, src_tag("name"))
    name_node.text = func_name
    args = ET.SubElement(call, src_tag("argument_list"))
    args.text = "()"
    call.tail = tail
    return call


def make_expr_stmt(expr_node: ET.Element, tail: str | None = "\n") -> ET.Element:
    stmt = ET.Element(src_tag("expr_stmt"))
    stmt.append(expr_node)
    expr_node.tail = ";"
    stmt.tail = tail
    return stmt


def make_return_stmt(expr_node: ET.Element, tail: str | None = "\n") -> ET.Element:
    ret = ET.Element(src_tag("return"))
    ret.text = "return "
    ret.append(expr_node)
    expr_node.tail = ";"
    ret.tail = tail
    return ret


def make_decl_stmt(
    decl_type: str,
    decl_name: str,
    init_expr: ET.Element | None = None,
    tail: str | None = "\n",
) -> ET.Element:
    decl_stmt = ET.Element(src_tag("decl_stmt"))
    decl = ET.SubElement(decl_stmt, src_tag("decl"))
    type_node = ET.SubElement(decl, src_tag("type"))
    type_name = ET.SubElement(type_node, src_tag("name"))
    type_name.text = decl_type
    type_name.tail = " "

    name_node = ET.SubElement(decl, src_tag("name"))
    name_node.text = decl_name

    if init_expr is not None:
        name_node.tail = " "
        init = ET.SubElement(decl, src_tag("init"))
        init.text = "= "
        init.append(init_expr)

    decl.tail = ";"
    decl_stmt.tail = tail
    return decl_stmt


def make_condition(expr_node: ET.Element) -> ET.Element:
    condition = ET.Element(src_tag("condition"))
    condition.text = "("
    condition.append(expr_node)
    expr_node.tail = ")"
    return condition


def make_block(statements: Sequence[ET.Element], tail: str | None = None) -> ET.Element:
    block = ET.Element(src_tag("block"))
    block.text = "{"
    block_content = ET.SubElement(block, src_tag("block_content"))
    for stmt in statements:
        block_content.append(stmt)
    block_content.tail = "}"
    block.tail = tail
    return block


def make_if_stmt(condition_expr: ET.Element, statements: Sequence[ET.Element], tail: str | None = "\n") -> ET.Element:
    if_stmt = ET.Element(src_tag("if_stmt"))
    if_node = ET.SubElement(if_stmt, src_tag("if"))
    if_node.text = "if"
    if_node.append(make_condition(condition_expr))
    if_node.append(make_block(statements))
    if_stmt.tail = tail
    return if_stmt


def make_while_stmt(condition_expr: ET.Element, statements: Sequence[ET.Element], tail: str | None = None) -> ET.Element:
    while_node = ET.Element(src_tag("while"))
    while_node.text = "while"
    while_node.append(make_condition(condition_expr))
    while_node.append(make_block(statements))
    while_node.tail = tail
    return while_node


def _temp_suffix_for_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized in {"c++", "cpp", "cxx"}:
        return ".cpp"
    return ".c"


def code_to_srcml_root(code: str, language: str = "C") -> ET.Element:
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_path = Path(tmp_dir) / f"snippet{_temp_suffix_for_language(language)}"
        code_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(["srcml", str(code_path)], capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return ET.fromstring(proc.stdout)


def srcml_root_to_code(root: ET.Element) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        xml_path = Path(tmp_dir) / "snippet.xml"
        xml_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        proc = subprocess.run(["srcml", str(xml_path), "-S"], capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return proc.stdout.decode("utf-8")


def apply_srcml_mutation(
    code: str,
    rng: random.Random,
    mutator: Callable[[ET.Element, random.Random], bool],
    language: str = "C",
) -> str:
    try:
        root = code_to_srcml_root(code, language=language)
        changed = mutator(root, rng)
        if not changed:
            return code
        transformed = srcml_root_to_code(root)
        return transformed if transformed.strip() else code
    except Exception:
        if os.getenv(STRICT_MUTATION_ENV) == "1":
            raise
        return code


def next_numeric_id(data: List[dict]) -> int:
    max_id = -1
    for item in data:
        sid = item.get("id")
        if isinstance(sid, int):
            max_id = max(max_id, sid)
        elif isinstance(sid, str) and sid.isdigit():
            max_id = max(max_id, int(sid))
    return max_id + 1


def build_augmented_dataset(
    data: List[dict],
    variants_per_sample: int,
    transform_fn: Callable[[str, random.Random], Tuple[str, List[str]]],
    seed: int,
    include_original: bool,
    category: str,
) -> List[dict]:
    rng = random.Random(seed)
    out: List[dict] = []
    new_id = next_numeric_id(data)

    for item in data:
        code = item.get("code", "")

        if include_original:
            base = deepcopy(item)
            base["is_transformed"] = False
            base["source_id"] = item.get("id")
            base["transformations"] = []
            base["transform_category"] = category
            out.append(base)

        for i in range(variants_per_sample):
            try:
                transformed_code, applied = transform_fn(code, rng)
            except Exception:
                continue

            if transformed_code == code or not applied:
                continue

            new_item = deepcopy(item)
            new_item["id"] = new_id
            new_id += 1
            new_item["source_id"] = item.get("id")
            new_item["variant_index"] = i + 1
            new_item["is_transformed"] = True
            new_item["transform_category"] = category
            new_item["transformations"] = applied
            new_item["code"] = transformed_code
            out.append(new_item)

    return out


def transform_code_by_pool(
    code: str,
    enabled_transforms: Sequence[str],
    transform_pool: Dict[str, Callable[[str, random.Random], str]],
    rng: random.Random,
    max_transforms_per_variant: int,
) -> Tuple[str, List[str]]:
    if not enabled_transforms:
        return code, []

    attempts = 8
    for _ in range(attempts):
        selected_count = rng.randint(1, min(max_transforms_per_variant, len(enabled_transforms)))
        selected = rng.sample(list(enabled_transforms), selected_count)

        transformed = code
        applied: List[str] = []
        for name in selected:
            fn = transform_pool[name]
            new_code = fn(transformed, rng)
            if new_code != transformed:
                transformed = new_code
                applied.append(name)

        if transformed != code and applied:
            return transformed, applied

    return code, []


def load_dataset(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Input JSON root must be list, got {type(data).__name__}")
    return data


def save_dataset(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def default_parser(
    description: str,
    default_input: Path | None = None,
    default_output: Path | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        type=Path,
        required=default_input is None,
        default=default_input,
        help="Path to BaseCodeFilesReason.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=default_output is None,
        default=default_output,
        help="Output JSON path",
    )
    parser.add_argument("--variants-per-sample", type=int, default=2, help="Variants per input sample")
    parser.add_argument(
        "--max-transforms-per-variant",
        type=int,
        default=2,
        help="Maximum number of transforms composed in one variant",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--drop-original",
        dest="drop_original",
        action="store_true",
        default=False,
        help="Output transformed variants (and optional fallbacks), without copying base originals first.",
    )
    parser.add_argument(
        "--keep-original",
        dest="drop_original",
        action="store_false",
        help="Keep base originals in output before variants.",
    )
    return parser
