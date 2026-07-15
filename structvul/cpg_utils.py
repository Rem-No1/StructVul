# # cpg_utils.py
import os
import subprocess
import tempfile
import re

# If the joern command is not in PATH, set this to an absolute path such as "/opt/joern/joern-parse".
JOERN_PARSE_CMD = "joern-parse" 
JOERN_EXPORT_CMD = "joern-export"

def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
    """
    Extract CPG information with Joern and convert it into text suitable for embedding.
    This extracts summaries of the AST structure and CDG/DDG flows.
    """
    if not code or len(code.strip()) == 0:
        return ""

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 1. Write the code to a temporary file.
    code_path = os.path.join(temp_dir, "source.c")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    cpg_bin_path = os.path.join(temp_dir, "cpg.bin")
    
    try:
        # 2. Call joern-parse to generate the binary graph.
        # Suppress output to keep the console clean.
        subprocess.run(
            [JOERN_PARSE_CMD, code_path, "--out", cpg_bin_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 3. Export the graph structure as dot, including AST, CFG, and related graphs.
        # --repr ast extracts the abstract syntax tree, where structural features are clearest.
        # --repr cpg14 provides fuller information, but can produce too many tokens.
        out_dot_path = os.path.join(temp_dir, "out.dot")
        subprocess.run(
            [JOERN_EXPORT_CMD, cpg_bin_path, "--repr", "ast", "--out", out_dot_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # 4. Read the dot output and simplify it.
        # Extract node types and edges while removing line-number noise.
        graph_text = ""
        if os.path.exists(out_dot_path):
            with open(out_dot_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple regex cleanup to extract labels and structural relationships.
                # For example: "CALL" -> "IDENTIFIER".
                matches = re.findall(r'label\s*=\s*"([^"]+)"', content)
                # Serialize graph nodes into linear text while preserving structural semantics.
                graph_text = " -> ".join(matches[:200]) # Limit length to avoid token overflow.

        return f"Graph Structure: {graph_text}\nRaw Code:\n{code}"

    except Exception as e:
        # If Joern fails, fall back to raw code so the pipeline does not crash.
        # print(f"⚠️ Joern extraction failed: {e}. Using raw code.")
        return code 
    finally:
        # Clean up temporary files.
        if os.path.exists(code_path): os.remove(code_path)
        if os.path.exists(cpg_bin_path): os.remove(cpg_bin_path)
        # Dot export is usually a directory.
        if os.path.exists(out_dot_path): 
            import shutil
            shutil.rmtree(out_dot_path, ignore_errors=True)

# Test helper.
if __name__ == "__main__":
    code = "int main() { char buf[10]; strcpy(buf, input); }"
    print(extract_cpg_structure(code))


# In environments without Joern, comment out Joern code and use tree-sitter for structure extraction.
# import tree_sitter
# import tree_sitter_c

# def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
#     """
#     High-precision fix: resolves an infinite-loop bug during AST traversal.
#     """
#     if not code or len(code.strip()) == 0:
#         return ""

#     try:
#         c_lang = tree_sitter.Language(tree_sitter_c.language())
#         parser = tree_sitter.Parser(c_lang)
#         tree = parser.parse(bytes(code, "utf8"))
        
        
#         structure_tokens = []
#         cursor = tree.walk()
        
#         # Track whether traversal just returned from a child node.
#         visited_children = False
        
#         while True:
#             # 1. Node extraction logic, unchanged.
#             if not visited_children and cursor.node.is_named:
#                 node_type = cursor.node.type
#                 if node_type in ["identifier", "type_identifier", "field_identifier", "call_expression"]:
#                     node_text = code[cursor.node.start_byte : cursor.node.end_byte]
#                     # Clean up newlines.
#                     node_text = node_text.replace("\n", "").strip()
#                     structure_tokens.append(f"{node_type}:{node_text}")
#                 else:
#                     structure_tokens.append(node_type)

#             # 2. Cursor movement logic, which previously caused the infinite loop.
#             # -------------------------------------------------
#             # Key fix: only try to enter child nodes when NOT visited_children.
#             if not visited_children and cursor.goto_first_child():
#                 visited_children = False
            
#             # If traversal cannot go down, try the next sibling.
#             elif cursor.goto_next_sibling():
#                 visited_children = False
            
#             # If traversal cannot go right, go up to the parent.
#             elif cursor.goto_parent():
#                 visited_children = True
            
#             # If no movement is possible, traversal is complete.
#             else:
#                 break
#             # -------------------------------------------------
        
#         graph_text = " -> ".join(structure_tokens[:300])
#         return f"Graph Structure (High Precision): {graph_text}\nRaw Code:\n{code}"

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         print(f"⚠️ Tree-sitter extraction failed: {e}. Using raw code.")
#         return f"Raw Code:\n{code}"



# # def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
# #     return code



# # --- Validation test ---
# if __name__ == "__main__":
#     test_code = """
#     int main() {
#         char buf[10];
#         if (check_user()) {
#             strcpy(buf, input);
#         }
#         return 0;
#     }
#     """
#     print("Start extraction...")
#     result = extract_cpg_structure(test_code)
#     print("Extraction complete!")
#     print(result)
