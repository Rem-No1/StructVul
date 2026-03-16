# # # cpg_utils.py
# import os
# import subprocess
# import tempfile
# import re

# # 如果 joern 命令不在环境变量中，请修改这里为绝对路径，例如 "/opt/joern/joern-parse"
# JOERN_PARSE_CMD = "joern-parse" 
# JOERN_EXPORT_CMD = "joern-export"

# def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
#     """
#     使用 Joern 提取代码的 CPG 信息，并转化为一种适合 Embedding 的文本格式。
#     这里我们提取 AST (结构) 和 CDG/DDG (流) 的摘要。
#     """
#     if not code or len(code.strip()) == 0:
#         return ""

#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     # 1. 将代码写入临时文件
#     code_path = os.path.join(temp_dir, "source.c")
#     with open(code_path, "w", encoding="utf-8") as f:
#         f.write(code)

#     cpg_bin_path = os.path.join(temp_dir, "cpg.bin")
    
#     try:
#         # 2. 调用 joern-parse 生成二进制图
#         # 抑制输出以保持控制台整洁
#         subprocess.run(
#             [JOERN_PARSE_CMD, code_path, "--out", cpg_bin_path],
#             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )

#         # 3. 导出图结构 (这里导出为 dot 格式，包含 AST, CFG 等)
#         # --repr ast 提取抽象语法树，结构特征最明显
#         # 你也可以尝试 --repr cpg14 获得更全的信息，但 token 会爆炸
#         out_dot_path = os.path.join(temp_dir, "out.dot")
#         subprocess.run(
#             [JOERN_EXPORT_CMD, cpg_bin_path, "--repr", "ast", "--out", out_dot_path],
#             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
        
#         # 4. 读取 dot 文件并进行简化处理
#         # 主要是提取节点类型和边，去除具体的行号噪音
#         graph_text = ""
#         if os.path.exists(out_dot_path):
#             with open(out_dot_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 # 简单的正则清洗，提取 Label 和 结构关系
#                 # 例如: "CALL" -> "IDENTIFIER"
#                 matches = re.findall(r'label\s*=\s*"([^"]+)"', content)
#                 # 将图节点序列化为线性文本，保留结构语义
#                 graph_text = " -> ".join(matches[:200]) # 限制长度，防止 Token 溢出

#         return f"Graph Structure: {graph_text}\nRaw Code:\n{code}"

#     except Exception as e:
#         # 如果 Joern 调用失败（没安装），为了代码不崩，回退到使用原始代码
#         # print(f"⚠️ Joern extraction failed: {e}. Using raw code.")
#         return code 
#     finally:
#         # 清理临时文件
#         if os.path.exists(code_path): os.remove(code_path)
#         if os.path.exists(cpg_bin_path): os.remove(cpg_bin_path)
#         # dot 导出通常是一个目录
#         if os.path.exists(out_dot_path): 
#             import shutil
#             shutil.rmtree(out_dot_path, ignore_errors=True)

# # 测试用
# if __name__ == "__main__":
#     code = "int main() { char buf[10]; strcpy(buf, input); }"
#     print(extract_cpg_structure(code))


import tree_sitter
import tree_sitter_c

def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
    """
    [高精度修复版] 修复了遍历 AST 时的死循环 Bug。
    """
    if not code or len(code.strip()) == 0:
        return ""

    try:
        c_lang = tree_sitter.Language(tree_sitter_c.language())
        parser = tree_sitter.Parser(c_lang)
        tree = parser.parse(bytes(code, "utf8"))
        
        
        structure_tokens = []
        cursor = tree.walk()
        
        # 标记状态：是否刚刚从子节点回溯上来
        visited_children = False
        
        while True:
            # 1. 提取节点信息的逻辑 (保持不变)
            if not visited_children and cursor.node.is_named:
                node_type = cursor.node.type
                if node_type in ["identifier", "type_identifier", "field_identifier", "call_expression"]:
                    node_text = code[cursor.node.start_byte : cursor.node.end_byte]
                    # 简单清洗一下换行符
                    node_text = node_text.replace("\n", "").strip()
                    structure_tokens.append(f"{node_type}:{node_text}")
                else:
                    structure_tokens.append(node_type)

            # 2. 光标移动逻辑 (❌ 之前死循环的地方)
            # -------------------------------------------------
            # 关键修复：只有当 NOT visited_children 时，才尝试进入子节点
            if not visited_children and cursor.goto_first_child():
                visited_children = False
            
            # 如果不能向下，就尝试向右（兄弟节点）
            elif cursor.goto_next_sibling():
                visited_children = False
            
            # 如果不能向右，就尝试向上（回溯父节点）
            elif cursor.goto_parent():
                visited_children = True
            
            # 如果都不能，说明遍历结束
            else:
                break
            # -------------------------------------------------
        
        graph_text = " -> ".join(structure_tokens[:300])
        return f"Graph Structure (High Precision): {graph_text}\nRaw Code:\n{code}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Tree-sitter extraction failed: {e}. Using raw code.")
        return f"Raw Code:\n{code}"



# def extract_cpg_structure(code: str, temp_dir: str = ".tmp_cpg") -> str:
#     return code



# --- 验证测试 ---
if __name__ == "__main__":
    test_code = """
    int main() {
        char buf[10];
        if (check_user()) {
            strcpy(buf, input);
        }
        return 0;
    }
    """
    print("开始提取...")
    result = extract_cpg_structure(test_code)
    print("提取完成！")
    print(result)