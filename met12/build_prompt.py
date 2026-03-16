# base/build_prompt.py
from typing import Tuple, List, Dict, Any

# def build_code_prediction_prompts(code: str, language: str, similar_cases: List[Dict[str, Any]] = None) -> Tuple[str, str]:
#     """
#     构建 RAG + CoT (思维链) 的 Prompt。
#     核心思想：要求模型利用检索到的案例，进行 step-by-step 的对比分析。
#     """
    
#     # 1. 构建 RAG 参考信息文本
#     rag_context_str = ""
#     if similar_cases and len(similar_cases) > 0:
#         rag_context_str = "\n[KNOWLEDGE BASE: SIMILAR HISTORICAL CASES]\nThe following cases are retrieved because they share code structures with the target. Use them as the 'Gold Standard' for your reasoning.\n"
        
#         for idx, case in enumerate(similar_cases):
#             label_str = "VULNERABLE" if str(case['is_vulnerable']) == '1' else "SAFE"
#             vuln_type = case.get('vuln_type', 'Unknown')
            
#             # 代码截断，保留核心逻辑
#             ref_code = case['code']
#             if len(ref_code) > 600: ref_code = ref_code[:600] + "\n... (truncated)"
            
#             # 原因截断
#             ref_reason = case['reason']
#             if len(ref_reason) > 400: ref_reason = ref_reason[:400] + "..."
            
#             rag_context_str += f"""
# --- Reference Case {idx+1} ({label_str}) ---
# Vulnerability Type: {vuln_type}
# Code Snippet:
# {ref_code}

# Expert Analysis:
# {ref_reason}
# -----------------------
# """
#     else:
#         rag_context_str = "\n[KNOWLEDGE BASE]\nNo similar cases found. Rely on general security knowledge.\n"

#     # 2. System Prompt: 设定专家人设，并强制 CoT 模式
#     system_prompt = (
#         "You are an expert Security Code Auditor implementing a 'Comparative Analysis' methodology.\n"
#         "Your goal is to detect vulnerabilities by comparing the target code against known historical cases.\n\n"
#         "Your Reasoning Process (Chain of Thought):\n"
#         "1. Analyze the Target Code: Identify sources (user inputs) and sinks (sensitive operations).\n"
#         "2. Compare with References: For each provided Reference Case, ask:\n"
#         "   - Does the target code share the same vulnerability pattern?\n"
#         "   - Does the target code implement the sanitization/fix shown in a SAFE reference?\n"
#         "3. Synthesize: Based on the comparison, determine if the target is vulnerable.\n\n"
#         "Output Requirement:\n"
#         "- Return ONLY a single JSON object.\n"
#         "- The 'prediction_reason' field MUST contain your step-by-step comparison logic."
#     )

#     # 3. User Prompt: 具体的指令
#     user_prompt = f"""
# [TASK]
# Analyze the provided code for security vulnerabilities.

# [LANGUAGE]
# {language}

# {rag_context_str}

# [TARGET CODE]
# {code}
# [CODE END]

# [INSTRUCTION]
# Think step-by-step.
# 1. First, identify if there is any data flow from external input to a dangerous function (SQL, Command, Memory, etc.).
# 2. Second, look at the [REFERENCE CASES] above. 
#    - If the target looks like a VULNERABLE reference, explain the similarity.
#    - If the target looks like a SAFE reference (e.g., has checks), explain why it is safe.
# 3. Finally, generate the JSON output.

# Return a single JSON object with EXACTLY this format:

# {{
#   "is_vulnerable": true or false,
#   "vuln_type": "string (e.g., 'SQL Injection', 'Buffer Overflow', or 'NONE')",
#   "prediction_reason": "string. IMPORTANT: Write your step-by-step comparison analysis here. Start with 'Analysis: ...', then 'Comparison with Ref 1: ...', then 'Conclusion: ...'"
# }}
# """.strip()

#     return system_prompt, user_prompt


# base/build_prompt.py

def build_code_prediction_prompts(code: str, language: str, similar_cases: list = None) -> tuple[str, str]:
    # 1. RAG 上下文 
    rag_context_str = ""
    if similar_cases:
        rag_context_str = "\n[REFERENCE CASES]\n(Use these ONLY if they are relevant. If they describe a different logic, IGNORE them.)\n"
        for idx, case in enumerate(similar_cases):
            
            ref_code = case['code']
            ref_reason = case.get('reason', '')
            rag_context_str += f"--- Case {idx+1} ({'VULNERABLE' if str(case['is_vulnerable'])=='1' else 'SAFE'}) ---\nCode:\n{ref_code}\nAnalysis:\n{ref_reason}\n\n"

    # 2. System Prompt
    system_prompt = (
        "You are a security expert. "
        "Your goal is to accurately detect vulnerabilities."
    )

    # 3. User Prompt (核心修改：分离思考与输出)
    user_prompt = f"""
[TASK]
Analyze the code for security vulnerabilities.

[LANGUAGE]
{language}

{rag_context_str}

[TARGET CODE]
{code}
[CODE END]

[INSTRUCTION]
1. **Thinking Process**: 
   - First, enclose your analysis inside <thinking> tags.
   - Check if the code receives external input.
   - Trace the data flow to sensitive functions (sinks).
   - Check if the retrieved [REFERENCE CASES] are relevant. If yes, compare them. If no, analyze independently.
   - Verify if any sanitization or validation exists.
   
2. **Final Output**:
   - After the </thinking> tag, output a SINGLE JSON object containing the final verdict.

[FORMAT]
<thinking>
... write your step-by-step analysis here ...
</thinking>

{{
  "is_vulnerable": true/false,
  "vuln_type": "string",
  "prediction_reason": "Summarize your <thinking> into a short concise paragraph (max 100 words)."
}}
""".strip()

    return system_prompt, user_prompt


def build_judge_prompts(
    code: str,
    language: str,
    gt_label: int,
    reference_reason: str,
    model_is_vul: bool,
    model_vuln_type: str,
    model_reason: str,
) -> Tuple[str, str]:
    """
    Judge Prompt 保持不变。
    因为前面强制模型在 model_reason 里写了 CoT，这里的 Judge 将会看到非常详尽的推理，有助于提升判分准确率。
    """
    gt_is_vulnerable = bool(gt_label)
    gt_is_vul_str = str(gt_is_vulnerable).lower()
    model_is_vul_str = str(model_is_vul).lower()

    system_prompt = (
        "You are a rigorous security code review judge. Your job is to evaluate whether another "
        "model’s vulnerability analysis is correct.\n"
        "Output ONLY a single JSON object."
    )

    user_prompt = f"""
[LANGUAGE]
{language}

[CODE START]
{code}
[CODE END]

[GROUND TRUTH]
is_vulnerable: {gt_is_vul_str}

[REFERENCE REASON]
{reference_reason}

[MODEL PREDICTION]
is_vulnerable: {model_is_vul_str}
vuln_type: {model_vuln_type}

[MODEL REASONING (Chain of Thought)]
{model_reason}

Based on the information above, output ONLY a single JSON object in EXACTLY this format:

{{
  "prediction_correct": 0 or 1,
  "reason_correct": 0 or 1,
  "missing_points": ["list of strings"],
  "wrong_points": ["list of strings"]
}}

Criteria:
- prediction_correct: 1 if model label matches ground truth, else 0.
- reason_correct: 1 if the model's reasoning logic is sound and matches the code reality. The model uses RAG-based comparison; if the comparison is logical and leads to the right conclusion, mark it correct.
""".strip()

    return system_prompt, user_prompt