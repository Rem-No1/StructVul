# build_prompt.py
# Build prompts for code vulnerability prediction and LLM-as-a-judge.

"""
Prompt builders for:
1) Code prediction model (code LLM).
2) Judge model (LLM-as-a-judge).
"""

import json
from typing import Any, Dict, List, Tuple


PROMPT_VARIANT_ALIASES = {
    "baseline": "zero_shot_direct",
    "default": "zero_shot_direct",
    "zero_shot_direct": "zero_shot_direct",
    "zero_shot_cot": "zero_shot_cot",
    "zero_shot_cot_explicit": "zero_shot_cot_explicit",
    "zero_shot_cot_visible": "zero_shot_cot_explicit",
    "few_shot_direct": "few_shot_direct",
    "few_shot_cot": "few_shot_cot",
    "few_shot_cot_explicit": "few_shot_cot_explicit",
    "few_shot_cot_visible": "few_shot_cot_explicit",
    "zs_direct": "zero_shot_direct",
    "zs_cot": "zero_shot_cot",
    "zs_cot_explicit": "zero_shot_cot_explicit",
    "fs_direct": "few_shot_direct",
    "fs_cot": "few_shot_cot",
    "fs_cot_explicit": "few_shot_cot_explicit",
}


def get_supported_code_prompt_variants() -> Tuple[str, ...]:
    """Return supported prompt variants for code vulnerability prediction."""
    return (
        "zero_shot_direct",
        "zero_shot_cot",
        "zero_shot_cot_explicit",
        "few_shot_direct",
        "few_shot_cot",
        "few_shot_cot_explicit",
    )


def _normalize_prompt_variant(prompt_variant: str) -> str:
    normalized = PROMPT_VARIANT_ALIASES.get(prompt_variant.strip().lower())
    if normalized is None:
        supported = ", ".join(get_supported_code_prompt_variants())
        raise ValueError(
            f"Unsupported prompt_variant: {prompt_variant}. Supported variants: {supported}"
        )
    return normalized


def _build_prediction_system_prompt(cot_mode: str) -> str:
    system_prompt = (
        "You are a senior security code auditor. Your expertise is finding and analyzing "
        "security vulnerabilities in source code.\n\n"
        "Your tasks:\n"
        "1. Read the given code.\n"
        "2. Decide whether the code contains a security vulnerability under the given task definition.\n"
        "3. Provide a concise but informative reason in English.\n"
        "4. Output ONLY a single JSON object and nothing else (no extra text, no Markdown, no comments).\n\n"
        "Important:\n"
        "- Base your analysis on the code as given. Do not assume behaviors that are not implied by the code.\n"
        "- If you are uncertain, make the most reasonable decision given the evidence, but do not fabricate non-existent code.\n"
        "- Use exactly the JSON field names and formats specified below.\n"
    )
    if cot_mode == "silent":
        system_prompt += (
            "- Analyze the code with a private step-by-step checklist: attacker control, dangerous operations, "
            "missing protections, and exploitability.\n"
            "- Do not reveal your intermediate reasoning steps; think silently and only output the final JSON.\n"
        )
    elif cot_mode == "explicit":
        system_prompt += (
            "- Analyze the code with a step-by-step checklist: attacker control, dangerous operations, "
            "missing protections, and exploitability.\n"
            "- Output ONLY a single JSON object, and write the reasoning steps inside the `prediction_reason` field.\n"
        )
    else:
        system_prompt += (
            "- Do not reveal your intermediate reasoning steps; only output the final JSON.\n"
        )
    return system_prompt


def _build_prediction_task_description() -> str:
    return """
[TASK]
You must determine whether the given code contains a security vulnerability, and provide the vulnerability type and an English explanation.

Label definitions:
- is_vulnerable:
  - true means there exists at least one security vulnerability
  - false means no obvious vulnerability is found under common security standards
- vuln_type:
  - If is_vulnerable = true, provide a short vulnerability type name (e.g., "Buffer Overflow", "SQL Injection", "Hardcoded Secret", etc.)
  - If is_vulnerable = false, use "NONE"
- prediction_reason:
  - Provide an English explanation of about 50-150 words.
  - Try to include:
    (1) where the issue is (key variables/functions/operations);
    (2) the approximate triggering condition or attacker control;
    (3) why this is a vulnerability or why it appears safe.
""".strip()


def _build_prediction_output_format() -> str:
    return """
Return a single JSON object with EXACTLY this format:

{
  "is_vulnerable": true or false,
  "vuln_type": "string",
  "prediction_reason": "string"
}

Notes:
- Do not add any extra fields.
- Do not output anything outside the JSON.
- Use lowercase true/false for booleans (valid JSON).
""".strip()


def _build_zero_shot_instruction(cot_mode: str) -> str:
    if cot_mode == "silent":
        return """
[REASONING INSTRUCTION]
Reason step by step internally before answering:
1. Identify attacker-controlled inputs or externally influenced data.
2. Identify dangerous operations, sensitive sinks, memory access, parsing, or command/query construction.
3. Check whether validation, bounds checks, sanitization, escaping, authorization, or safe APIs are present.
4. Decide whether an actual vulnerability exists in the shown code.

Keep the reasoning private. Output only the final JSON.
""".strip()
    if cot_mode == "explicit":
        return """
[REASONING INSTRUCTION]
Reason step by step and expose the reasoning inside JSON.
Use this structure inside `prediction_reason`:
- Step 1 (Input/Control): attacker-controlled input or external influence.
- Step 2 (Sink/Operation): dangerous sink or operation.
- Step 3 (Protection Check): validation, bounds check, sanitization, escaping, authorization, or safe API.
- Step 4 (Conclusion): vulnerable or safe, and why.

Output only one JSON object. Do not output anything outside JSON.
""".strip()
    return """
[ANSWER STYLE]
Answer directly based on the given code. Do not include any analysis outside the final JSON.
""".strip()


def _build_few_shot_examples(use_cot: bool) -> str:
    if use_cot:
        return """
[FEW-SHOT EXAMPLES]
Example 1
Language: Python
Code:
import os

def run(user_input):
    os.system("grep " + user_input + " /var/log/app.log")

Reasoning pattern:
- Attacker-controlled data: user_input
- Dangerous operation: shell command construction and execution
- Missing protection: no quoting, validation, or safe subprocess usage
- Decision: vulnerable

Expected JSON:
{
  "is_vulnerable": true,
  "vuln_type": "Command Injection",
  "prediction_reason": "User-controlled input is concatenated into a shell command and executed with os.system. An attacker can inject shell metacharacters to execute arbitrary commands. The code has no sanitization, quoting, or safer subprocess API, so the command construction is directly exploitable."
}

Example 2
Language: C
Code:
int copy_name(const char *src) {
    char buf[8];
    if (strlen(src) >= sizeof(buf)) return -1;
    strcpy(buf, src);
    return 0;
}

Reasoning pattern:
- Attacker-controlled data: src
- Dangerous operation: strcpy into fixed-size stack buffer
- Present protection: explicit length check rejects oversized input before copy
- Decision: not vulnerable based on shown code

Expected JSON:
{
  "is_vulnerable": false,
  "vuln_type": "NONE",
  "prediction_reason": "Although strcpy is used on a fixed-size stack buffer, the function first checks whether the input length is at least the buffer size and returns early for oversized data. Based on the shown path, attacker-controlled input that would overflow buf is blocked, so no obvious vulnerability is present."
}
""".strip()

    return """
[FEW-SHOT EXAMPLES]
Example 1
Language: Python
Code:
import os

def run(user_input):
    os.system("grep " + user_input + " /var/log/app.log")

Expected JSON:
{
  "is_vulnerable": true,
  "vuln_type": "Command Injection",
  "prediction_reason": "User-controlled input is concatenated into a shell command and executed with os.system. An attacker can inject shell metacharacters to execute arbitrary commands. The code has no sanitization, quoting, or safer subprocess API, so the command construction is directly exploitable."
}

Example 2
Language: C
Code:
int copy_name(const char *src) {
    char buf[8];
    if (strlen(src) >= sizeof(buf)) return -1;
    strcpy(buf, src);
    return 0;
}

Expected JSON:
{
  "is_vulnerable": false,
  "vuln_type": "NONE",
  "prediction_reason": "Although strcpy is used on a fixed-size stack buffer, the function first checks whether the input length is at least the buffer size and returns early for oversized data. Based on the shown path, attacker-controlled input that would overflow buf is blocked, so no obvious vulnerability is present."
}
""".strip()


def _build_few_shot_examples_from_dataset(
    few_shot_examples: List[Dict[str, Any]],
    use_cot: bool,
) -> str:
    sections = [
        "[FEW-SHOT EXAMPLES]",
        "The following examples come from the current dataset (first sample as requested).",
    ]

    for idx, sample in enumerate(few_shot_examples, start=1):
        example_language = str(sample.get("language", "Unknown"))
        example_code = str(sample.get("code", ""))
        example_reason = str(sample.get("reason", ""))
        example_label = int(sample.get("label", 0))
        example_vuln_type = str(sample.get("vuln_type", "UNKNOWN"))

        if example_label == 0:
            example_vuln_type = "NONE"

        if len(example_code) > 1800:
            example_code = example_code[:1800] + "\n... (truncated)"
        if len(example_reason) > 500:
            example_reason = example_reason[:500] + "..."
        vuln_type_json = json.dumps(example_vuln_type, ensure_ascii=False)
        reason_json = json.dumps(example_reason, ensure_ascii=False)

        block = f"""Example {idx}
Language: {example_language}
Code:
{example_code}
"""
        if use_cot:
            block += f"""
Reasoning pattern:
- Use the dataset annotation as reference evidence.
- Focus on attacker control, dangerous operation, existing protections, and final conclusion.
- Reference explanation: {example_reason}
"""

        block += f"""
Expected JSON:
{{
  "is_vulnerable": {"true" if example_label == 1 else "false"},
  "vuln_type": {vuln_type_json},
  "prediction_reason": {reason_json}
}}
"""
        sections.append(block.strip())

    return "\n\n".join(sections).strip()


def build_code_prediction_prompts(
    code: str,
    language: str,
    prompt_variant: str = "zero_shot_direct",
    few_shot_examples: List[Dict[str, Any]] | None = None,
) -> Tuple[str, str]:
    """
    Build system and user prompts for the code prediction model.

    Supported prompt variants:
    - zero_shot_direct
    - zero_shot_cot
    - zero_shot_cot_explicit
    - few_shot_direct
    - few_shot_cot
    - few_shot_cot_explicit

    The model is expected to output a JSON object:
    {
      "is_vulnerable": true/false,
      "vuln_type": "string",
      "prediction_reason": "string"
    }
    """
    normalized_variant = _normalize_prompt_variant(prompt_variant)
    if normalized_variant.endswith("_cot_explicit"):
        cot_mode = "explicit"
    elif normalized_variant.endswith("_cot"):
        cot_mode = "silent"
    else:
        cot_mode = "none"

    use_few_shot = normalized_variant.startswith("few_shot")

    system_prompt = _build_prediction_system_prompt(cot_mode=cot_mode)

    sections = [_build_prediction_task_description()]
    if use_few_shot:
        if few_shot_examples:
            sections.append(
                _build_few_shot_examples_from_dataset(
                    few_shot_examples=few_shot_examples,
                    use_cot=(cot_mode != "none"),
                )
            )
        else:
            sections.append(_build_few_shot_examples(use_cot=(cot_mode != "none")))
    sections.append(_build_zero_shot_instruction(cot_mode=cot_mode))
    sections.append(
        f"""
[PROMPT VARIANT]
{normalized_variant}

[LANGUAGE]
{language}

[CODE START]
{code}
[CODE END]
""".strip()
    )
    sections.append(_build_prediction_output_format())
    user_prompt = "\n\n".join(sections)

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
    Build system and user prompts for the judge model.

    gt_label: 1 for vulnerable, 0 for non-vulnerable.
    """
    gt_is_vulnerable = bool(gt_label)
    gt_is_vul_str = str(gt_is_vulnerable).lower()
    model_is_vul_str = str(model_is_vul).lower()

    system_prompt = (
        "You are a rigorous security code review judge. Your job is to evaluate whether another "
        "model's vulnerability analysis is correct.\n\n"
        "You will receive:\n"
        "1) the code itself;\n"
        "2) the ground-truth label for this sample;\n"
        "3) the dataset-provided reference reason (around 100 words);\n"
        "4) the evaluated model's predicted label;\n"
        "5) the evaluated model's generated reason.\n\n"
        "Your goals:\n"
        "- Determine whether the model's predicted label is correct (prediction_correct).\n"
        "- Determine whether the model's reason/explanation is correct (reason_correct), i.e., whether it "
        "accurately explains the vulnerability mechanism or the safety rationale.\n\n"
        "Evaluation principles (very important):\n"
        "1. Base all judgments on the CODE and the GROUND TRUTH + REFERENCE REASON.\n"
        "2. 'Reason correct' does NOT require matching wording. It requires covering key points and avoiding critical errors.\n"
        "3. If the model's reason contradicts the code or the ground truth, it must be judged incorrect.\n"
        "4. Stay objective and strict. Do not reward verbosity, fancy wording, or length.\n"
        "5. Output ONLY a single JSON object and nothing else.\n\n"
        "Reason-correctness checklist:\n"
        "- If the ground truth is 'vulnerable', a correct reason typically:\n"
        "  a) identifies the approximate location or key variables/functions/operations involved;\n"
        "  b) describes the triggering condition or attacker-controlled input path;\n"
        "  c) explains the vulnerability mechanism (e.g., out-of-bounds access, missing validation, unsafe string concatenation leading to injection, etc.);\n"
        "  d) does not contradict the code or the ground truth.\n"
        "- If the ground truth is 'non-vulnerable', a correct reason typically:\n"
        "  a) does not fabricate dangerous operations that do not exist in the code;\n"
        "  b) explains why the implementation appears safe (e.g., proper bounds checks, safe APIs, correct validation);\n"
        "  c) does not contradict the code or the ground truth.\n\n"
        "Use the following definition for prediction_correct:\n"
        "- prediction_correct:\n"
        "  - Set to 1 if the model's `is_vulnerable` exactly matches the ground-truth label "
        "(`label = 1` -> `is_vulnerable = true`, `label = 0` -> `is_vulnerable = false`).\n"
        "  - Otherwise set to 0.\n"
        "If the model's reason covers most key mechanisms correctly and contains no critical contradictions, set reason_correct = 1.\n"
        "If the reason is overly generic or clearly misunderstands the mechanism or code logic, set reason_correct = 0.\n"
    )

    user_prompt = f"""
[LANGUAGE]
{language}

[CODE START]
{code}
[CODE END]

[GROUND TRUTH]
is_vulnerable: {gt_is_vul_str}

[REFERENCE REASON] (from dataset annotation)
{reference_reason}

[MODEL PREDICTION]
is_vulnerable: {model_is_vul_str}
vuln_type: {model_vuln_type}

[MODEL REASON]
{model_reason}

Based on the information above, output ONLY a single JSON object in EXACTLY this format:

{{
  "prediction_correct": 0 or 1,
  "reason_correct": 0 or 1,
  "missing_points": ["list of strings; key points missing from the model reason; can be empty"],
  "wrong_points": ["list of strings; critical errors or misleading statements in the model reason; can be empty"]
}}

Requirements:
- prediction_correct:
  - Set to 1 if the model's `is_vulnerable` exactly matches the ground-truth label (`label = 1` -> `is_vulnerable = true`, `label = 0` -> `is_vulnerable = false`).
  - Otherwise set to 0.
- reason_correct:
  - Set to 1 if the model reason is largely correct about mechanism, trigger conditions, and key locations, and does not contradict the ground truth and code.
  - Set to 0 if the model reason clearly misunderstands the vulnerability mechanism or code logic, or conflicts with the ground truth/code.
- missing_points:
  - Briefly list important points that appear in the reference reason and/or code but are missing from the model reason.
- wrong_points:
  - Briefly list clearly wrong or misleading statements in the model reason.

Output ONLY JSON. Do not output any explanations outside the JSON.
""".strip()

    return system_prompt, user_prompt
