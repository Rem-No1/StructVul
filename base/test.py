import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from base.build_prompt import build_code_prediction_prompts, build_judge_prompts

# ================= 配置区域 =================
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


MY_API_KEY = require_env("CODE_API_KEY")
BASE_URL = os.getenv("CODE_BASE_URL", "https://api.vectorengine.ai/v1")
CODE_MODEL_NAME = os.getenv("CODE_MODEL_NAME", "gpt-5.1-chat-latest")

# Supported values:
# - zero_shot_direct
# - zero_shot_cot
# - zero_shot_cot_explicit
# - few_shot_direct
# - few_shot_cot
# - few_shot_cot_explicit
# PROMPT_VARIANT = "zero_shot_cot"
PROMPT_VARIANT = "few_shot_cot"

# Runtime few-shot source for few-shot variants.
# Policy: use the first sample of the current DATASET_FILE.
DATASET_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = []

client = OpenAI(
    api_key=MY_API_KEY,
    base_url=BASE_URL
)


JUDGE_API_KEY = require_env("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "deepseek-chat")

client_jugde = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL
)


# 路径配置
# CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v2_control")
# DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason_control.json"
# OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_BaseCodeFilesReason_results_fc.json"

# CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v2_dataflow")
# DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason_dataflow.json"
# OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_BaseCodeFilesReason_results_fc.json"

# CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v2_expression")
# DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason_expression.json"
# OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_BaseCodeFilesReason_results_fc.json"

# CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v2_lexical")
# DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason_lexical.json"
# OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_BaseCodeFilesReason_results_fc.json"

CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v2")
DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason.json"
OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_BaseCodeFilesReason_results_fc.json"

# 并发配置：同时请求 API 的数量
# 注意：DeepSeek API 有速率限制（RPM/TPM），如果报错 429，请调小这个数字
MAX_WORKERS = 32  

# ================= 工具函数 =================

def extract_json_from_text(text: str) -> str:
    """鲁棒的 JSON 提取函数，处理 Markdown 代码块和非标准输出"""
    text = text.strip()
    # 尝试直接解析
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 尝试提取 ```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
            
    # 尝试提取最外层 {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
        
    return text

def call_code_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """调用代码分析模型"""
    try:
        response = client.chat.completions.create(
            model=CODE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"enable_thinking": False},# qwen3-32b 特有参数，关闭思考链
            temperature=0.0,
            timeout=60 # 设置超时防止卡死
        )
        content = response.choices[0].message.content or ""
        json_text = extract_json_from_text(content)
        data = json.loads(json_text)
    except Exception as e:
        # 出错时返回默认结构
        return {
            "is_vulnerable": False,
            "vuln_type": "PARSE_ERROR",
            "prediction_reason": f"API_ERROR: {str(e)}",
            "raw_output": str(e)
        }

    # 【关键逻辑修复】处理布尔值的各种情况
    raw_is_vul = data.get("is_vulnerable", False)
    if isinstance(raw_is_vul, str):
        is_vul = raw_is_vul.lower() in ["true", "yes", "1"]
    else:
        is_vul = bool(raw_is_vul)

    return {
        "is_vulnerable": is_vul,
        "vuln_type": str(data.get("vuln_type", "UNKNOWN")),
        "prediction_reason": str(data.get("prediction_reason", "")),
    }

def call_judge_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """调用判分模型"""
    try:
        response = client_jugde.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            timeout=60
        )
        content = response.choices[0].message.content or ""
        json_text = extract_json_from_text(content)
        data = json.loads(json_text)
    except Exception as e:
        return {
            "prediction_correct": 0,
            "reason_correct": 0,
            "missing_points": [f"API_ERROR: {str(e)}"],
            "wrong_points": [],
        }

    # 健壮性转换
    pc = data.get("prediction_correct", 0)
    if isinstance(pc, bool): pc = 1 if pc else 0
    
    rc = data.get("reason_correct", 0)
    if isinstance(rc, bool): rc = 1 if rc else 0

    return {
        "prediction_correct": int(pc),
        "reason_correct": int(rc),
        "missing_points": data.get("missing_points", []),
        "wrong_points": data.get("wrong_points", []),
    }

# ================= 核心处理逻辑 =================

def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个样本的完整流程：
    Code LLM -> Judge LLM -> 组装结果
    """
    # 1. 提取基础信息
    sample_id = sample.get("id", "unknown")
    # 原始数据集中的 'label' 是 int (0或1)
    gt_label = int(sample.get("label", 0))
    reference_reason = sample.get("reason", "")
    code = sample.get("code", "")
    language = sample.get("language", "C")

    # 2. 预测阶段
    code_sys, code_user = build_code_prediction_prompts(
        code=code,
        language=language,
        prompt_variant=PROMPT_VARIANT,
        few_shot_examples=DATASET_FEW_SHOT_EXAMPLES,
    )
    code_res = call_code_llm(code_sys, code_user)

    # 3. 判分阶段
    judge_sys, judge_user = build_judge_prompts(
        code=code,
        language=language,
        gt_label=gt_label,
        reference_reason=reference_reason,
        model_is_vul=code_res["is_vulnerable"],
        model_vuln_type=code_res["vuln_type"],
        model_reason=code_res["prediction_reason"],
    )
    judge_res = call_judge_llm(judge_sys, judge_user)

    # 4. 组装最终结果
    result_record = {
        "id": sample_id,
        "language": language,
        "prompt_variant": PROMPT_VARIANT,
        "code": code,
        "gt_label": gt_label,
        "reference_reason": reference_reason,
        # 预测结果
        "pred_is_vul": code_res["is_vulnerable"],
        "pred_vuln_type": code_res["vuln_type"],
        "pred_reason": code_res["prediction_reason"],
        # 判分结果
        "prediction_correct": judge_res["prediction_correct"],
        "reason_correct": judge_res["reason_correct"],
        "missing_points": judge_res["missing_points"],
        "wrong_points": judge_res["wrong_points"]
    }
    
    return result_record

# ================= 主程序 =================

def main():
    global DATASET_FEW_SHOT_EXAMPLES
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到输入文件: {DATASET_FILE}")
        return

    print(f"📂 正在加载数据集: {DATASET_FILE}")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pv = PROMPT_VARIANT.strip().lower()
    is_few_shot_variant = pv.startswith("few_shot") or pv.startswith("fs_")
    eval_dataset = dataset

    if is_few_shot_variant:
        if dataset:
            first_sample = dataset[0]
            DATASET_FEW_SHOT_EXAMPLES = [first_sample]
            print(f"[Few-shot Source] first dataset sample id={first_sample.get('id', 'unknown')}")
            eval_dataset = dataset[1:]
            print(f"[Few-shot Eval] skip first sample; evaluate from second sample. eval_count={len(eval_dataset)}")
        else:
            DATASET_FEW_SHOT_EXAMPLES = []
            eval_dataset = []
            print("[Few-shot Source] dataset is empty; fallback few-shot examples will be used.")
    else:
        DATASET_FEW_SHOT_EXAMPLES = []
        print("[Few-shot Source] prompt variant is not few-shot; disabled dataset few-shot injection.")

    total_samples = len(eval_dataset)
    print(f"[Prompt Variant] {PROMPT_VARIANT}")
    print(f"🚀 开始处理 {total_samples} 条数据，并发数: {MAX_WORKERS}")
    
    # 准备输出文件（清空并写入 JSON 数组开头）
    # 使用 'w' 模式清空旧文件
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write("[\n")  # 写入数组起始符

    # 线程锁，用于多线程同时写文件时防止冲突
    file_lock = threading.Lock()
    # 标记是否是第一条数据（用于控制逗号）
    is_first_entry = True

    # 进度条
    pbar = tqdm(total=total_samples, unit="sample")

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(process_single_sample, sample): sample 
            for sample in eval_dataset
        }

        # 获取完成的结果
        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                
                # 【线程安全写入】
                with file_lock:
                    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
                        if not is_first_entry:
                            f.write(",\n")  # 非第一条，前面加逗号
                        else:
                            is_first_entry = False # 标记第一条已写完
                        
                        json.dump(result, f, ensure_ascii=False, indent=4)
                        
            except Exception as e:
                print(f"\n❌ 处理样本时发生异常: {e}")
            
            pbar.update(1)

    # 所有任务完成后，闭合 JSON 数组
    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write("\n]")

    pbar.close()
    print(f"\n✅ 所有任务完成！结果已保存至: {OUTPUT_JSON}")
    # 验证一下生成的 JSON 是否合法
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            json.load(f)
        print("✅ 输出文件格式验证通过（Valid JSON）。")
    except Exception as e:
        print(f"⚠️ 警告：输出文件可能格式有误（通常是因为强制中断导致）：{e}")

if __name__ == "__main__":
    main()
