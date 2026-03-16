# test.py
# RAG + Cot+ cpg
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# 导入你的模块
from met12.build_prompt import build_code_prediction_prompts, build_judge_prompts
# 假设 rag_engine.py 就在同级目录下
from met12.rag_engine import VulnRAG 

# ================= 配置区域 =================
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


MY_API_KEY = require_env("CODE_API_KEY")
BASE_URL = os.getenv("CODE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CODE_MODEL_NAME = os.getenv("CODE_MODEL_NAME", "qwen3-32b")

JUDGE_API_KEY = require_env("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "deepseek-chat")

# 路径配置
CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/ddd/dataset_v1")
# CODE_ROOT = Path("D:/LIHAOZE/bishe/thecode/ddd/dataset_v2")
DATASET_FILE = CODE_ROOT / "BaseCodeFilesReason.json" # 既是测试集也是知识库源
OUTPUT_JSON = CODE_ROOT / f"{CODE_MODEL_NAME}_RAG_COT2_Results.json" # 改个名区分
RAG_DB_DIR = CODE_ROOT / "chroma_db_storage" 

# === 策略开关 ===
ENABLE_RAG = True
RAG_TOP_K = 3         # 检索 3 个最相似的
MAX_WORKERS =16       # 开启 RAG 后建议稍微调低并发，避免 Embedding API 拥塞

# 初始化 Clients
client = OpenAI(api_key=MY_API_KEY, base_url=BASE_URL)
client_judge = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

rag_engine = None # 全局变量

# ================= 工具函数 =================

def extract_json_from_text(text: str) -> str:
    """提取 JSON 字符串"""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except:
        pass
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1: return text[start:end].strip()
    if "{" in text:
        # 找最后一个 } 确保包含完整的 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start: return text[start : end + 1]
    return text

# def call_code_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
#     try:
#         response = client.chat.completions.create(
#             model=CODE_MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             # 注意：即便我们手动写了 CoT Prompt，这里设为 True 也无妨，
#             # 但 Qwen API 的 enable_thinking 通常是指显示内部思考过程。
#             # 为了完全控制 prompt 里的 CoT，我们设为 False，完全依赖 prompt 指令。
#             extra_body={"enable_thinking": False}, 
#             temperature=0.0, # 保持确定性
#             timeout=80       # CoT 输出较长，增加超时时间
#         )
#         content = response.choices[0].message.content or ""
#         json_text = extract_json_from_text(content)
#         data = json.loads(json_text)
#     except Exception as e:
#         return {
#             "is_vulnerable": False,
#             "vuln_type": "API_ERROR",
#             "prediction_reason": f"System Error: {str(e)}",
#             "raw_output": str(e)
#         }

#     # 鲁棒性处理
#     raw_is_vul = data.get("is_vulnerable", False)
#     if isinstance(raw_is_vul, str):
#         is_vul = raw_is_vul.lower() in ["true", "yes", "1"]
#     else:
#         is_vul = bool(raw_is_vul)

#     return {
#         "is_vulnerable": is_vul,
#         "vuln_type": str(data.get("vuln_type", "UNKNOWN")),
#         # 这里实际上会包含模型生成的长篇 CoT
#         "prediction_reason": str(data.get("prediction_reason", "")),
#     }


# test.py 中的工具函数更新

def call_code_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=CODE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # extra_body={"enable_thinking": False},
            temperature=0.0,
            timeout=80 
        )
        content = response.choices[0].message.content or ""
        
        # === 调试点：你可以打印 content 看看现在的思考过程 ===
        # print("RAW_OUTPUT:", content) 
        
        # 提取 JSON
        json_text = extract_json_from_text(content)
        data = json.loads(json_text)
        
        # 如果需要，可以将 <thinking> 内容提取出来存到 prediction_reason 里，方便 Judge 评判
        # 这样 Judge 就能看到完整的思维链，但 JSON 结构保持简洁
        thinking_content = ""
        if "<thinking>" in content and "</thinking>" in content:
            start = content.find("<thinking>") + 10
            end = content.find("</thinking>")
            thinking_content = content[start:end].strip()
        
        # 组合：Reason = 简短总结 + (可选：详细思考过程)
        final_reason = data.get("prediction_reason", "")
        # 如果你想让 Judge 看到思维链，可以拼接到这里
        # final_reason = f"Summary: {final_reason}\n\nDetailed Thought Process:\n{thinking_content}"
        
        data["prediction_reason"] = final_reason

    except Exception as e:
        return {
            "is_vulnerable": False,
            "vuln_type": "PARSE_ERROR",
            "prediction_reason": f"Error: {str(e)}",
            "raw_output": str(e) # 方便排查
        }

    # ... (后续布尔值处理逻辑不变)
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
    try:
        response = client_judge.chat.completions.create(
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
            "prediction_correct": 0, "reason_correct": 0,
            "missing_points": [f"Judge Error: {str(e)}"], "wrong_points": []
        }

    pc = data.get("prediction_correct", 0)
    rc = data.get("reason_correct", 0)
    return {
        "prediction_correct": 1 if pc else 0,
        "reason_correct": 1 if rc else 0,
        "missing_points": data.get("missing_points", []),
        "wrong_points": data.get("wrong_points", []),
    }

# ================= 核心处理逻辑 =================

def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(sample.get("id", "unknown"))
    gt_label = int(sample.get("label", 0))
    code = sample.get("code", "")
    language = sample.get("language", "C")
    
    # 1. RAG 检索 (Retrieve)
    similar_cases = []
    retrieved_ids = []
    
    if ENABLE_RAG and rag_engine:
        # 传入 exclude_id 防止泄露
        similar_cases = rag_engine.search(
            query_code=code,
            rerank_top_k=RAG_TOP_K,
            exclude_id=sample_id
        )
        # 记录一下检索到了哪些 ID，方便分析
        retrieved_ids = [c.get('id', 'N/A') for c in similar_cases]

    # 2. 预测 (Predict with CoT)
    # prompt 中会包含 retrieved cases 和 CoT 指令
    code_sys, code_user = build_code_prediction_prompts(
        code=code, 
        language=language, 
        similar_cases=similar_cases
    )
    code_res = call_code_llm(code_sys, code_user)

    # 3. 判分 (Judge)
    judge_sys, judge_user = build_judge_prompts(
        code=code,
        language=language,
        gt_label=gt_label,
        reference_reason=sample.get("reason", ""),
        model_is_vul=code_res["is_vulnerable"],
        model_vuln_type=code_res["vuln_type"],
        model_reason=code_res["prediction_reason"], # 这里面是 CoT 内容
    )
    judge_res = call_judge_llm(judge_sys, judge_user)

    # 4. 组装结果
    return {
        "id": sample_id,
        "code": code,
        "gt_label": gt_label,
        # RAG 信息
        "rag_retrieved_ids": retrieved_ids,
        "rag_context_count": len(similar_cases),
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

# ================= 主程序 =================

def main():
    global rag_engine
    
    # 1. 初始化 RAG
    if ENABLE_RAG:
        if not os.path.exists(DATASET_FILE):
            print(f"❌ 找不到数据文件: {DATASET_FILE}")
            return
            
        print("🔧 初始化 RAG 引擎 (Loading Knowledge Base)...")
        # 如果是第一次运行，会读取 DATASET_FILE 并调用 Embedding API
        # 如果是第二次运行，会直接加载 RAG_DB_DIR
        rag_engine = VulnRAG(
            dataset_path=str(DATASET_FILE), 
            persist_dir=str(RAG_DB_DIR)
        )

    # 2. 读取测试数据
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"🚀 开始 RAG+CoT 测试，共 {len(dataset)} 条样本...")

    # 3. 结果文件初始化
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write("[\n")

    file_lock = threading.Lock()
    is_first = True
    pbar = tqdm(total=len(dataset), unit="sample")

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sample = {executor.submit(process_single_sample, s): s for s in dataset}

        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                
                # 写入文件
                with file_lock:
                    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
                        if not is_first: f.write(",\n")
                        else: is_first = False
                        json.dump(result, f, ensure_ascii=False, indent=4)
                        
            except Exception as e:
                print(f"Error processing sample: {e}")
            
            pbar.update(1)

    # 结束
    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write("\n]")
    pbar.close()
    print(f"\n✅ 任务完成！RAG-CoT 结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
