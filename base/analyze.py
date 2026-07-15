import json
import os
from pathlib import Path

# MODEL_NAME = "deepseek-chat"
MODEL_NAME = "qwen3-32b"
# MODEL_NAME = "gpt-5.1-chat-latest"


types="fc" # zd对应ZS-Direct, zc对应ZS-CoT, fd对应FS-Direct, fc对应FS-Direct
dataset_name_id = "1" # 1对应SV-TrustEval-C,2对应SecLLMHolmes
# ================= 配置 =================
# 这里填入上一频生成的 json 文件路径
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}/{MODEL_NAME}_BaseCodeFilesReason_results_{types}.json"

# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_dataflow/{MODEL_NAME}_BaseCodeFilesReason_results_{types}.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_control/{MODEL_NAME}_BaseCodeFilesReason_results_{types}.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_expression/{MODEL_NAME}_BaseCodeFilesReason_results_{types}.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_lexical/{MODEL_NAME}_BaseCodeFilesReason_results_{types}.json"


# RESULT_FILE = f"D:\\LIHAOZE\\bishe\\thecode\\eee\\dataset_v{dataset_name_id}\\{MODEL_NAME}_kfold_vanilla_rag\\vanilla_rag_seed1\\kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_dataflow/{MODEL_NAME}_kfold_vanilla_rag\\vanilla_rag_seed1\\kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_control/{MODEL_NAME}_kfold_vanilla_rag\\vanilla_rag_seed1\\kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_expression/{MODEL_NAME}_kfold_vanilla_rag\\vanilla_rag_seed1\\kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_lexical/{MODEL_NAME}_kfold_vanilla_rag\\vanilla_rag_seed1\\kfold_all_results.json"
# =======================================

# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}/{MODEL_NAME}_kfold_dual_system/structvul_seed1/kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_dataflow/{MODEL_NAME}_kfold_dual_system/structvul_seed1/kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_control/{MODEL_NAME}_kfold_dual_system/structvul_seed1/kfold_all_results.json"
# RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_expression/{MODEL_NAME}_kfold_dual_system/structvul_seed1/kfold_all_results.json"
RESULT_FILE = f"D:/LIHAOZE/bishe/thecode/eee/dataset_v{dataset_name_id}_lexical/{MODEL_NAME}_kfold_dual_system/structvul_seed1/kfold_all_results.json"




def safe_div(a, b):
    """安全除法，防止除以 0"""
    return a / b if b != 0 else 0.0


def compute_metrics_from_records(data):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    details = {
        1: {"corr_corr": 0, "corr_wrong": 0, "wrong_corr": 0, "wrong_wrong": 0},
        0: {"corr_corr": 0, "corr_wrong": 0, "wrong_corr": 0, "wrong_wrong": 0}
    }

    total = len(data)

    for item in data:
        gt = int(item.get("gt_label", item.get("label", 0)))
        pred_is_vul = bool(item.get("pred_is_vul", False))
        judge_pred_correct = int(item.get("prediction_correct", 0))
        judge_reason_correct = int(item.get("reason_correct", 0))

        if gt == 1 and pred_is_vul:
            tp += 1
        elif gt == 0 and not pred_is_vul:
            tn += 1
        elif gt == 0 and pred_is_vul:
            fp += 1
        elif gt == 1 and not pred_is_vul:
            fn += 1

        p_key = "corr" if judge_pred_correct == 1 else "wrong"
        r_key = "corr" if judge_reason_correct == 1 else "wrong"
        final_key = f"{p_key}_{r_key}"
        details[gt][final_key] += 1

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "details": details,
    }


def analyze_metrics(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return compute_metrics_from_records(data)

def analyze(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    print(f"📂 正在加载分析文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}")
        return

    # 1. 初始化计数器
    tp = 0  # 真阳性 (GT=1, Pred=1)
    tn = 0  # 真阴性 (GT=0, Pred=0)
    fp = 0  # 假阳性 (GT=0, Pred=1)
    fn = 0  # 假阴性 (GT=1, Pred=0)

    # 详细统计：格式为 details[真实标签][预测是否正确][理由是否正确]
    # keys: 1 (Vulnerable), 0 (Safe)
    # sub-keys: "p_corr_r_corr", "p_corr_r_wrong", "p_wrong_r_corr", "p_wrong_r_wrong"
    details = {
        1: {"corr_corr": 0, "corr_wrong": 0, "wrong_corr": 0, "wrong_wrong": 0},
        0: {"corr_corr": 0, "corr_wrong": 0, "wrong_corr": 0, "wrong_wrong": 0}
    }

    total = len(data)
    
    for item in data:
        # 获取字段 (确保类型正确)
        gt = int(item.get("gt_label", item.get("label", 0)))
        pred_is_vul = bool(item.get("pred_is_vul", False))
        
        # Judge 结果 (1代表正确，0代表错误)
        judge_pred_correct = int(item.get("prediction_correct", 0))
        judge_reason_correct = int(item.get("reason_correct", 0))

        # --- 计算基础混淆矩阵 ---
        if gt == 1 and pred_is_vul:
            tp += 1
        elif gt == 0 and not pred_is_vul:
            tn += 1
        elif gt == 0 and pred_is_vul:
            fp += 1
        elif gt == 1 and not pred_is_vul:
            fn += 1

        # --- 计算详细理由统计 ---
        # 确定由哪个键来存储
        p_key = "corr" if judge_pred_correct == 1 else "wrong"
        r_key = "corr" if judge_reason_correct == 1 else "wrong"
        final_key = f"{p_key}_{r_key}"
        
        details[gt][final_key] += 1

    # 2. 计算指标
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # 3. 输出报告
    print("\n" + "="*50)
    print("📊 模型评估报告 (Model Evaluation Report)")
    print("="*50)
    print(f"总样本数 (Total Samples): {total}")
    
    print("\n🔹 基础指标 (Basic Metrics):")
    print(f"{'Accuracy (准确率):':<25} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'Precision (精确率):':<25} {precision:.4f} ({precision*100:.2f}%)")
    print(f"{'Recall (召回率):':<25} {recall:.4f} ({recall*100:.2f}%)")
    print(f"{'F1 Score (F1分数):':<25} {f1:.4f}")

    print("\n🔹 混淆矩阵 (Confusion Matrix):")
    print(f"{'TP (真阳性 - 报对漏洞):':<30} {tp}")
    print(f"{'TN (真阴性 - 报对安全):':<30} {tn}")
    print(f"{'FP (假阳性 - 误报):':<30} {fp}")
    print(f"{'FN (假阴性 - 漏报):':<30} {fn}")

    print("\n" + "-"*50)
    print("🧠 详细推理质量分析 (Reasoning Analysis)")
    print("-"*50)

    # 打印 GT=1 的情况
    d1 = details[1]
    print("\n📌 真实标签 = 1 (Vulnerable/有漏洞) 的样本分布:")
    print(f"1. 判断正确 ✅ & 理由正确 ✅ : {d1['corr_corr']}")
    print(f"2. 判断正确 ✅ & 理由错误 ❌ : {d1['corr_wrong']}  <--(猜对了但理由不对)")
    print(f"3. 判断错误 ❌ & 理由正确 ✅ : {d1['wrong_corr']}  <--(罕见: Judge认为理由合理但结论下错了)")
    print(f"4. 判断错误 ❌ & 理由错误 ❌ : {d1['wrong_wrong']}  <--(完全错误)")
    
    subtotal_1 = sum(d1.values())
    print(f"   --> 合计 (Total GT=1): {subtotal_1}")

    # 打印 GT=0 的情况
    d0 = details[0]
    print("\n📌 真实标签 = 0 (Safe/无漏洞) 的样本分布:")
    print(f"1. 判断正确 ✅ & 理由正确 ✅ : {d0['corr_corr']}")
    print(f"2. 判断正确 ✅ & 理由错误 ❌ : {d0['corr_wrong']}")
    print(f"3. 判断错误 ❌ & 理由正确 ✅ : {d0['wrong_corr']}")
    print(f"4. 判断错误 ❌ & 理由错误 ❌ : {d0['wrong_wrong']}")

    subtotal_0 = sum(d0.values())
    print(f"   --> 合计 (Total GT=0): {subtotal_0}")

    print("\n📌 所有的样本分布:")
    print(f"1. 判断正确 ✅ & 理由正确 ✅ : {d0['corr_corr']+d1['corr_corr']}")
    print(f"2. 判断正确 ✅ & 理由错误 ❌ : {d0['corr_wrong']+d1['corr_wrong']}")
    print(f"3. 判断错误 ❌ & 理由正确 ✅ : {d0['wrong_corr']+d1['wrong_corr']}")
    print(f"4. 判断错误 ❌ & 理由错误 ❌ : {d0['wrong_wrong']+d1['wrong_wrong']}")

    print("\n📌 所有的样本分布百分比:")
    print(f"1. 判断正确 ✅ & 理由正确 ✅ : {d0['corr_corr']+d1['corr_corr']} ({safe_div(d0['corr_corr']+d1['corr_corr'], total)*100:.2f}%)")
    print(f"2. 判断正确 ✅ & 理由错误 ❌ : {d0['corr_wrong']+d1['corr_wrong']} ({safe_div(d0['corr_wrong']+d1['corr_wrong'], total)*100:.2f}%)")
    print(f"3. 判断错误 ❌ & 理由正确 ✅ : {d0['wrong_corr']+d1['wrong_corr']} ({safe_div(d0['wrong_corr']+d1['wrong_corr'], total)*100:.2f}%)")
    print(f"4. 判断错误 ❌ & 理由错误 ❌ : {d0['wrong_wrong']+d1['wrong_wrong']} ({safe_div(d0['wrong_wrong']+d1['wrong_wrong'], total)*100:.2f}%)")
    print(f"{'F1 Score (F1分数):':<25} {f1:.4f}")
    print("file:", RESULT_FILE)

    # subtotal_0 = sum(d0.values())
    print(f"   --> 合计 (Total): { total}")
    print("="*50)

if __name__ == "__main__":
    analyze(RESULT_FILE)
