import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

import met12.test as dual_test
from met12.rag_engine import VulnRAG

# ================= K-Fold Config =================
K_FOLDS = 5
RANDOM_SEED = 1
MAX_WORKERS = 6
REUSE_PERSISTENT_RAG_STORE = True
FORCE_REBUILD_RAG = False

CODE_MODEL_NAME = "qwen3-32b"
# CODE_MODEL_NAME = "deepseek-chat"
# CODE_MODEL_NAME = "gpt-5.1-chat-latest"

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1/BaseCodeFilesReason.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1/{CODE_MODEL_NAME}_kfold_dual_system")
# # # 永久存储目录：向量库会长期保存在这里，重启后仍可复用（除非手动删除）
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1/{CODE_MODEL_NAME}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/BaseCodeFilesReason_control.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/{CODE_MODEL_NAME}_kfold_dual_system")
# # # 永久存储目录：向量库会长期保存在这里，重启后仍可复用（除非手动删除）
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/{CODE_MODEL_NAME}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/BaseCodeFilesReason_dataflow.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/{CODE_MODEL_NAME}_kfold_dual_system")
# # # # 永久存储目录：向量库会长期保存在这里，重启后仍可复用（除非手动删除）
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/{CODE_MODEL_NAME}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/BaseCodeFilesReason_expression.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/{CODE_MODEL_NAME}_kfold_dual_system")
# # # # # 永久存储目录：向量库会长期保存在这里，重启后仍可复用（除非手动删除）
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/{CODE_MODEL_NAME}_kfold_rag_store")

DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/BaseCodeFilesReason_lexical.json")
OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/{CODE_MODEL_NAME}_kfold_dual_system")
# # 永久存储目录：向量库会长期保存在这里，重启后仍可复用（除非手动删除）
PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/{CODE_MODEL_NAME}_kfold_rag_store")

# RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_ID="met12_seed"+str(RANDOM_SEED)
RUN_DIR = OUTPUT_DIR / RUN_ID
# 当你修改了 CPG/Embedding 逻辑时，手动增加版本号可避免旧库污染
RAG_STORE_VERSION = "vmet12"+str(RANDOM_SEED)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Dataset root must be list, got {type(data).__name__}")
    return data


def build_stratified_folds(data: List[Dict[str, Any]], k: int, seed: int) -> List[List[int]]:
    if k < 2:
        raise ValueError("k must be >= 2")

    label_to_indices: Dict[int, List[int]] = {}
    for idx, item in enumerate(data):
        label = int(item.get("label", item.get("gt_label", 0)))
        label_to_indices.setdefault(label, []).append(idx)

    rnd = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(k)]

    for indices in label_to_indices.values():
        indices_copy = indices[:]
        rnd.shuffle(indices_copy)
        for i, idx in enumerate(indices_copy):
            folds[i % k].append(idx)

    for fold in folds:
        fold.sort()

    return folds


def build_fold_signature(train_samples: List[Dict[str, Any]]) -> str:
    """
    基于训练集样本构造稳定签名，用于复用同一 fold 的向量库缓存。
    """
    basis: List[str] = []
    for item in train_samples:
        sid = str(item.get("id", ""))
        label = int(item.get("label", item.get("gt_label", 0)))
        code_len = len(str(item.get("code", "")))
        basis.append(f"{sid}|{label}|{code_len}")
    basis.sort()
    digest = hashlib.sha1("\n".join(basis).encode("utf-8")).hexdigest()[:12]
    return digest


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    tp = tn = fp = fn = 0

    for item in results:
        gt = int(item.get("gt_label", item.get("label", 0)))
        pred = bool(item.get("pred_is_vul", False))
        if gt == 1 and pred:
            tp += 1
        elif gt == 0 and not pred:
            tn += 1
        elif gt == 0 and pred:
            fp += 1
        elif gt == 1 and not pred:
            fn += 1

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

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
    }


def evaluate_fold(
    fold_id: int,
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print(f"\n===== Fold {fold_id + 1}/{K_FOLDS} =====")
    print(f"Train size: {len(train_samples)}, Test size: {len(test_samples)}")

    fold_signature = build_fold_signature(train_samples)
    plan_tag = f"{DATASET_PATH.stem}_k{K_FOLDS}_seed{RANDOM_SEED}"

    if REUSE_PERSISTENT_RAG_STORE and not FORCE_REBUILD_RAG:
        fold_rag_dir = (
            PERSISTENT_RAG_STORE_ROOT
            / RAG_STORE_VERSION
            / plan_tag
            / f"fold_{fold_id + 1}_{fold_signature}"
        )
        collection_name = f"cpg_vuln_kb_fold_{fold_id + 1}_{fold_signature}"
    else:
        fold_rag_dir = RUN_DIR / "rag_db" / f"fold_{fold_id + 1}_{fold_signature}_{RUN_ID}"
        collection_name = f"cpg_vuln_kb_fold_{fold_id + 1}_{fold_signature}_{RUN_ID}"

    db_exists_before = fold_rag_dir.exists() and any(fold_rag_dir.iterdir())
    fold_rag_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = fold_rag_dir / "train_split.json"
    if not train_json_path.exists():
        with train_json_path.open("w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False)

    # 每个 fold 只用训练集构建知识库；若缓存已有则直接复用
    dual_test.rag_engine = VulnRAG(
        dataset_path=str(train_json_path),
        persist_dir=str(fold_rag_dir),
        collection_name=collection_name,
    )
    kb_count = dual_test.rag_engine.collection.count()
    store_hit = db_exists_before and kb_count > 0
    print(f"RAG store dir: {fold_rag_dir}")
    print(f"RAG store reused: {store_hit} (entries={kb_count})")

    fold_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(dual_test.process_single_sample, sample): sample for sample in test_samples}
        for future in tqdm(as_completed(futures), total=len(test_samples), desc=f"Fold {fold_id + 1}"):
            try:
                res = future.result()
                res["fold_id"] = fold_id + 1
                fold_results.append(res)
            except Exception as exc:
                sample = futures[future]
                fold_results.append(
                    {
                        "id": str(sample.get("id", "unknown")),
                        "fold_id": fold_id + 1,
                        "error": str(exc),
                    }
                )

    metrics = compute_metrics([r for r in fold_results if "error" not in r])

    fold_out_path = RUN_DIR / f"fold_{fold_id + 1}_results.json"
    with fold_out_path.open("w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)

    print(
        f"Fold {fold_id + 1} metrics: "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )

    return {
        "fold_id": fold_id + 1,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "metrics": metrics,
        "fold_signature": fold_signature,
        "rag_store_dir": str(fold_rag_dir),
        "rag_store_reused": store_hit,
        "rag_kb_count": kb_count,
        "result_file": str(fold_out_path),
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if REUSE_PERSISTENT_RAG_STORE and not FORCE_REBUILD_RAG:
        PERSISTENT_RAG_STORE_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)
    folds = build_stratified_folds(dataset, K_FOLDS, RANDOM_SEED)

    summaries: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for fold_id in range(K_FOLDS):
        test_idx_set = set(folds[fold_id])
        train_samples = [dataset[i] for i in range(len(dataset)) if i not in test_idx_set]
        test_samples = [dataset[i] for i in folds[fold_id]]

        summary = evaluate_fold(fold_id, train_samples, test_samples)
        summaries.append(summary)

        fold_result_path = Path(summary["result_file"])
        fold_result_data = json.loads(fold_result_path.read_text(encoding="utf-8"))
        all_results.extend(fold_result_data)

    valid_results = [r for r in all_results if "error" not in r]
    overall_metrics = compute_metrics(valid_results)

    report = {
        "run_id": RUN_ID,
        "dataset_path": str(DATASET_PATH),
        "k_folds": K_FOLDS,
        "random_seed": RANDOM_SEED,
        "max_workers": MAX_WORKERS,
        "reuse_persistent_rag_store": REUSE_PERSISTENT_RAG_STORE,
        "force_rebuild_rag": FORCE_REBUILD_RAG,
        "rag_store_version": RAG_STORE_VERSION,
        "persistent_rag_store_root": str(PERSISTENT_RAG_STORE_ROOT),
        "fold_summaries": summaries,
        "overall_metrics": overall_metrics,
        "total_results": len(all_results),
        "valid_results": len(valid_results),
        "error_results": len(all_results) - len(valid_results),
    }

    summary_path = RUN_DIR / "kfold_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_path = RUN_DIR / "kfold_all_results.json"
    merged_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== K-Fold Completed =====")
    print(f"Run dir: {RUN_DIR}")
    print(f"Summary: {summary_path}")
    print(f"Merged results: {merged_path}")
    print(
        "Overall metrics: "
        f"acc={overall_metrics['accuracy']:.4f}, "
        f"prec={overall_metrics['precision']:.4f}, "
        f"recall={overall_metrics['recall']:.4f}, "
        f"f1={overall_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
