import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tqdm import tqdm

import met12.test as dual_test
from met12.build_prompt import (
    build_code_prediction_prompts as build_full_prompts,
    build_judge_prompts,
)
from met12.rag_engine import VulnRAG as StructuredVulnRAG


# ================= K-Fold Config =================
K_FOLDS = 5
RANDOM_SEED = 1
MAX_WORKERS = 6
REUSE_PERSISTENT_RAG_STORE = True
FORCE_REBUILD_RAG = False

CODE_MODEL_NAME = dual_test.CODE_MODEL_NAME

# DEFAULT_MODE = "all"
# DEFAULT_MODE = "no_cot"
DEFAULT_MODE = "no_structured_retrieval"


DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1/BaseCodeFilesReason.json")
OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_dual_system")
PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/BaseCodeFilesReason_control.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_dual_system")
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_control/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/BaseCodeFilesReason_dataflow.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_dual_system")
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_dataflow/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/BaseCodeFilesReason_expression.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_dual_system")
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_expression/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_rag_store")

# DATASET_PATH = Path("D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/BaseCodeFilesReason_lexical.json")
# OUTPUT_DIR = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_dual_system")
# PERSISTENT_RAG_STORE_ROOT = Path(f"D:/LIHAOZE/bishe/thecode/eee/dataset_v1_lexical/{CODE_MODEL_NAME}_{DEFAULT_MODE}_kfold_rag_store")


RAG_TOP_K = 3


PromptBuilder = Callable[[str, str, List[Dict[str, Any]] | None], Tuple[str, str]]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_structured_retrieval: bool
    use_cot: bool

    @property
    def run_id(self) -> str:
        return f"met12_abl_{self.name}_seed{RANDOM_SEED}"

    @property
    def rag_store_version(self) -> str:
        return f"vmet12_abl_{self.name}_seed{RANDOM_SEED}"

    @property
    def retrieval_mode(self) -> str:
        return "cpg_structured_retrieval" if self.use_structured_retrieval else "raw_code_retrieval"

    @property
    def cot_mode(self) -> str:
        return "enabled" if self.use_cot else "disabled"


ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    "full": AblationConfig(
        name="full",
        use_structured_retrieval=True,
        use_cot=True,
    ),
    "no_structured_retrieval": AblationConfig(
        name="no_structured_retrieval",
        use_structured_retrieval=False,
        use_cot=True,
    ),
    "no_cot": AblationConfig(
        name="no_cot",
        use_structured_retrieval=True,
        use_cot=False,
    ),
}


class RawCodeAblationRAG(StructuredVulnRAG):
    """
    Ablation RAG:
    - keep the same embedding / rerank / metadata pipeline as met12
    - remove CPG structured representation
    - index and query raw code directly
    """

    def _ingest_data(self, dataset_path: str) -> None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []

        print("Indexing raw code only (no CPG structured retrieval) ...")
        for idx, item in enumerate(tqdm(data, desc="Processing Raw Code")):
            code = str(item.get("code", ""))
            if not code.strip():
                continue

            doc_id = str(item.get("id", f"idx_{idx}"))
            ids.append(doc_id)
            documents.append(code)
            metadatas.append(
                {
                    "original_code": code,
                    "is_vulnerable": str(item.get("label", 0)),
                    "vuln_type": str(item.get("vuln_type", "Unknown")),
                    "reason": str(item.get("reason", "No reason"))[:1000],
                }
            )

        batch_size = 10
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print("Raw-code ablation knowledge base build completed.")

    def search(
        self,
        query_code: str,
        k: int = 10,
        rerank_top_k: int = 3,
        exclude_id: str = None,
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_code],
            n_results=k + 2,
        )

        candidates: List[Dict[str, Any]] = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                cid = str(results["ids"][0][i])
                if exclude_id and cid == str(exclude_id):
                    continue

                candidates.append(
                    {
                        "id": cid,
                        "code": results["metadatas"][0][i]["original_code"],
                        "is_vulnerable": results["metadatas"][0][i]["is_vulnerable"],
                        "vuln_type": results["metadatas"][0][i]["vuln_type"],
                        "reason": results["metadatas"][0][i]["reason"],
                        "vector_distance": results["distances"][0][i] if "distances" in results else 0,
                    }
                )

        return self.rerank(query=query_code, retrieved_docs=candidates, top_n=rerank_top_k)


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


def format_reference_cases(similar_cases: List[Dict[str, Any]] | None) -> str:
    if not similar_cases:
        return "[REFERENCE CASES]\nNo similar cases retrieved.\n"

    sections = [
        "[REFERENCE CASES]",
        "Use them only if they are truly relevant to the target logic.",
    ]

    for idx, case in enumerate(similar_cases, start=1):
        label_str = "VULNERABLE" if str(case.get("is_vulnerable", "0")) == "1" else "SAFE"
        ref_code = str(case.get("code", ""))
        ref_reason = str(case.get("reason", ""))
        if len(ref_code) > 1200:
            ref_code = ref_code[:1200] + "\n... (truncated)"
        if len(ref_reason) > 400:
            ref_reason = ref_reason[:400] + "..."

        sections.append(
            f"""--- Case {idx} ({label_str}) ---
Vulnerability Type: {case.get("vuln_type", "Unknown")}
Code:
{ref_code}

Analysis:
{ref_reason}
"""
        )

    return "\n".join(sections).strip()


def build_no_cot_prompts(
    code: str,
    language: str,
    similar_cases: List[Dict[str, Any]] | None = None,
) -> Tuple[str, str]:
    rag_context = format_reference_cases(similar_cases)

    system_prompt = (
        "You are a security expert. "
        "Your goal is to accurately detect vulnerabilities. "
        "Output ONLY a single JSON object and nothing else."
    )

    user_prompt = f"""
[TASK]
Analyze the code for security vulnerabilities.

[LANGUAGE]
{language}

{rag_context}

[TARGET CODE]
{code}
[CODE END]

[INSTRUCTION]
1. Check whether the code receives external input and reaches sensitive operations.
2. Use retrieved reference cases only when they match the target logic.
3. If the references are irrelevant, ignore them and judge only from the target code.
4. Do not output <thinking> tags, step-by-step reasoning, markdown, or extra text.
5. Return only the final JSON object.

Return a single JSON object with EXACTLY this format:

{{
  "is_vulnerable": true or false,
  "vuln_type": "string",
  "prediction_reason": "A concise paragraph explaining the key evidence."
}}
""".strip()

    return system_prompt, user_prompt


def get_prompt_builder(config: AblationConfig) -> PromptBuilder:
    return build_full_prompts if config.use_cot else build_no_cot_prompts


def get_rag_class(config: AblationConfig):
    return StructuredVulnRAG if config.use_structured_retrieval else RawCodeAblationRAG


def process_single_sample(
    sample: Dict[str, Any],
    rag_engine: Any,
    prompt_builder: PromptBuilder,
    config: AblationConfig,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id", "unknown"))
    gt_label = int(sample.get("label", 0))
    code = sample.get("code", "")
    language = sample.get("language", "C")

    similar_cases: List[Dict[str, Any]] = []
    retrieved_ids: List[str] = []

    if rag_engine is not None:
        similar_cases = rag_engine.search(
            query_code=code,
            rerank_top_k=RAG_TOP_K,
            exclude_id=sample_id,
        )
        retrieved_ids = [str(case.get("id", "N/A")) for case in similar_cases]

    code_sys, code_user = prompt_builder(
        code=code,
        language=language,
        similar_cases=similar_cases,
    )
    code_res = dual_test.call_code_llm(code_sys, code_user)

    judge_sys, judge_user = build_judge_prompts(
        code=code,
        language=language,
        gt_label=gt_label,
        reference_reason=sample.get("reason", ""),
        model_is_vul=code_res["is_vulnerable"],
        model_vuln_type=code_res["vuln_type"],
        model_reason=code_res["prediction_reason"],
    )
    judge_res = dual_test.call_judge_llm(judge_sys, judge_user)

    return {
        "id": sample_id,
        "code": code,
        "gt_label": gt_label,
        "ablation_mode": config.name,
        "retrieval_mode": config.retrieval_mode,
        "cot_mode": config.cot_mode,
        "rag_retrieved_ids": retrieved_ids,
        "rag_context_count": len(similar_cases),
        "pred_is_vul": code_res["is_vulnerable"],
        "pred_vuln_type": code_res["vuln_type"],
        "pred_reason": code_res["prediction_reason"],
        "prediction_correct": judge_res["prediction_correct"],
        "reason_correct": judge_res["reason_correct"],
        "missing_points": judge_res["missing_points"],
        "wrong_points": judge_res["wrong_points"],
    }


def evaluate_fold(
    fold_id: int,
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    config: AblationConfig,
    run_dir: Path,
) -> Dict[str, Any]:
    print(f"\n===== {config.name} | Fold {fold_id + 1}/{K_FOLDS} =====")
    print(f"Train size: {len(train_samples)}, Test size: {len(test_samples)}")

    fold_signature = build_fold_signature(train_samples)
    plan_tag = f"{DATASET_PATH.stem}_{config.name}_k{K_FOLDS}_seed{RANDOM_SEED}"

    if REUSE_PERSISTENT_RAG_STORE and not FORCE_REBUILD_RAG:
        fold_rag_dir = (
            PERSISTENT_RAG_STORE_ROOT
            / config.rag_store_version
            / plan_tag
            / f"fold_{fold_id + 1}_{fold_signature}"
        )
        collection_name = f"{config.name}_fold_{fold_id + 1}_{fold_signature}"
    else:
        fold_rag_dir = run_dir / "rag_db" / f"fold_{fold_id + 1}_{fold_signature}_{config.run_id}"
        collection_name = f"{config.name}_fold_{fold_id + 1}_{fold_signature}_{config.run_id}"

    db_exists_before = fold_rag_dir.exists() and any(fold_rag_dir.iterdir())
    fold_rag_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = fold_rag_dir / "train_split.json"
    if not train_json_path.exists():
        with train_json_path.open("w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False)

    rag_cls = get_rag_class(config)
    rag_engine = rag_cls(
        dataset_path=str(train_json_path),
        persist_dir=str(fold_rag_dir),
        collection_name=collection_name,
    )
    kb_count = rag_engine.collection.count()
    store_hit = db_exists_before and kb_count > 0

    print(f"RAG store dir: {fold_rag_dir}")
    print(f"RAG store reused: {store_hit} (entries={kb_count})")
    print(f"Retrieval mode: {config.retrieval_mode}")
    print(f"CoT mode: {config.cot_mode}")

    prompt_builder = get_prompt_builder(config)

    fold_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_sample, sample, rag_engine, prompt_builder, config): sample
            for sample in test_samples
        }
        for future in tqdm(as_completed(futures), total=len(test_samples), desc=f"{config.name} Fold {fold_id + 1}"):
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
                        "ablation_mode": config.name,
                        "error": str(exc),
                    }
                )

    metrics = compute_metrics([r for r in fold_results if "error" not in r])

    fold_out_path = run_dir / f"fold_{fold_id + 1}_results.json"
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


def run_single_experiment(config: AblationConfig) -> None:
    run_dir = OUTPUT_DIR / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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

        summary = evaluate_fold(
            fold_id=fold_id,
            train_samples=train_samples,
            test_samples=test_samples,
            config=config,
            run_dir=run_dir,
        )
        summaries.append(summary)

        fold_result_path = Path(summary["result_file"])
        fold_result_data = json.loads(fold_result_path.read_text(encoding="utf-8"))
        all_results.extend(fold_result_data)

    valid_results = [r for r in all_results if "error" not in r]
    overall_metrics = compute_metrics(valid_results)

    report = {
        "run_id": config.run_id,
        "dataset_path": str(DATASET_PATH),
        "method": "met12_ablation",
        "ablation_mode": config.name,
        "retrieval_mode": config.retrieval_mode,
        "cot_mode": config.cot_mode,
        "k_folds": K_FOLDS,
        "random_seed": RANDOM_SEED,
        "max_workers": MAX_WORKERS,
        "reuse_persistent_rag_store": REUSE_PERSISTENT_RAG_STORE,
        "force_rebuild_rag": FORCE_REBUILD_RAG,
        "rag_store_version": config.rag_store_version,
        "persistent_rag_store_root": str(PERSISTENT_RAG_STORE_ROOT),
        "fold_summaries": summaries,
        "overall_metrics": overall_metrics,
        "total_results": len(all_results),
        "valid_results": len(valid_results),
        "error_results": len(all_results) - len(valid_results),
    }

    summary_path = run_dir / "kfold_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_path = run_dir / "kfold_all_results.json"
    merged_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== {config.name} Completed =====")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {summary_path}")
    print(f"Merged results: {merged_path}")
    print(
        "Overall metrics: "
        f"acc={overall_metrics['accuracy']:.4f}, "
        f"prec={overall_metrics['precision']:.4f}, "
        f"recall={overall_metrics['recall']:.4f}, "
        f"f1={overall_metrics['f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run met12 k-fold ablation experiments for structured retrieval and CoT."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "full", "no_structured_retrieval", "no_cot"],
        default=DEFAULT_MODE,
        help="Which experiment to run. 'all' runs full + both ablations.",
    )
    return parser.parse_args()


def resolve_experiments(mode: str) -> List[AblationConfig]:
    if mode == "all":
        return [
            ABLATION_CONFIGS["full"],
            ABLATION_CONFIGS["no_structured_retrieval"],
            ABLATION_CONFIGS["no_cot"],
        ]
    return [ABLATION_CONFIGS[mode]]


def main() -> None:
    args = parse_args()
    for config in resolve_experiments(args.mode):
        run_single_experiment(config)


if __name__ == "__main__":
    main()
