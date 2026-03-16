import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import base.test as base_test
from base.build_prompt import get_supported_code_prompt_variants


REPO_ROOT = Path(__file__).resolve().parents[1]


DATASET_PRESETS: Dict[str, Tuple[str, str]] = {
    "v1": ("dataset_v1", "BaseCodeFilesReason.json"),
    "v1_control": ("dataset_v1_control", "BaseCodeFilesReason_control.json"),
    "v1_dataflow": ("dataset_v1_dataflow", "BaseCodeFilesReason_dataflow.json"),
    "v1_expression": ("dataset_v1_expression", "BaseCodeFilesReason_expression.json"),
    "v1_lexical": ("dataset_v1_lexical", "BaseCodeFilesReason_lexical.json"),
}


def parse_csv_arg(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg.strip().lower() == "all":
        return list(DATASET_PRESETS.keys())

    selected = parse_csv_arg(dataset_arg)
    unknown = [k for k in selected if k not in DATASET_PRESETS]
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {unknown}. Available: {list(DATASET_PRESETS.keys())}")
    return selected


def resolve_strategies(strategy_arg: str) -> List[str]:
    supported = set(get_supported_code_prompt_variants())
    if strategy_arg.strip().lower() == "all":
        return sorted(supported)

    selected = parse_csv_arg(strategy_arg)
    unknown = [s for s in selected if s not in supported]
    if unknown:
        raise ValueError(
            f"Unknown prompt variant(s): {unknown}. Supported: {sorted(supported)}"
        )
    return selected


def parse_strategy_suffix_map(raw_value: str) -> Dict[str, str]:
    """
    Parse mapping format:
    "few_shot_cot_explicit=fce,zero_shot_cot_explicit=zce"
    """
    mapping: Dict[str, str] = {}
    if not raw_value.strip():
        return mapping

    for pair in raw_value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Invalid strategy-suffix pair: '{pair}'. Expected format strategy=suffix."
            )
        strategy, suffix = pair.split("=", 1)
        strategy = strategy.strip()
        suffix = suffix.strip()
        if not strategy or not suffix:
            raise ValueError(
                f"Invalid strategy-suffix pair: '{pair}'. Strategy and suffix must be non-empty."
            )
        mapping[strategy] = suffix
    return mapping


def build_output_path(
    code_root: Path,
    dataset_file: Path,
    strategy: str,
    model_name: str,
    output_name_mode: str,
    strategy_suffix_map: Dict[str, str],
) -> Path:
    stem = dataset_file.stem
    if output_name_mode == "stem_suffix":
        suffix = strategy_suffix_map.get(strategy, strategy)
        out_name = f"{stem}_{suffix}.json"
    else:
        out_name = f"{model_name}_{stem}_{strategy}_results_batch.json"
    return code_root / out_name


def run_once(
    dataset_key: str,
    strategy: str,
    max_workers: int | None,
    output_name_mode: str,
    strategy_suffix_map: Dict[str, str],
) -> Dict[str, str]:
    root_dir_name, dataset_file_name = DATASET_PRESETS[dataset_key]
    code_root = REPO_ROOT / root_dir_name
    dataset_file = code_root / dataset_file_name
    output_json = build_output_path(
        code_root=code_root,
        dataset_file=dataset_file,
        strategy=strategy,
        model_name=base_test.CODE_MODEL_NAME,
        output_name_mode=output_name_mode,
        strategy_suffix_map=strategy_suffix_map,
    )

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    base_test.CODE_ROOT = code_root
    base_test.DATASET_FILE = dataset_file
    base_test.OUTPUT_JSON = output_json
    base_test.PROMPT_VARIANT = strategy
    if max_workers is not None:
        base_test.MAX_WORKERS = max_workers

    print("=" * 80)
    print(f"[RUN] dataset={dataset_key} strategy={strategy}")
    print(f"[DATASET] {dataset_file}")
    print(f"[OUTPUT]  {output_json}")
    print(f"[WORKERS] {base_test.MAX_WORKERS}")

    base_test.main()

    return {
        "dataset_key": dataset_key,
        "strategy": strategy,
        "dataset_file": str(dataset_file),
        "output_json": str(output_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner for base/test.py across multiple datasets and prompt strategies."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'. "
        f"Available: {', '.join(DATASET_PRESETS.keys())}",
    )
    # parser.add_argument(
    #     "--strategies",
    #     default="few_shot_cot_explicit,zero_shot_cot_explicit",
    #     help="Comma-separated prompt variants or 'all'. "
    #     f"Supported: {', '.join(get_supported_code_prompt_variants())}",
    # )

    # parser.add_argument(
    #     "--strategies",
    #     default="zero_shot_direct",
    #     help="Comma-separated prompt variants or 'all'. "
    #     f"Supported: {', '.join(get_supported_code_prompt_variants())}",
    # )


    parser.add_argument(
        "--strategies",
        default="all",
        help="Comma-separated prompt variants or 'all'. "
        f"Supported: {', '.join(get_supported_code_prompt_variants())}",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for MAX_WORKERS in base/test.py.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one run fails.",
    )
    parser.add_argument(
        "--output-name-mode",
        choices=["default", "stem_suffix"],
        default="default",
        help=(
            "Output naming mode. "
            "'default': model_stem_strategy_results_batch.json; "
            "'stem_suffix': stem_suffix.json (suffix from --strategy-suffix-map or strategy name)."
        ),
    )
    parser.add_argument(
        "--strategy-suffix-map",
        default="",
        help=(
            "Strategy suffix mapping, format: "
            "few_shot_cot_explicit=fce,zero_shot_cot_explicit=zce"
        ),
    )
    args = parser.parse_args()

    datasets = resolve_datasets(args.datasets)
    strategies = resolve_strategies(args.strategies)
    strategy_suffix_map = parse_strategy_suffix_map(args.strategy_suffix_map)

    summary: List[Dict[str, str]] = []
    started_at = time.time()

    print("[BATCH] Start")
    print(f"[BATCH] datasets={datasets}")
    print(f"[BATCH] strategies={strategies}")
    print(f"[BATCH] output_name_mode={args.output_name_mode}")
    if strategy_suffix_map:
        print(f"[BATCH] strategy_suffix_map={strategy_suffix_map}")

    for dataset_key in datasets:
        for strategy in strategies:
            run_start = time.time()
            try:
                record = run_once(
                    dataset_key=dataset_key,
                    strategy=strategy,
                    max_workers=args.max_workers,
                    output_name_mode=args.output_name_mode,
                    strategy_suffix_map=strategy_suffix_map,
                )
                record["status"] = "success"
                record["elapsed_sec"] = f"{time.time() - run_start:.2f}"
                summary.append(record)
            except Exception as exc:
                error_record = {
                    "dataset_key": dataset_key,
                    "strategy": strategy,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_sec": f"{time.time() - run_start:.2f}",
                }
                summary.append(error_record)
                print(f"[FAILED] dataset={dataset_key} strategy={strategy}: {exc}")
                if args.stop_on_error:
                    break
        if args.stop_on_error and summary and summary[-1].get("status") == "failed":
            break

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = REPO_ROOT / "base" / f"batch_run_summary_{timestamp}.json"
    summary_data = {
        "started_at_epoch": started_at,
        "finished_at_epoch": time.time(),
        "datasets": datasets,
        "strategies": strategies,
        "output_name_mode": args.output_name_mode,
        "strategy_suffix_map": strategy_suffix_map,
        "results": summary,
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    success_count = sum(1 for x in summary if x.get("status") == "success")
    fail_count = sum(1 for x in summary if x.get("status") == "failed")
    print("=" * 80)
    print(f"[BATCH] done. success={success_count}, failed={fail_count}")
    print(f"[BATCH] summary={summary_path}")


if __name__ == "__main__":
    main()
