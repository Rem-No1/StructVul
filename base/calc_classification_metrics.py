import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_RESULT_PATTERNS = [
    "*_BaseCodeFilesReason_results_*.json",
    "*/kfold_all_results.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Accuracy, Precision, Recall, and F1-score from result JSON files.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Result JSON files or directories containing result JSON files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root used when no positional inputs are provided.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places for reported metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def round_half_up(value: float, decimals: int) -> str:
    quantizer = Decimal("1").scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def collect_result_files(inputs: List[Path], root: Path) -> List[Path]:
    if inputs:
        result_files: List[Path] = []
        for input_path in inputs:
            resolved = input_path.resolve()
            if resolved.is_file():
                result_files.append(resolved)
                continue
            if resolved.is_dir():
                for pattern in DEFAULT_RESULT_PATTERNS:
                    result_files.extend(resolved.rglob(pattern))
                continue
            raise FileNotFoundError(f"Input path not found: {input_path}")
        return sorted(set(result_files))

    root = root.resolve()
    result_files = []
    for dataset_dir in sorted(root.glob("dataset_v*")):
        if not dataset_dir.is_dir():
            continue
        for pattern in DEFAULT_RESULT_PATTERNS:
            result_files.extend(dataset_dir.rglob(pattern))
    return sorted(set(result_files))


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def compute_metrics(records: Iterable[dict]) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    total = 0

    for item in records:
        gt = int(item.get("gt_label", item.get("label", 0)))
        pred_is_vul = bool(item.get("pred_is_vul", False))
        total += 1

        if gt == 1 and pred_is_vul:
            tp += 1
        elif gt == 0 and not pred_is_vul:
            tn += 1
        elif gt == 0 and pred_is_vul:
            fp += 1
        elif gt == 1 and not pred_is_vul:
            fn += 1

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_row(label: str, metrics: Dict[str, float], decimals: int) -> str:
    return (
        f"{label}\n"
        f"  Samples:   {metrics['samples']}\n"
        f"  TP/TN/FP/FN: {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}\n"
        f"  Accuracy:  {round_half_up(metrics['accuracy'], decimals)}\n"
        f"  Precision: {round_half_up(metrics['precision'], decimals)}\n"
        f"  Recall:    {round_half_up(metrics['recall'], decimals)}\n"
        f"  F1-score:  {round_half_up(metrics['f1'], decimals)}"
    )


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "samples",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_files = collect_result_files(args.inputs, args.root)
    if not result_files:
        raise FileNotFoundError("No result JSON files found.")

    root = args.root.resolve()
    csv_rows: List[Dict[str, str]] = []

    for path in result_files:
        metrics = compute_metrics(load_json(path))
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)

        print(format_row(label, metrics, args.decimals))
        print()

        csv_rows.append(
            {
                "file": label,
                "samples": str(metrics["samples"]),
                "tp": str(metrics["tp"]),
                "tn": str(metrics["tn"]),
                "fp": str(metrics["fp"]),
                "fn": str(metrics["fn"]),
                "accuracy": round_half_up(metrics["accuracy"], args.decimals),
                "precision": round_half_up(metrics["precision"], args.decimals),
                "recall": round_half_up(metrics["recall"], args.decimals),
                "f1": round_half_up(metrics["f1"], args.decimals),
            }
        )

    if args.output_csv is not None:
        write_csv(csv_rows, args.output_csv)
        print(f"Wrote CSV to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
