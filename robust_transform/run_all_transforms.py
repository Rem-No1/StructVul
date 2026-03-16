import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_SPECS = [
    ("lexical_transform.py", "lexical", "lexical"),
    ("expression_transform.py", "expression", "expression"),
    ("control_flow_transform.py", "control", "control"),
    ("dataflow_encap_transform.py", "dataflow", "dataflow"),
    ("augment_c_code.py", "augmented", "augmented"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all robust_transform pipelines for one or more datasets.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path(r"D:\LIHAOZE\bishe\thecode\eee\dataset_v2\BaseCodeFilesReason.json")],
        help="One or more input dataset JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional shared output root directory. "
            "If omitted, outputs are created next to each dataset using *_lexical/*_expression/... folders."
        ),
    )
    parser.add_argument(
        "--variants-per-sample",
        type=int,
        default=None,
        help="Optional override passed to every transform script.",
    )
    parser.add_argument(
        "--max-transforms-per-variant",
        type=int,
        default=None,
        help="Optional override passed to every transform script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed passed to every transform script.",
    )
    parser.add_argument(
        "--drop-original",
        dest="drop_original",
        action="store_true",
        default=None,
        help="Pass through --drop-original to every transform script.",
    )
    parser.add_argument(
        "--keep-original",
        dest="drop_original",
        action="store_false",
        help="Pass through --keep-original to every transform script.",
    )
    parser.add_argument(
        "--keep-untransformed",
        dest="keep_untransformed",
        action="store_true",
        default=None,
        help="Pass through --keep-untransformed to every transform script.",
    )
    parser.add_argument(
        "--drop-untransformed",
        dest="keep_untransformed",
        action="store_false",
        help="Pass through --drop-untransformed to every transform script.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one script fails.",
    )
    return parser.parse_args()


def _output_dir_for(input_path: Path, category: str, output_root: Path | None) -> Path:
    dataset_folder = input_path.parent.name
    if output_root is not None:
        return output_root / dataset_folder / category
    return input_path.parent.parent / f"{dataset_folder}_{category}"


def _output_file_for(input_path: Path, category: str, output_root: Path | None) -> Path:
    out_dir = _output_dir_for(input_path, category, output_root)
    return out_dir / f"{input_path.stem}_{category}.json"


def _build_command(args: argparse.Namespace, script_name: str, input_path: Path, output_path: Path) -> List[str]:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / script_name)]
    cmd.extend(["--input", str(input_path), "--output", str(output_path)])
    if args.variants_per_sample is not None:
        cmd.extend(["--variants-per-sample", str(max(1, args.variants_per_sample))])
    if args.max_transforms_per_variant is not None:
        cmd.extend(["--max-transforms-per-variant", str(max(1, args.max_transforms_per_variant))])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.drop_original is True:
        cmd.append("--drop-original")
    elif args.drop_original is False:
        cmd.append("--keep-original")
    if args.keep_untransformed is True:
        cmd.append("--keep-untransformed")
    elif args.keep_untransformed is False:
        cmd.append("--drop-untransformed")
    return cmd


def _run_one(args: argparse.Namespace, input_path: Path, script_name: str, category: str, label: str) -> bool:
    output_path = _output_file_for(input_path, category, args.output_root)
    cmd = _build_command(args, script_name, input_path, output_path)
    print(f"[RUN] {label}: {input_path} -> {output_path}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK]  {label}", flush=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] {label}: exit code {exc.returncode}", flush=True)
        if not args.continue_on_error:
            raise
        return False


def main() -> None:
    args = parse_args()

    inputs = [p.expanduser().resolve() for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    total_jobs = len(inputs) * len(SCRIPT_SPECS)
    succeeded = 0

    for input_path in inputs:
        print(f"\n=== Dataset: {input_path} ===", flush=True)
        for script_name, category, label in SCRIPT_SPECS:
            if _run_one(args, input_path, script_name, category, label):
                succeeded += 1

    print(f"\nDone: {succeeded}/{total_jobs} jobs succeeded.", flush=True)


if __name__ == "__main__":
    main()
