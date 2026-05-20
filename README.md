# Code Vulnerability Detection Robustness Evaluation

This repository contains experiments for evaluating LLM-based code vulnerability detection and rationale quality under prompt, retrieval, and semantic-preserving code transformation settings.

The project includes:

- Baseline LLM vulnerability prediction and judge-based rationale evaluation.
- Prompt strategy comparison, including zero-shot, chain-of-thought, and few-shot variants.
- CPG/RAG-based retrieval evaluation with k-fold splits.
- Semantic-preserving transformations for C/Python code robustness experiments.
- Metric utilities for Accuracy, Precision, Recall, and F1-score.

## Repository Layout

```text
.
|-- base/                    # Baseline LLM prediction, judging, batch runs, metrics
|-- met12/                   # CPG/RAG evaluation and k-fold experiment scripts
|-- robust_transform/        # Semantic-preserving code transformation tools
|-- dataset_v1*/             # Dataset v1 and transformed variants
|-- dataset_v2*/             # Dataset v2 and transformed variants
|-- .env.example             # Environment variable template, no real secrets
|-- .gitignore
`-- README.md
```

Generated outputs such as RAG stores, result JSON files, summaries, plots, LaTeX tables, caches, and local `.env` files are ignored by Git.

## Environment

Recommended:

- Python 3.10+
- `srcml` installed and available on `PATH` if using C semantic transformations
- A compiler such as `gcc` or `clang` for transformation regression checks

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openai tqdm requests chromadb dashscope libcst tree-sitter tree-sitter-c
```

Check optional external tools:

```powershell
srcml --version
gcc --version
```

## API Keys

Do not commit real API keys. This project reads credentials from environment variables.

PowerShell example:

```powershell
$env:CODE_API_KEY="your-code-model-api-key"
$env:CODE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:CODE_MODEL_NAME="qwen3-32b"

$env:JUDGE_API_KEY="your-judge-model-api-key"
$env:JUDGE_BASE_URL="https://api.deepseek.com"
$env:JUDGE_MODEL_NAME="deepseek-chat"

$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

`.env.example` documents the expected names. The scripts do not automatically load `.env`; either set variables in the shell, configure them in your IDE, or load them with your own environment manager.

## Dataset Format

Datasets are JSON lists. Each record is expected to contain fields similar to:

```json
{
  "id": 1,
  "language": "C",
  "code": "int main() { return 0; }",
  "label": 0,
  "reason": "Reference explanation"
}
```

Common dataset directories:

- `dataset_v1/`, `dataset_v2/`: base datasets
- `dataset_v*_control/`: control-flow transformed datasets
- `dataset_v*_dataflow/`: dataflow/encapsulation transformed datasets
- `dataset_v*_expression/`: expression transformed datasets
- `dataset_v*_lexical/`: lexical transformed datasets
- `dataset_v*_augmented/`: mixed transformed datasets

## Baseline Evaluation

Run baseline experiments through the batch runner from the repository root:

```powershell
python -m base.run_batch `
  --datasets v1,v1_control,v1_dataflow,v1_expression,v1_lexical `
  --strategies zero_shot_direct,few_shot_cot `
  --max-workers 4
```

Supported prompt strategies include:

- `zero_shot_direct`
- `zero_shot_cot`
- `few_shot_direct`
- `few_shot_cot`
- `few_shot_cot_explicit`

You can also run all configured datasets and strategies:

```powershell
python -m base.run_batch --datasets all --strategies all --max-workers 4
```

The batch runner overrides the dataset paths used by `base/test.py`, so it is the safer entry point for normal runs.

## Metrics

Compute classification metrics from result JSON files:

```powershell
python -m base.calc_classification_metrics dataset_v1
```

Write metrics to CSV:

```powershell
python -m base.calc_classification_metrics dataset_v1 --output-csv base/metrics.csv
```

When no input is provided, the script searches under `dataset_v*` directories for matching result files.

## CPG/RAG K-Fold Evaluation

The RAG evaluation uses DashScope embeddings/reranking and ChromaDB persistence. Set all required API environment variables before running.

Main script:

```powershell
python -m met12.kfold_eval
```

Before running, check the configuration constants at the top of `met12/kfold_eval.py`, especially:

- `K_FOLDS`
- `RANDOM_SEED`
- `MAX_WORKERS`
- `CODE_MODEL_NAME`
- `DATASET_PATH`
- `OUTPUT_DIR`
- `PERSISTENT_RAG_STORE_ROOT`

Some historical experiment paths are absolute Windows paths. Adjust them to your local checkout before reproducing a run.

## Robust Code Transformations

Transformation tools live in `robust_transform/`. They generate semantic-preserving dataset variants.

Examples:

```powershell
python robust_transform/lexical_transform.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_lexical/BaseCodeFilesReason_lexical.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 2
```

```powershell
python robust_transform/augment_c_code.py `
  --input dataset_v1/BaseCodeFilesReason.json `
  --output dataset_v1_augmented/BaseCodeFilesReason_augmented.json `
  --variants-per-sample 1 `
  --max-transforms-per-variant 3
```

Available transformation entry points:

- `robust_transform/lexical_transform.py`
- `robust_transform/expression_transform.py`
- `robust_transform/control_flow_transform.py`
- `robust_transform/dataflow_encap_transform.py`
- `robust_transform/augment_c_code.py`

The C transformation path depends on `srcml`. Python transformations additionally use Python AST/CST tooling.

