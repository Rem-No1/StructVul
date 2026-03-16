import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List

from python_transforms import LEXICAL_TRANSFORMS as PYTHON_LEXICAL_TRANSFORMS
from srcml_transforms import LEXICAL_TRANSFORMS as C_LEXICAL_TRANSFORMS
from transform_utils import (
    default_parser,
    load_dataset,
    next_numeric_id,
    save_dataset,
    transform_code_by_pool,
)


DEFAULT_INPUT = Path("dataset_v1/BaseCodeFilesReason.json")
DEFAULT_OUTPUT = Path("dataset_v1_lexical/BaseCodeFilesReason_lexical.json")
DEFAULT_VARIANTS_PER_SAMPLE = 1
DEFAULT_MAX_TRANSFORMS_PER_VARIANT = 3
DEFAULT_SEED = 42
DEFAULT_DROP_ORIGINAL = True
DEFAULT_KEEP_UNTRANSFORMED = True


def parse_args():
    all_transforms = sorted(set(C_LEXICAL_TRANSFORMS.keys()) | set(PYTHON_LEXICAL_TRANSFORMS.keys()))
    parser = default_parser(
        "Language-aware lexical semantic-preserving transforms for C/Python",
        default_input=DEFAULT_INPUT,
        default_output=DEFAULT_OUTPUT,
    )
    parser.set_defaults(
        variants_per_sample=DEFAULT_VARIANTS_PER_SAMPLE,
        max_transforms_per_variant=DEFAULT_MAX_TRANSFORMS_PER_VARIANT,
        seed=DEFAULT_SEED,
        drop_original=DEFAULT_DROP_ORIGINAL,
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=all_transforms,
        choices=all_transforms,
        help="Enabled lexical transforms",
    )
    parser.add_argument(
        "--keep-untransformed",
        dest="keep_untransformed",
        action="store_true",
        default=DEFAULT_KEEP_UNTRANSFORMED,
        help="Keep a fallback sample when a variant fails to transform.",
    )
    parser.add_argument(
        "--drop-untransformed",
        dest="keep_untransformed",
        action="store_false",
        help="Drop variants that fail to transform.",
    )
    return parser.parse_args()


def _normalize_language(value: object) -> str:
    if not isinstance(value, str):
        return "c"
    normalized = value.strip().lower()
    if normalized in {"c", "c99", "c11", "c17", "c18", "c23", "cpp", "c++", "cxx"}:
        return "c"
    if normalized in {"python", "py"}:
        return "python"
    return "unknown"


def _pool_for_language(language: str) -> Dict[str, Callable[[str, random.Random], str]]:
    if language == "python":
        return PYTHON_LEXICAL_TRANSFORMS
    if language == "c":
        return C_LEXICAL_TRANSFORMS
    return {}


def main() -> None:
    args = parse_args()
    data = load_dataset(args.input)
    rng = random.Random(args.seed)
    variants_per_sample = max(1, args.variants_per_sample)
    max_transforms_per_variant = max(1, args.max_transforms_per_variant)

    language_counters = {"c": 0, "python": 0, "unknown": 0}
    transformed_language_counters = {"c": 0, "python": 0}

    augmented: List[dict] = []
    new_id = next_numeric_id(data)

    for item in data:
        language = _normalize_language(item.get("language", "C"))
        language_counters[language] = language_counters.get(language, 0) + 1
        code = item.get("code", "")

        if not args.drop_original:
            base = deepcopy(item)
            base["is_transformed"] = False
            base["source_id"] = item.get("id")
            base["transformations"] = []
            base["transform_category"] = f"lexical_{language}"
            augmented.append(base)

        transform_pool = _pool_for_language(language)
        enabled_for_language = [name for name in args.transforms if name in transform_pool]
        if not enabled_for_language:
            continue

        for i in range(variants_per_sample):
            transformed_code, applied = transform_code_by_pool(
                code=code,
                enabled_transforms=enabled_for_language,
                transform_pool=transform_pool,
                rng=rng,
                max_transforms_per_variant=max_transforms_per_variant,
            )
            if transformed_code == code or not applied:
                if args.drop_original and args.keep_untransformed:
                    fallback_item = deepcopy(item)
                    fallback_item["id"] = new_id
                    new_id += 1
                    fallback_item["source_id"] = item.get("id")
                    fallback_item["variant_index"] = i + 1
                    fallback_item["is_transformed"] = False
                    fallback_item["transform_fallback"] = True
                    fallback_item["transform_category"] = f"lexical_{language}"
                    fallback_item["transformations"] = []
                    fallback_item["code"] = code
                    augmented.append(fallback_item)
                continue

            new_item = deepcopy(item)
            new_item["id"] = new_id
            new_id += 1
            new_item["source_id"] = item.get("id")
            new_item["variant_index"] = i + 1
            new_item["is_transformed"] = True
            new_item["transform_category"] = f"lexical_{language}"
            new_item["transformations"] = applied
            new_item["code"] = transformed_code
            augmented.append(new_item)
            if language in transformed_language_counters:
                transformed_language_counters[language] += 1

    save_dataset(args.output, augmented)
    transformed_count = sum(1 for x in augmented if x.get("is_transformed"))
    fallback_count = sum(1 for x in augmented if x.get("transform_fallback"))
    print(f"Input samples: {len(data)}")
    print(
        f"Language distribution: C={language_counters.get('c', 0)}, "
        f"Python={language_counters.get('python', 0)}, Unknown={language_counters.get('unknown', 0)}"
    )
    print(f"Output samples: {len(augmented)}")
    print(f"Transformed samples: {transformed_count}")
    print(f"Fallback samples: {fallback_count}")
    print(
        f"Transformed by language: C={transformed_language_counters.get('c', 0)}, "
        f"Python={transformed_language_counters.get('python', 0)}"
    )
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
