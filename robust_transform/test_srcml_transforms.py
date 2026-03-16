import os
import random
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srcml_transforms import ALL_TRANSFORMS
from transform_utils import STRICT_MUTATION_ENV


class SrcmlTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_strict = os.environ.get(STRICT_MUTATION_ENV)
        os.environ[STRICT_MUTATION_ENV] = "1"

        if shutil.which("srcml") is None:
            raise unittest.SkipTest("srcml is required for robust_transform tests")
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is required for robust_transform tests")

        cls.cases = [
            {
                "name": "obfuscate_local_variables",
                "seed": 0,
                "code": """
int f(int a) {
    int x = 1;
    int y = x + a;
    return y;
}
""",
                "contains": ["return "],
                "not_contains": ["int x = 1;", "int y = x + a;"],
            },
            {
                "name": "obfuscate_parameters",
                "seed": 0,
                "code": """
int add(int left, int right) {
    int sum = left + right;
    return sum;
}
""",
                "contains": ["int sum = "],
                "not_contains": ["left", "right"],
            },
            {
                "name": "obfuscate_local_identifiers",
                "seed": 0,
                "code": """
int add(int left, int right) {
    int sum = left + right;
    return sum;
}
""",
                "contains": ["return "],
                "not_contains": ["left", "right"],
            },
            {
                "name": "swap_symmetric_comparisons",
                "seed": 1,
                "code": """
int f(int x, int y) {
    if (x == y) return 1;
    return 0;
}
""",
                "contains": ["if (y == x)"],
            },
            {
                "name": "append_identity_condition",
                "seed": 0,
                "code": """
int f(int x, int y) {
    if (x < y) return 1;
    return 0;
}
""",
                "regex": [r"if\s*\(x < y\s*(\|\| 0|&& 1)\)"],
            },
            {
                "name": "wrap_numeric_literals_identity",
                "seed": 1,
                "code": """
int f(void) {
    return 7;
}
""",
                "regex": [r"return\s*\(\s*7\s*[\+\-\*]\s*[01]\s*\);"],
            },
            {
                "name": "wrap_numeric_literals_parentheses",
                "seed": 1,
                "code": """
int f(void) {
    return 7;
}
""",
                "regex": [r"return\s*\(7\);"],
            },
            {
                "name": "convert_for_to_while",
                "seed": 0,
                "code": """
int f(void) {
    int acc = 0;
    for (int i = 0; i < 3; i++) {
        acc += i;
    }
    return acc;
}
""",
                "regex": [r"while\s*\(i < 3\)"],
                "not_contains": ["for ("],
            },
            {
                "name": "wrap_statements_if_true",
                "seed": 1,
                "code": """
int f(int x) {
    x++;
    return x;
}
""",
                "regex": [r"if\s*\(1\)"],
            },
            {
                "name": "inject_dead_code",
                "seed": 0,
                "code": """
int f(int x) {
    x++;
    return x;
}
""",
                "regex": [r"if\s*\(0\)", r"dead_1"],
            },
            {
                "name": "extract_call_argument_literals",
                "seed": 0,
                "code": """
#include <stdio.h>
int f(void) {
    printf("%d %d\\n", 7, 9);
    return 0;
}
""",
                "contains": ["const char*", "printf("],
                "not_contains": ['printf("%d %d\\n", 7, 9);'],
                "regex": [r"int\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*7;", r"int\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*9;"],
            },
            {
                "name": "split_declaration_and_initialization",
                "seed": 0,
                "code": """
int f(void) {
    int x = 3;
    return x;
}
""",
                "regex": [r"int x\s*;", r"x = 3;"],
                "not_contains": ["int x = 3;"],
            },
            {
                "name": "encapsulate_literals_with_helpers",
                "seed": 0,
                "code": """
int f(void) {
    int x = 3;
    return x + 7;
}
""",
                "contains": ["get_hardcode_1"],
                "regex": [r"return x \+ get_hardcode_\d+\(\);"],
            },
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old_strict is None:
            os.environ.pop(STRICT_MUTATION_ENV, None)
        else:
            os.environ[STRICT_MUTATION_ENV] = cls._old_strict

    def _compile_c(self, code: str) -> None:
        proc = subprocess.run(
            ["gcc", "-x", "c", "-fsyntax-only", "-"],
            input=code.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=proc.stderr.decode("utf-8", errors="ignore"),
        )

    def test_transform_cases(self) -> None:
        for case in self.cases:
            with self.subTest(transform=case["name"]):
                output = ALL_TRANSFORMS[case["name"]](case["code"], random.Random(case["seed"]))
                self.assertNotEqual(output, case["code"], "transform did not change the input")
                self._compile_c(output)

                for text in case.get("contains", []):
                    self.assertIn(text, output)
                for text in case.get("not_contains", []):
                    self.assertNotIn(text, output)
                for pattern in case.get("regex", []):
                    self.assertRegex(output, re.compile(pattern))

    def test_name_based_transform_is_seed_deterministic(self) -> None:
        code = """
int add(int left, int right) {
    int sum = left + right;
    return sum;
}
"""
        out1 = ALL_TRANSFORMS["obfuscate_parameters"](code, random.Random(1234))
        out2 = ALL_TRANSFORMS["obfuscate_parameters"](code, random.Random(1234))
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
