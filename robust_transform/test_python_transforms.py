import random
import unittest

from python_transforms import obfuscate_local_variables, obfuscate_parameters


class PythonTransformTests(unittest.TestCase):
    def test_noop_parameter_obfuscation_keeps_source(self) -> None:
        code = """def f(self):
    # keep this comment
    return 1
"""
        output = obfuscate_parameters(code, random.Random(0))
        self.assertEqual(output, code)

    def test_name_obfuscation_is_seed_deterministic(self) -> None:
        code = """def f(left, right):
    total = left + right
    return total
"""
        out1 = obfuscate_local_variables(code, random.Random(1234))
        out2 = obfuscate_local_variables(code, random.Random(1234))
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
