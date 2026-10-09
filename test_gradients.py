import json
import math
import unittest
from pathlib import Path

# Load only the Value class; tests do not require notebook or plotting packages.
source = "".join(
    next(
        cell["source"]
        for cell in json.loads((Path(__file__).parent / "backprop.ipynb").read_text())[
            "cells"
        ]
        if "class Value:" in "".join(cell["source"])
    )
)
namespace = {"math": math}
exec(source[source.index("class Value:") : source.index("from graphviz")], namespace)
Value = namespace["Value"]


class GradientTests(unittest.TestCase):
    def test_division_matches_finite_differences(self):
        x, y = Value(2.0), Value(3.0)
        z = x / y + 4 / x
        z.backward()
        eps = 1e-6
        f = lambda a, b: a / b + 4 / a
        self.assertAlmostEqual(
            x.grad, (f(2 + eps, 3) - f(2 - eps, 3)) / (2 * eps), places=6
        )
        self.assertAlmostEqual(
            y.grad, (f(2, 3 + eps) - f(2, 3 - eps)) / (2 * eps), places=6
        )

    def test_subtraction(self):
        x = Value(2)
        z = x - 5
        z.backward()
        self.assertEqual(z.data, -3)
        self.assertEqual(x.grad, 1)

    def test_reverse_subtraction(self):
        x = Value(2)
        z = 5 - x
        z.backward()
        self.assertEqual(z.data, 3)
        self.assertEqual(x.grad, -1)

    def test_shared_node_accumulates_gradients(self):
        x = Value(3)
        z = x * x + x
        z.backward()
        self.assertEqual(x.grad, 7)

    def test_tanh_large_input_is_finite(self):
        x = Value(1000)
        z = x.tanh()
        z.backward()
        self.assertEqual(z.data, 1)
        self.assertTrue(math.isfinite(x.grad))


if __name__ == "__main__":
    unittest.main()
