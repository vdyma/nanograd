# Tests for NN interface
# Written with GitHub Copilot

import unittest

from src.nanograd.value import Value


class TestValue(unittest.TestCase):
    def test_init(self):
        x = Value(2, (), "-", "x")
        self.assertEqual(x.data, 2)
        self.assertEqual(x.grad, 0)
        self.assertEqual(len(x.children), 0)
        self.assertEqual(x.operator, "-")
        self.assertEqual(x.label, "x")

    def test_mul_pos(self):
        x = Value(2)
        y = Value(3)
        z = x * y
        z.backward()
        self.assertEqual(z.data, 6)
        self.assertEqual(x.grad, 3)
        self.assertEqual(y.grad, 2)

    def test_mul_neg(self):
        x = Value(-2)
        y = Value(-3)
        z = x * y
        z.backward()
        self.assertEqual(z.data, 6)
        self.assertEqual(x.grad, -3)
        self.assertEqual(y.grad, -2)

    def test_add_pos(self):
        x = Value(2)
        y = Value(3)
        z = x + y
        z.backward()
        self.assertEqual(z.data, 5)
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_add_neg(self):
        x = Value(-2)
        y = Value(-3)
        z = x + y
        z.backward()
        self.assertEqual(z.data, -5)
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_add_zero(self):
        x = Value(0)
        y = Value(0)
        z = x + y
        z.backward()
        self.assertEqual(z.data, 0)
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_pow_pos(self):
        x = Value(2)
        y = x**3
        y.backward()
        self.assertEqual(y.data, 8)
        self.assertEqual(x.grad, 12)

    def test_pow_neg(self):
        x = Value(2)
        y = x**-3
        y.backward()
        expected_data = 0.125
        expected_gradient = -0.187
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_pow_zero(self):
        x = Value(2)
        y = x**0
        y.backward()
        self.assertEqual(y.data, 1)
        self.assertEqual(x.grad, 0)

    def test_exp_pos(self):
        x = Value(2)
        y = x.exp()
        y.backward()
        expected_data = 7.389
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_data, x.grad, 2)

    def test_exp_neg(self):
        x = Value(-2)
        y = x.exp()
        y.backward()
        expected_data = 0.135
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_data, x.grad, 2)

    def test_exp_zero(self):
        x = Value(0)
        y = x.exp()
        y.backward()
        expected_data = 1
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_data, x.grad, 2)

    def test_log_pos(self):
        x = Value(2)
        y = x.log()
        y.backward()
        expected_data = 0.693
        expected_gradient = 0.500
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_log_pos_base(self):
        x = Value(2)
        y = x.log(10)
        y.backward()
        expected_data = 0.301
        expected_gradient = 0.217
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_tanh_pos(self):
        x = Value(2)
        y = x.tanh()
        y.backward()
        expected_data = 0.964
        expected_gradient = 0.070
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_tanh_neg(self):
        x = Value(-2)
        y = x.tanh()
        y.backward()
        expected_data = -0.964
        expected_gradient = 0.070
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_tanh_zero(self):
        x = Value(0)
        y = x.tanh()
        y.backward()
        expected_data = 0
        expected_gradient = 1
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_relu_pos(self):
        x = Value(2)
        y = x.relu()
        y.backward()
        self.assertEqual(y.data, 2)
        self.assertEqual(x.grad, 1)

        x = Value(-2)
        y = x.relu()
        y.backward()
        self.assertEqual(y.data, 0)
        self.assertEqual(x.grad, 0)

    def test_relu_neg(self):
        x = Value(-2)
        y = x.relu()
        y.backward()
        self.assertEqual(y.data, 0)
        self.assertEqual(x.grad, 0)

    def test_relu_zero(self):
        x = Value(0)
        y = x.relu()
        y.backward()
        self.assertEqual(y.data, 0)
        self.assertEqual(x.grad, 0)

    def test_sigmoid_pos(self):
        x = Value(2)
        y = x.sigmoid()
        y.backward()
        expected_data = 0.880
        expected_gradient = 0.104
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

        x = Value(-2)
        y = x.sigmoid()
        y.backward()
        expected_data = 0.119
        expected_gradient = 0.104
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_sigmoid_neg(self):
        x = Value(-2)
        y = x.sigmoid()
        y.backward()
        expected_data = 0.119
        expected_gradient = 0.104
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_sigmoid_zero(self):
        x = Value(0)
        y = x.sigmoid()
        y.backward()
        expected_data = 0.5
        expected_gradeint = 0.25
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradeint, x.grad, 2)

    def test_rpow(self):
        x = Value(2)
        y = 3**x
        y.backward()
        expected_data = 9
        expected_gradient = 9.887
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_rpow_zero(self):
        x = Value(0)
        y = 3**x
        y.backward()
        expected_data = 1
        expected_gradient = 1.098
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_rpow_neg(self):
        x = Value(-2)
        y = 3**x
        y.backward()
        expected_data = 0.111
        expected_gradient = 0.122
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_cos_pos(self):
        x = Value(2)
        y = x.cos()
        y.backward()
        expected_data = -0.416
        expected_gradient = -0.909
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_cos_zero(self):
        x = Value(0)
        y = x.cos()
        y.backward()
        expected_data = 1
        expected_gradient = 0
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_cos_neg(self):
        x = Value(-2)
        y = x.cos()
        y.backward()
        expected_data = -0.416
        expected_gradient = 0.909
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_sin_pos(self):
        x = Value(2)
        y = x.sin()
        y.backward()
        expected_data = 0.909
        expected_gradient = -0.416
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_sin_zero(self):
        x = Value(0)
        y = x.sin()
        y.backward()
        expected_data = 0
        expected_gradient = 1
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_sin_neg(self):
        x = Value(-2)
        y = x.sin()
        y.backward()
        expected_data = -0.909
        expected_gradient = -0.416
        self.assertAlmostEqual(expected_data, y.data, 2)
        self.assertAlmostEqual(expected_gradient, x.grad, 2)

    def test_visualization(self):
        x = Value(2)
        y = x.sigmoid()
        y.backward()
        y.visualize()


if __name__ == "__main__":
    unittest.main()
