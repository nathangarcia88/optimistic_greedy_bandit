import unittest
from bandit_algorithm import simpleBandit, getTestbed

class TestBanditAlgorithm(unittest.TestCase):
    def test_getTestbed(self):
        k = 10
        testbed = getTestbed(k)
        self.assertEqual(len(testbed), k)

    def test_simpleBandit(self):
        k = 10
        T = 1000
        qinit = 1.6
        alpha = 0.1
        qstar = getTestbed(k)
        eps = 0.1
        R, Nopt = simpleBandit(k, T, qstar, eps, qinit, alpha)
        self.assertEqual(len(R), T)
        self.assertEqual(len(Nopt), T)

if __name__ == '__main__':
    unittest.main()
