import unittest
from opt_greedy_bandit_code import simpleBandit, getTestbed, alpha, getAction, getReward, updateQconstep

class TestBanditAlgorithm(unittest.TestCase):
    # Test for the getTestbed function
    def test_getTestbed(self):
        # Check if getTestbed generates a testbed with the correct number of bandit arms
        k = 10
        testbed = getTestbed(k)
        self.assertEqual(len(testbed), k, "The length of the testbed should be equal to k")

    # Test for the simpleBandit function
    def test_simpleBandit(self):
        # Check if simpleBandit runs correctly and returns arrays of the correct length
        k = 10
        T = 1000
        qinit = 1.6
        alpha_a = 0.1
        alpha_b = 0.5
        qstar = getTestbed(k)
        eps = 0.1
        R, Nopt = simpleBandit(k, T, qstar, eps, qinit, lambda n: alpha(n, alpha_a, alpha_b))
        self.assertEqual(len(R), T, "The length of the rewards array R should be equal to T")
        self.assertEqual(len(Nopt), T, "The length of the optimal action array Nopt should be equal to T")

    # Test for the alpha function
    def test_alpha(self):
        # Check if the alpha function returns the correct step size for given parameters
        n = 10
        alpha_a = 0.1
        alpha_b = 0.5
        result = alpha(n, alpha_a, alpha_b)
        expected = alpha_a / (1 + n) ** alpha_b
        self.assertAlmostEqual(result, expected, places=5, msg="Alpha function did not return the expected value")

    # Test for the getAction function
    def test_getAction(self):
        # Check if getAction selects an action correctly using epsilon-greedy strategy
        Q = [1.0, 2.0, 3.0]
        k = 3
        eps = 0.0  # No exploration, always exploit
        action = getAction(Q, k, eps)
        self.assertEqual(action, 2, "With eps=0, the action should be the one with the highest Q-value")

    # Test for the getReward function
    def test_getReward(self):
        # Check if getReward returns a reward based on the true mean reward of the selected action
        a = 0
        qstar = [1.0, 2.0, 3.0]
        reward = getReward(a, qstar)
        self.assertTrue(isinstance(reward, float), "The reward should be a float")

    # Test for the updateQconstep function
    def test_updateQconstep(self):
        # Check if updateQconstep updates the Q-value correctly using a constant step size
        Q = [1.0, 2.0, 3.0]
        r = 5.0
        a = 1
        alpha = 0.1
        updated_Q = updateQconstep(Q, r, a, alpha)
        expected_Q = 2.3  # 2.0 + 0.1 * (5.0 - 2.0)
        self.assertAlmostEqual(updated_Q[a], expected_Q, places=5, msg="Q-value was not updated correctly")

if __name__ == '__main__':
    unittest.main()
