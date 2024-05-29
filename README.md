OPER 651 Individual Algorithm Project
# Bandit Algorithm

## Algorithm Purpose
The Bandit Algorithm is designed to solve the k-armed bandit problem using optimistic initial values and an epsilon-greedy strategy. The goal is to balance exploration and exploitation to maximize the cumulative reward over time.

## Hyperparameters
- `alpha_a`: Controls the step size parameter's numerator.
- `alpha_b`: Controls the step size parameter's denominator scaling.
- `qinit`: The initial optimistic value for the Q-values.
- `eps`: The probability of exploring a random action instead of exploiting the best-known action.

## Background
The k-armed bandit problem is a classic reinforcement learning problem where an agent must choose between k different actions (or arms) to maximize its reward. Each arm provides a stochastic reward based on an unknown probability distribution. The challenge is to balance exploration (trying new arms) with exploitation (choosing the best-known arm).

## History
The bandit problem has been studied extensively in statistics, economics, and computer science. It is named after slot machines (one-armed bandits) in casinos. The optimistic initial values approach encourages exploration by initially overestimating the value of all actions.

## Variations
- **Standard Epsilon-Greedy:** A simpler version without optimistic initial values.
- **UCB (Upper Confidence Bound):** Uses confidence intervals to balance exploration and exploitation.
- **Thompson Sampling:** Uses Bayesian methods to sample actions based on their probability of being optimal.

## Pseudo Code
```pseudo
Initialize Q-values to qinit for all actions
For each time step t:
    With probability 1 - eps, choose action with highest Q-value
    With probability eps, choose a random action
    Take the action, observe reward r
    Update the Q-value for the action using the reward



# Bandit A

## Example Code
Provide example code for importing and using the module.

## Visualization or Animation
Add visualizations or animations of the algorithm's steps or results.

## Benchmark Results
Present benchmark results comparing efficiency and effectiveness.

## Lessons Learned
Share any lessons learned, including new code snippets.

## Unit-Testing Strategy
Explain the unit-testing strategy and what steps of the algorithm were tested.

## Code-Coverage Measurement
Include code-coverage measurement results.
