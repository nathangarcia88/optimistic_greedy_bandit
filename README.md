OPER 651 Individual Algorithm Project
# Bandit Algorithm

## Algorithm Purpose
The Optimistic Greedy Bandit algorithm is designed to solve the multi-armed bandit problem, where the goal is to maximize the cumulative reward by selecting actions (bandit arms) based on estimated rewards. The algorithm optimistically initializes the Q-values to encourage exploration and uses an epsilon-greedy strategy to balance exploration and exploitation. Simple Bandit initializes Q values to 0 so other This type of "trick" works well for stationary problems such as pulling slot machines with a known distribution; however, the algorithm is very ineffective when it comes to non-stationary environments because its drive for exploration is tempoarary. 

## Hyperparameters
- `alpha_a`: Controls the step size parameter's numerator.
- `alpha_b`: Controls the step size parameter's denominator scaling.
  The alpha parameters are combined in an equation to make an alpha, this is really helpful when using a latin hypercube sampling to hypertune.
- `qinit`: The initial optimistic value for the Q-values. Determines how explorative we want to be, the higher the more explorative.
- `eps`: The probability of exploring a random action instead of exploiting the best-known action. They are predetermined in the code for the most likely choices.

## Background
The k-armed bandit problem is a classic reinforcement learning problem where an agent must choose between k different actions (or arms) to maximize its reward. Each arm provides a stochastic reward based on an unknown probability distribution. The challenge is to balance exploration (trying new arms) with exploitation (choosing the best-known arm). Essentially, there is an unknown distribution for each arm and the bandit attempts to get familiar enough with the arms to eventually choose the optimal arm to get the most rewards. This directly relates to our projects of finding the best strategy for slot machines.

## History
The bandit problem has been studied extensively in statistics, economics, and computer science. It is named after slot machines (one-armed bandits) in casinos. The optimistic initial values approach encourages exploration by initially overestimating the value of all actions.

## Variations
- **Simple Bandit:** A simpler version without optimistic initial values.
- **UCB (Upper Confidence Bound):** Uses confidence intervals to balance exploration and exploitation.

## Pseudo Code
```pseudo
Initialize Q-values to qinit for all actions
For each time step t:
    With probability 1 - eps, choose action with highest Q-value
    With probability eps, choose a random action
    Take the action, observe reward r
    Update the Q-value for the action using the reward
```

# Bandit A

## Example Code
Provide example code for importing and using the module.

## Visualization or Animation
![Alt text](rew1.png)
![Alt text](opt1.png)
![Alt text](rew2.png)
![Alt text](opt2.png)

## Benchmark Results
Present benchmark results comparing efficiency and effectiveness.

## Lessons Learned
Share any lessons learned, including new code snippets.

## Unit-Testing Strategy
Explain the unit-testing strategy and what steps of the algorithm were tested.

## Code-Coverage Measurement
Include code-coverage measurement results.
