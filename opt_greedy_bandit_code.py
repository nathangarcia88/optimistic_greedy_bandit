import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from joblib import Parallel, delayed
from typing import Tuple
import os

def alpha(n, alpha_a, alpha_b):
    # Calculate the step size alpha
    return alpha_a / (1 + n) ** alpha_b

def getAction(Q, k, eps):
    # Select an action using epsilon-greedy strategy
    if np.random.rand() > eps:
        return np.argmax(Q)
    else:
        return np.random.randint(k)

def getReward(a, qstar):
    # Get the reward from the chosen bandit arm
    return np.random.normal(loc=qstar[a])

def updateQconstep(Q, r, a, alpha):
    # Update the Q-value using a constant step size (alpha)
    Q[a] += alpha * (r - Q[a])
    return Q

def simpleBandit(k, T, qstar, eps, qinit, alpha):
    # Initialize Q-values optimistically
    Q = qinit * np.ones(k)
    # Initialize action counts
    N = np.zeros(k, dtype=int)
    # Initialize arrays to store rewards and optimal actions
    R = np.zeros(T)
    Nopt = np.zeros(T, dtype=int)

    # Identify the optimal action (bandit with the highest true mean reward)
    astar = np.argmax(qstar)
    for t in range(T):
        # Select an action using epsilon-greedy strategy
        a = getAction(Q, k, eps)
        if a == astar:
            # Count optimal action
            Nopt[t] = 1
        # Get the reward from the chosen action
        r = getReward(a, qstar)
        # Store the reward
        R[t] = r
        # Update the action count
        N[a] += 1
        # Update the Q-value using the constant step size
        Q = updateQconstep(Q, r, a, alpha(t))
    return R, Nopt

def executeDesignRun(run: int, factors: np.ndarray, num_reps: int, qinit, alpha_a, alpha_b, qstar, num_steps_per_rep) -> Tuple[int, np.ndarray, np.ndarray]:
    # Seed the random number generator for reproducibility
    np.random.seed(run)
    k = int(factors[0])
    eps = factors[1]
    # Initialize arrays to store rewards and optimal actions for each repetition
    R_table = np.zeros((num_reps, num_steps_per_rep))
    Nopt_table = np.zeros((num_reps, num_steps_per_rep))

    for rep in range(num_reps):
        # Run the bandit algorithm for each repetition
        R, Nopt = simpleBandit(k, num_steps_per_rep, qstar, eps, qinit, lambda n: alpha(n, alpha_a, alpha_b))
        R_table[rep, :] = R
        Nopt_table[rep, :] = Nopt
    return run, R_table, Nopt_table

def optimistic_greedy(alpha_a, alpha_b, qinit, dataset_config, qstar, num_steps_per_rep, num_reps_per_run, dataset_name):
    tic = time.perf_counter()
    NUM_CPU_CORE_PROCS = os.cpu_count() - 2

    factor_bandits, factor_eps = dataset_config
    factor_table = np.array(list(itertools.product(factor_bandits, factor_eps)))
    num_DOE_runs = factor_table.shape[0]

    meanR = np.zeros((num_DOE_runs, num_steps_per_rep))
    meanNopt = np.zeros((num_DOE_runs, num_steps_per_rep))

    print(f"\nInitializing experiment with {num_DOE_runs} design runs...")
    experiment_start_time = time.perf_counter()

    parallel_manager = Parallel(n_jobs=NUM_CPU_CORE_PROCS)
    DOE_run_list = [delayed(executeDesignRun)(run_index, factor_table[run_index], num_reps_per_run, qinit, alpha_a, alpha_b, qstar, num_steps_per_rep) for run_index in range(num_DOE_runs)]
    results_table = parallel_manager(DOE_run_list)
    results_table = np.array(results_table, dtype=object)
    print(f"Completed experiment ({time.perf_counter() - experiment_start_time:.3f}s)")

    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    
    # Generate x-axis ticks from 0 to num_steps_per_rep with 8 intervals
    xticks = np.linspace(0, num_steps_per_rep, 9, dtype=int)

    # Plotting average reward
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'r', 'g', 'c']

    for run in range(num_DOE_runs):
        eps = factor_table[run, 1]
        meanR = results_table[run, 1].mean(axis=0)
        ax.plot(meanR, color=colors[run % len(colors)], label=f'eps = {eps}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_ylim([-5, 5])
    ax.set_title(f"Average Reward for {dataset_name}\nqinit={qinit}, alpha_a={alpha_a}, alpha_b={alpha_b}")
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.savefig(os.path.join(downloads_path, f'average_reward_{dataset_name}.png'))

    # Plotting optimal action
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'r', 'g', 'c']

    for run in range(num_DOE_runs):
        eps = factor_table[run, 1]
        meanNopt = results_table[run, 2].mean(axis=0)
        ax.plot(meanNopt, color=colors[run % len(colors)], label=f'eps = {eps}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Optimal Action (%)', rotation='vertical', labelpad=25)
    ax.set_xticks(xticks)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_ylim([0, 1])
    ax.set_title(f"Optimal Action for {dataset_name}\nqinit={qinit}, alpha_a={alpha_a}, alpha_b={alpha_b}")
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.savefig(os.path.join(downloads_path, f'optimal_action_{dataset_name}.png'))

# Load the datasets
load_path = os.path.join(os.path.expanduser('~'), 'Downloads')

qstar_list = [
    np.load(os.path.join(load_path, 'dataset1.npy')),
    np.load(os.path.join(load_path, 'dataset2.npy')),
    np.load(os.path.join(load_path, 'dataset3.npy'))
]

datasets = [
    ([10], [0, 0.1, 0.2, 0.4]),
    ([20], [0, 0.1, 0.3]),
    ([5], [0, 0.05, 0.15, 0.25])
]

dataset_names = ['dataset1', 'dataset2', 'dataset3']

# Parameters for the algorithm (get user inputs)
# recommended 0.05, 0.1, 0.2
alpha_a = float(input("Enter the value for alpha_a: "))

# recommended 0.4, 0.5, 0.6
alpha_b = float(input("Enter the value for alpha_b: "))

# recommended -1, 0, 1
qinit = float(input("Enter the initial value for Q-values (qinit): "))

# recommended 100, 200, 300 for initial test runs that go quick for more accurate longer runs try 1000, 1500, 2000
num_steps_per_rep = int(input("Enter the number of steps per repetition: "))
num_reps_per_run = int(input("Enter the number of repetitions per run: "))

# Run the experiment for each dataset configuration
for dataset_config, qstar, dataset_name in zip(datasets, qstar_list, dataset_names):
    optimistic_greedy(alpha_a, alpha_b, qinit, dataset_config, qstar, num_steps_per_rep, num_reps_per_run, dataset_name)

