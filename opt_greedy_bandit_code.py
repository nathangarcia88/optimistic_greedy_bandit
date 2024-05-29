# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:04:39 2024

@author: 19258
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from joblib import Parallel, delayed
from typing import Tuple
import os

# Define num_steps_per_rep globally
num_steps_per_rep = 1000
num_reps_per_run = 2000

def alpha(n, alpha_a, alpha_b):
    return alpha_a / (1 + n) ** alpha_b

def getTestbed(k):
    return np.random.normal(size=k)

def getAction(Q, k, eps):
    if np.random.rand() > eps:
        return np.argmax(Q)
    else:
        return np.random.randint(k)

def getReward(a, qstar):
    return np.random.normal(loc=qstar[a])

def updateQ(Q, r, a, n):
    Q[a] += (r - Q[a]) / n
    return Q

def updateQconstep(Q, r, a, alpha):
    Q[a] += alpha * (r - Q[a])
    return Q

def simpleBandit(k, T, qstar, eps, qinit, alpha):
    Q = qinit * np.ones(k)
    N = np.zeros(k, dtype=int)
    R = np.zeros(T)
    Nopt = np.zeros(T, dtype=int)

    astar = np.argmax(qstar)
    for t in range(T):
        a = getAction(Q, k, eps)
        if a == astar:
            Nopt[t] = 1
        r = getReward(a, qstar)
        R[t] = r
        N[a] += 1
        Q = updateQconstep(Q, r, a, alpha)
    return R, Nopt

def executeDesignRun(run: int, factors: np.ndarray, num_reps: int, qinit, alpha) -> Tuple[int, np.ndarray, np.ndarray]:
    np.random.seed(run)
    k = int(factors[0])
    eps = factors[1]
    R_table = np.zeros((num_reps, num_steps_per_rep))
    Nopt_table = np.zeros((num_reps, num_steps_per_rep))

    for rep in range(num_reps):
        qstar = getTestbed(k)
        R, Nopt = simpleBandit(k, num_steps_per_rep, qstar, eps, qinit, alpha)
        R_table[rep, :] = R
        Nopt_table[rep, :] = Nopt
    return run, R_table, Nopt_table

def computeLearningCurves(R_table, Nopt_table):
    meanR = np.mean(R_table, axis=0)
    meanNopt = np.mean(Nopt_table, axis=0)
    return meanR, meanNopt

def optimistic_greedy(alpha_a, alpha_b, qinit):
    tic = time.perf_counter()
    NUM_CPU_CORE_PROCS = os.cpu_count()-2

    factor_bandits = [10]
    factor_eps = [0, 0.1, 0.2, 0.4]
    factor_table = np.array(list(itertools.product(factor_bandits, factor_eps)))
    num_DOE_runs = factor_table.shape[0]

    meanR = np.zeros((num_DOE_runs, num_steps_per_rep))
    meanNopt = np.zeros((num_DOE_runs, num_steps_per_rep))

    print(f"\nInitializing experiment with {num_DOE_runs} design runs...")
    experiment_start_time = time.perf_counter()

    parallel_manager = Parallel(n_jobs=NUM_CPU_CORE_PROCS)
    DOE_run_list = [delayed(executeDesignRun)(run_index, factor_table[run_index], num_reps_per_run, qinit, alpha) for run_index in range(num_DOE_runs)]
    results_table = parallel_manager(DOE_run_list)
    results_table = np.array(results_table, dtype=object)
    print(f"Completed experiment ({time.perf_counter() - experiment_start_time:.3f}s)")

    fig = plt.figure()
    color = ['b', 'r', 'g', 'c']
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for run in range(num_DOE_runs):
        eps = factor_table[run, 1]
        meanR = results_table[run, 1]
        ax.plot(meanR, color[run], label=f'eps = {eps}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_ylim([-1, 1])
    ax.set_title(f"Optimistic eps-greedy Performance\nqinit={qinit}, alpha={alpha}")
    ax.legend(loc='lower right')
    plt.grid()
    plt.show()

    fig = plt.figure()
    color = ['b', 'r', 'g', 'c']
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for run in range(num_DOE_runs):
        eps = factor_table[run, 1]
        meanNopt = results_table[run, 2]
        ax.plot(meanNopt, color[run], label=f'eps = {eps}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Optimal Action', rotation='horizontal', labelpad=25)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_ylim([0, 1])
    ax.set_title(f"Optimistic eps-greedy Performance\nqinit={qinit}, alpha={alpha}")
    ax.legend(loc='lower right')
    plt.grid()
    plt.show()
