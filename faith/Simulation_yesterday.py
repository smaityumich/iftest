#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 02:09:12 2019

@author: sxue
"""

import numpy.random
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# K (dimension of feature space) ----------------------------------------------
K = 20

# epsilon (transportation cost control) ---------------------------------------
eps = 10

# X (feature space) -----------------------------------------------------------
X = [[0,0,1], [1,0,1], [0,0,0], [1,0,0],
     [0,1,1], [1,1,1], [0,1,0], [1,1,0],
     [0,2,1], [1,2,1], [0,2,0], [1,2,0],
     [0,3,1], [1,3,1], [0,3,0], [1,3,0],
     [0,4,1], [1,4,1], [0,4,0], [1,4,0]]

# C (cost matrix) -------------------------------------------------------------
C = [[0 for _ in range(K)] for _ in range(K)]
for i in range(K):
    for j in range(K):
        if i == j:
            C[i][j] = 0
        elif X[i][1:] == X[j][1:]:
            C[i][j] = 1
        else:
            C[i][j] = 100

# sigmoid function ------------------------------------------------------------
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# f_true (true distribution on space X) ---------------------------------------
p_race = [0.3, 0.7]
p_score_race = [[0.1, 0.2, 0.4, 0.2, 0.1],
                [0.05, 0.15, 0.45, 0.25, 0.1]]
f_true = [0 for _ in range(K)]
for i in range(K):
    p = sigmoid(X[i][1] - 2.5)
    if X[i][2] == 0:
        p = 1 - p
    f_true[i] = p_race[X[i][0]] * p_score_race[X[i][0]][X[i][1]] * p
sum(f_true)

# l (vector of losses) --------------------------------------------------------
h0 = lambda x: sigmoid(x - 2.5)
l0 = [0 for _ in range(K)]
for i in range(K):
    if X[i][2] == 1:
        l0[i] = 1 - h0(X[i][1])
    else:
        l0[i] = h0(X[i][1])

h1 = lambda x, y: sigmoid(x + 0.5 * y - 2.75)
l1 = [0 for _ in range(K)]
for i in range(K):
    if X[i][2] == 1:
        l1[i] = 1 - h1(X[i][1], X[i][0])
    else:
        l1[i] = h1(X[i][1], X[i][0])
        
# Testing
[f_true[_] * l0[_] for _ in range(K)]
[f_true[_] * l1[_] for _ in range(K)]

# n ---------------------------------------------------------------------------
n = 1000

# -----------------------------------------------------------------------------
# ---- (1) Unfair classifier --------------------------------------------------
# -----------------------------------------------------------------------------
l = l1 #### Please switch between l0 and l1

# Linear programming on f_true ------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_true[i]

for i in range(K):
    for j in range(K):
        if abs(i - j) >= 2:
            prob += Pi[i][j] == 0

prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_true[_] for _ in range(K)]))
# True optimal value = 0.11560255881084397

# Do B times ------------------------------------------------------------------
psi_list_1 = []
B = 10000
np.random.seed(2019)
for _ in range(B):
    f_n = (numpy.random.multinomial(n, f_true, size = 1)/n).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_n[i]
        
    for i in range(K):
        for j in range(K):
            if abs(i - j) >= 2:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_1.append(pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)]))

# Plot ------------------------------------------------------------------------
sns.distplot([n**0.5 * (x - 0.11560255881084397) for x in psi_list_1], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.savefig("Limit_dist_unfair.pdf")
plt.show()

# Bootstrap -------------------------------------------------------------------
f_obs = [0.001, 0.002, 0.023, 0.03, 0.016, 0.016, 0.052, 0.084, 0.038, 0.117,
         0.08, 0.198, 0.032, 0.109, 0.025, 0.064, 0.029, 0.066, 0.008, 0.01]

# Test statistic --------------------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_obs[i]

for i in range(K):
    for j in range(K):
        if abs(i - j) >= 2:
            prob += Pi[i][j] == 0
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_obs[_] for _ in range(K)]))
test_statistic_1 = pulp.value(prob.objective) - sum([l[_]*f_obs[_] for _ in range(K)])

# Start bootstrap -------------------------------------------------------------
m = np.floor(2 * np.sqrt(n))

# Do B times ------------------------------------------------------------------
psi_list_boot_1 = []
B = 10000
np.random.seed(2019)
for _ in range(B):
    f_boot = (numpy.random.multinomial(m, f_obs, size = 1)/m).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
    for i in range(K):
        for j in range(K):
            if abs(i - j) >= 2:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_boot_1.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))

# Plot ------------------------------------------------------------------------
sns.distplot([m**0.5 * (x - test_statistic_1) for x in psi_list_boot_1], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.axvline(x=c_lower_1, color='orange', linestyle='--')
plt.axvline(x=c_upper_1, color='orange', linestyle='--')

plt.savefig("Boot_dist_unfair.pdf")
plt.show()

c_lower_1 = np.percentile([m**0.5 * (x - test_statistic_1) for x in psi_list_boot_1], 2.5)
c_upper_1 = np.percentile([m**0.5 * (x - test_statistic_1) for x in psi_list_boot_1], 97.5)

CI_lower_1 = test_statistic_1 - c_upper_1 / np.sqrt(n) # 0.111290
CI_upper_1 = test_statistic_1 - c_lower_1 / np.sqrt(n) # 0.119176




# -----------------------------------------------------------------------------
# ---- (0) Fair classifier ----------------------------------------------------
# -----------------------------------------------------------------------------
l = l0 #### Please switch between l0 and l1

# Linear programming on f_true ------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_true[i]

for i in range(K):
    for j in range(K):
        if abs(i - j) >= 2:
            prob += Pi[i][j] == 0
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_true[_] for _ in range(K)]))
# True optimal value = 0.07287081170890491

# Do B times ------------------------------------------------------------------
psi_list_0 = []
B = 10000
np.random.seed(2019)
for _ in range(B):
    f_n = (numpy.random.multinomial(n, f_true, size = 1)/n).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_n[i]
        
    for i in range(K):
        for j in range(K):
            if abs(i - j) >= 2:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_0.append(pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)]))

# Plot ------------------------------------------------------------------------
sns.distplot([n**0.5 * (x - 0.07287081170890491) for x in psi_list_0], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.savefig("Limit_dist_fair.pdf")
plt.show()


# Bootstrap -------------------------------------------------------------------
f_obs = [0.001, 0.002, 0.023, 0.03, 0.016, 0.016, 0.052, 0.084, 0.038, 0.117,
         0.08, 0.198, 0.032, 0.109, 0.025, 0.064, 0.029, 0.066, 0.008, 0.01]

# Test statistic --------------------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_obs[i]

for i in range(K):
    for j in range(K):
        if abs(i - j) >= 2:
            prob += Pi[i][j] == 0
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_obs[_] for _ in range(K)]))
test_statistic_2 = pulp.value(prob.objective) - sum([l[_]*f_obs[_] for _ in range(K)])

# Start bootstrap -------------------------------------------------------------
m = np.floor(2 * np.sqrt(n))

# Do B times ------------------------------------------------------------------
psi_list_boot_2 = []
B = 10000
np.random.seed(2019)
for _ in range(B):
    f_boot = (numpy.random.multinomial(m, f_obs, size = 1)/m).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
    for i in range(K):
        for j in range(K):
            if abs(i - j) >= 2:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_boot_2.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))

# Plot ------------------------------------------------------------------------
sns.distplot([m**0.5 * (x - test_statistic_2) for x in psi_list_boot_2], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.axvline(x=c_lower_2, color='orange', linestyle='--')
plt.axvline(x=c_upper_2, color='orange', linestyle='--')

plt.savefig("Boot_dist_fair.pdf")
plt.show()



c_lower_2 = np.percentile([m**0.5 * (x - test_statistic_2) for x in psi_list_boot_2], 2.5)
c_upper_2 = np.percentile([m**0.5 * (x - test_statistic_2) for x in psi_list_boot_2], 97.5)

CI_lower_2 = test_statistic_2 - c_upper_2 / np.sqrt(n) # 0.069274
CI_upper_2 = test_statistic_2 - c_lower_2 / np.sqrt(n) # 0.073808




















