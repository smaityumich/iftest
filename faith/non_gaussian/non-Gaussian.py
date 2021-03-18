#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:33:36 2019

@author: sxue
@file_name: non-Gaussian.py
"""

# non-Gaussian example

import pulp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Chunks in this code:
# 1. To get the plot of empirical distribution
# 2. To get the plot of limiting distribution
# 3. To get the plot putting empirical and limiting distribution together
# 4. Q-Q plot
# 5. Bootstrap distribution vs. empirical distribution


#------------------------------------------------------------------------------
#-- 1. To get the plot of empirical distribution ------------------------------
#------------------------------------------------------------------------------

# K (dimension of feature space) ----------------------------------------------
K = 3

# epsilon (transportation cost control) ---------------------------------------
eps = 1 + 1/3**0.5

# X (feature space) -----------------------------------------------------------
# Not applied here

# C (cost matrix) -------------------------------------------------------------
C = [[0, 1, 2], [1, 0, 3**0.5], [2, 3**0.5, 0]]
    
# l (vector of losses) --------------------------------------------------------
l = list(range(1, K+1))

# f_true (true distribution on space X) ---------------------------------------
f_true = [x/6 for x in l[::-1]]

# n ---------------------------------------------------------------------------
n = 1000

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
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_true[_] for _ in range(K)]))
# True optimal value = 1.33333... = 4/3

# Do B times ------------------------------------------------------------------
psi_list = []
B = 100000
np.random.seed(2019)
for _ in range(B):
    f_n = (np.random.multinomial(n, f_true, size = 1)/n).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_n[i]
        
    prob.solve()
    psi_list.append(pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)]))

# Plot ------------------------------------------------------------------------
sns.distplot([n**0.5 * (x - 4/3) for x in psi_list], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})
plt.title('Empirical distribution ($n = 1000$)')
plt.savefig("Empirical_dist.pdf")
plt.show()


#------------------------------------------------------------------------------
#-- 2. To get the plot of limiting distribution -------------------------------
#------------------------------------------------------------------------------

Mean = np.array([0, 0, 0])
Cov = np.array([[1/4, -1/6, -1/12], [-1/6, 2/9, -1/18], [-1/12, -1/18, 5/36]])
zeta1 = np.array([-2, -1, 0])
zeta2 = np.array([-2+2/np.sqrt(3), 0, 0])

# Do B times ------------------------------------------------------------------
collection = []
B = 100000
np.random.seed(2019)

for _ in range(B):
    Z = np.random.multivariate_normal(Mean, Cov)
    value1 = np.inner(Z, zeta1)
    value2 = np.inner(Z, zeta2)
    collection.append(np.min([value1, value2]))
    
# Plot ------------------------------------------------------------------------
sns.distplot(collection, hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})
plt.title('Limiting distribution')
plt.savefig("Limit_dist.pdf")
plt.show()


#------------------------------------------------------------------------------
#-- 3. To get the plot putting empirical and limiting distribution together ---
#------------------------------------------------------------------------------

sns.distplot([n**0.5 * (x - 4/3) for x in psi_list], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 2}, label = "empirical distribution")
sns.distplot(collection, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 2}, label = "limiting distribution")
plt.legend(loc='upper left')
plt.title('Empirical vs. Limiting distribution ($n = 1000$)')
plt.savefig("Distribution_matched.pdf")
plt.show()


#------------------------------------------------------------------------------
#-- 4. Q-Q plot ---------------------------------------------------------------
#------------------------------------------------------------------------------

plt.figure()
plt.scatter(np.sort([n**0.5 * (x - 4/3) for x in psi_list]), np.sort(collection), s=0.1)
plt.xlabel("quantiles of empirical distribution")
plt.ylabel('quantiles of limiting distribution')
plt.axis('equal')
plt.title("Q-Q plot ($n = 1000$)")
plt.savefig("QQ_plot.pdf")
plt.show()


#------------------------------------------------------------------------------
#-- 5. Bootstrap distribution vs. empirical distribution ----------------------
#------------------------------------------------------------------------------

# Observed f_n
np.random.seed(2020)
f_obs = (np.random.multinomial(n, f_true, size = 1)/n).tolist()[0]

# Test statistic
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_obs[i]
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

test_statistic = pulp.value(prob.objective) - sum([l[_]*f_true[_] for _ in range(K)])

# Bootstrap sample size
m = 2 * int(np.sqrt(K*n))

# Bootstrap, do B times
psi_list_boot = []
B = 100000
np.random.seed(2019)
for _ in range(B):
    f_boot = (np.random.multinomial(m, f_obs, size = 1)/m).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
    prob.solve()
    psi_list_boot.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))

# Plot 1: density plots
sns.distplot([n**0.5 * (x - 4/3) for x in psi_list], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 2}, label = "empirical distribution")
sns.distplot([m**0.5 * (x - test_statistic) for x in psi_list_boot], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 2}, label = "bootstrap distribution")
plt.legend(loc='upper left')
plt.title('KDE: Bootstrap vs. empirical ($n = 1000, m =  2\sqrt{Kn} = 108$)')
plt.savefig("KDE_bootstrap_empirical.pdf")
plt.show()

# Plot 2: Q-Q plot
plt.figure()
plt.scatter(np.sort([n**0.5 * (x - 4/3) for x in psi_list]),
            np.sort([m**0.5 * (x - test_statistic) for x in psi_list_boot]),
            s=0.1)
plt.xlabel("quantiles of empirical distribution")
plt.ylabel('quantiles of bootstrap distribution')
plt.axis('equal')
plt.title("Q-Q: Bootstrap vs. empirical ($n = 1000, m = 2\sqrt{Kn} = 108$)")
plt.savefig("QQ_bootstrap_empirical.pdf")
plt.show()

