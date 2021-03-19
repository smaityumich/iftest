#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sxue
@file_name: compas_20190910.py
"""

# Python version: 3.6.5 -------------------------------------------------------

# Tasks -----------------------------------------------------------------------
# Compas dataset (full, do not split train and test)
# Fit logistic regression
# Test fairness based on 3 kinds of metrics:
# (1) changing race is free
# (2) using section 4.2 of SenSR paper: train logistic regression to predict
#     race and allow moving along the direction orthogonal to its decision
#     boundary for free (changing race is also free)
# (3) change in a feature is weighted by 1-abs(corr(feature,race))

# Reference -------------------------------------------------------------------
# [1] https://github.com/IBM/AIF360
# [2] https://github.com/IBM/AIF360/blob/master/examples/demo_reweighing_preproc.ipynb


# -----------------------------------------------------------------------------
# ---- 0. Packages ------------------------------------------------------------
# -----------------------------------------------------------------------------

# 0.1 General
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# 0.2 AIF360
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas


# -----------------------------------------------------------------------------
# ---- 1. Loading COMPAS data -------------------------------------------------
# -----------------------------------------------------------------------------

dataset_used = "compas" # "adult", "german", "compas"
protected_attribute_used = 2 # 1, 2

# protected attribute used is "race"

if dataset_used == "adult":
#     dataset_orig = AdultDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_adult(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_adult(['race'])
    
elif dataset_used == "german":
#     dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_german(['sex'])
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = load_preproc_data_german(['age'])
    
elif dataset_used == "compas":
#     dataset_orig = CompasDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 0}]
        unprivileged_groups = [{'sex': 1}]
        dataset_orig = load_preproc_data_compas(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_compas(['race'])

# use full dataset, do not split train and test
data = dataset_orig

# Shape of design matrix: (5278, 10)
print(data.features.shape)

# Feature names:
print(data.feature_names)
# (0) 'sex'                          a = 0, 1
# (1) 'race'                         b = 0, 1
# (2) 'age_cat=25 to 45'             c1
# (3) 'age_cat=Greater than 45'      c2
# (4) 'age_cat=Less than 25'         c3, c1+c2+c3 = 1
# (5) 'priors_count=0'               d1
# (6) 'priors_count=1 to 3'          d2
# (7) 'priors_count=More than 3'     d3, d1+d2+d3 = 1
# (8) 'c_charge_degree=F'            e1
# (9) 'c_charge_degree=M'            e2, e1+e2 = 1

# Label name: 'two_year_recid'
print(data.label_names)            # f = 0, 1


# -----------------------------------------------------------------------------
# ---- 2. Space X, Y, Z -------------------------------------------------------
# -----------------------------------------------------------------------------

# Space Z
space_Z = []
for a in [[0], [1]]:
    pa = tuple(a)
    for b in [[0], [1]]:
        pb = pa + tuple(b)
        for c in [(1,0,0), (0,1,0), (0,0,1)]:
            pc = pb + c
            for d in [(1,0,0), (0,1,0), (0,0,1)]:
                pd = pc + d
                for e in [(1,0), (0,1)]:
                    pe = pd + e
                    for f in [[0], [1]]:
                        data_point = pe + tuple(f)
                        space_Z.append(data_point)
                        
# Dimension of space Z: 144
print(len(space_Z))
K = len(space_Z)

# Frequency (f_n, empirical distribution)
f_n = list(0 for _ in range(K))
Z_dict = dict(zip(space_Z, list(range(K))))

for _ in range(len(data.labels)):
    z = tuple(data.features[_]) + tuple(data.labels[_])
    #z = tuple(int(x) for x in z)
    f_n[Z_dict[z]] += 1

print(sum(f_n)) # 5278

# Normalize to sum 1
f_n = [x/5278 for x in f_n]

# Barplot of f_n, from high to low
plt.bar(list(range(K)), sorted(f_n, reverse = True), width=1)
plt.title("Empirical distribution $f_n$ (sorted from high to low)")
plt.xlabel("Space $Z$")
plt.ylabel("Frequency")
#plt.savefig("empirical_distribution.pdf")
plt.show()

# Space X
space_X = [list(space_Z[2*i][:10]) for i in range(int(K/2))]
print(len(space_X))


# -----------------------------------------------------------------------------
# ---- 3. Loss vector l -------------------------------------------------------
# -----------------------------------------------------------------------------

# Remark: To get $l$, we fit logistic regression in this section

# Logistic regression classifier and predictions
X_train = data.features
y_train = data.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train)

y_pred = lmod.predict(space_X)

l = list(0 for _ in range(K))
for _ in range(int(K/2)):
    z = tuple(space_X[_]) + tuple([1 - y_pred[_]])
    l[Z_dict[z]] = 1
    
print(sum(l)) # 72


# -----------------------------------------------------------------------------
# ---- 4. Metric (1) ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Metric (1): changing race is free
#
# Specification:
# 1. The cost of a, b in Z, is equal to the number of features which are
#    not agree with each other on the same position of vector a and b,
#    excluding "race".
# 2. To change label is not allowed.
#
# Example:
# a = space_Z[10] = (0, "0", 1, 0, 0, 0, 0, 1, 0, 1, 0)
# b = space_Z[48] = (0, "1", 0, 1, 0, 1, 0, 0, 1, 0, 0)
# C[10][48] = 6
# where "0" and "1" are "race", should be excluded


# C (cost matrix) -------------------------------------------------------------
C = [[0 for _ in range(K)] for _ in range(K)]
for i in range(K):
    for j in range(K):
        a = space_Z[i]
        b = space_Z[j]
        if a[10] != b[10]:
            C[i][j] = 1000    # 1000 will be set to infinity in optimization
        else:
            a = [a[0]] + list(a[2:])    # excluding "race"
            b = [b[0]] + list(b[2:])
            C[i][j] = sum(a[_] != b[_] for _ in range(len(a)))
            
# epsilon (transportation cost control) ---------------------------------------
eps = 0

# sigmoid function ------------------------------------------------------------
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Test statistic --------------------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_n[i]

for i in range(K):
    for j in range(K):
        if C[i][j] == 1000:
            prob += Pi[i][j] == 0
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)]))
test_statistic = pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)])
            
# sample size n ---------------------------------------------------------------
n = len(data.labels)  # 5278

# Bootstrap sample size m -----------------------------------------------------
m = np.floor(2 * np.sqrt(K) * np.sqrt(n))  #  1743

# Do B times ------------------------------------------------------------------
psi_list_boot = []
B = 1000
np.random.seed(2019)
for _ in range(B):
    f_boot = (np.random.multinomial(m, f_n, size = 1)/m).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
    for i in range(K):
        for j in range(K):
            if C[i][j] == 1000:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_boot.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))
    
    if _ % 20 == 0:
        print(_)

for _ in range(B):  # Due to computational error, we do this correction
    if psi_list_boot[_] < 1e-8:
        psi_list_boot[_] = 0
        
# Plot ------------------------------------------------------------------------
c_lower = np.percentile([m**0.5 * (x - test_statistic) for x in psi_list_boot], 2.5)
c_upper = np.percentile([m**0.5 * (x - test_statistic) for x in psi_list_boot], 97.5)

CI_lower = test_statistic - c_upper / np.sqrt(n) # 0.001210 (CI lower bound)
CI_upper = test_statistic - c_lower / np.sqrt(n) # 0.003847 (CI upper bound)
                                                 # 0.002653 (Test statistic)

sns.distplot([m**0.5 * (x - test_statistic) for x in psi_list_boot], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.axvline(x=c_lower, color='orange', linestyle='--')
plt.axvline(x=c_upper, color='orange', linestyle='--')
#plt.savefig("boot_dist_1_m-out-of-n.pdf")
plt.show()

# ---- NEW ! ------------------------------------------------------------------
# Bootstrap strategy 2: numerical derivative ----------------------------------
# ---- NEW ! ------------------------------------------------------------------

def cov_Sigma(f):
    K = len(f)
    cov = [[0 for _ in range(K)] for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                cov[i][j] = f[i] * (1 - f[i])
            else:
                cov[i][j] = - f[i] * f[j]
    return cov

# How to generate a multivariate normal with mean $f_n$ and covariance $\Sigma(f_n)$
Mean = f_n
Cov = cov_Sigma(Mean)
Z = np.random.multivariate_normal(Mean, Cov)

# Numerical derivative step size: epsilon
# epsilon = (n * K) ** -0.25 # 0.033868, acceptance rate is too low
epsilon = 0.001
epsilon_inv = 1000

# Do B times ------------------------------------------------------------------
psi_list_boot = []
B = 1000
count = 0
np.random.seed(2019)

while count < B:
    Z = np.random.multivariate_normal(Mean, Cov)
    f_boot = np.add(f_n, np.multiply(epsilon, Z))
    if not np.all(f_boot >= 0):
        continue
    else:
        count += 1
        
        prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

        Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

        prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

        prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

        for i in range(K):
            prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
        for i in range(K):
            for j in range(K):
                if C[i][j] == 1000:
                    prob += Pi[i][j] == 0
            
        prob.solve()
        psi_list_boot.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))
        
        if count % 20 == 0:
            print(count)
        
# Plot ------------------------------------------------------------------------
c_lower = np.percentile([epsilon_inv * (x - test_statistic) for x in psi_list_boot], 2.5)
c_upper = np.percentile([epsilon_inv * (x - test_statistic) for x in psi_list_boot], 97.5)

CI_lower = test_statistic - c_upper / np.sqrt(n) # 0.001281 (CI lower bound)
CI_upper = test_statistic - c_lower / np.sqrt(n) # 0.004020 (CI upper bound)
                                                 # 0.002653 (Test statistic)

sns.distplot([epsilon_inv * (x - test_statistic) for x in psi_list_boot], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.axvline(x=c_lower, color='orange', linestyle='--')
plt.axvline(x=c_upper, color='orange', linestyle='--')
#plt.savefig("boot_dist_1_numerical_derivative.pdf")
plt.show()

## Acceptance rate
#test = []
#for _ in range(5000):
#    epsilon= 0.001
#    Z = np.random.multivariate_normal(Mean, Cov)
#    a = np.add(f_n, np.multiply(epsilon, Z))
#    test.append(np.all(a>=0))
#
#sum(test)/5000  # acceptance rate: 0.2522


# -----------------------------------------------------------------------------
# ---- 5. Metric (2) ----------------------------------------------------------
# -----------------------------------------------------------------------------

# blankspace reserved
#
#


# -----------------------------------------------------------------------------
# ---- 5. Metric (3) ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Correlation matrix for features, 10-by-10
corr_mat = np.corrcoef(np.transpose(data.features))
d = [1 - np.abs(_) for _ in corr_mat[1]]
d[1] = 0  # Correction avoids computational error

# C (cost matrix) -------------------------------------------------------------
C = [[0 for _ in range(K)] for _ in range(K)]
for i in range(K):
    for j in range(K):
        a = space_Z[i]
        b = space_Z[j]
        if a[10] != b[10]:
            C[i][j] = 1000    # 1000 will be set to infinity in optimization
        else:
            v = np.array([a[_] != b[_] for _ in range(10)]).astype(int)
            C[i][j] = np.inner(d, v)
            
# epsilon (transportation cost control) ---------------------------------------
eps = 1



# Test statistic --------------------------------------------------------------
Sequence = range(K)
Rows = Sequence
Cols = Sequence

prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

for i in range(K):
    prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_n[i]

for i in range(K):
    for j in range(K):
        if C[i][j] == 1000:
            prob += Pi[i][j] == 0
    
prob.solve()
pulp.LpStatus[prob.status]

for variable in prob.variables():
    print("{} = {}".format(variable.name, variable.varValue))

print(pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)]))
test_statistic = pulp.value(prob.objective) - sum([l[_]*f_n[_] for _ in range(K)])  # 0.612636
            
# sample size n ---------------------------------------------------------------
n = len(data.labels)  # 5278

# Bootstrap sample size m -----------------------------------------------------
m = np.floor(2 * np.sqrt(K) * np.sqrt(n))  #  1743

# Do B times ------------------------------------------------------------------
psi_list_boot = []
B = 1000
np.random.seed(2019)
for _ in range(B):
    f_boot = (np.random.multinomial(m, f_n, size = 1)/m).tolist()[0]

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)

    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')

    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])

    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= eps

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == f_boot[i]
        
    for i in range(K):
        for j in range(K):
            if C[i][j] == 1000:
                prob += Pi[i][j] == 0
            
    prob.solve()
    psi_list_boot.append(pulp.value(prob.objective) - sum([l[_]*f_boot[_] for _ in range(K)]))
    
    if _ % 20 == 0:
        print(_)
        
# Plot ------------------------------------------------------------------------
c_lower = np.percentile([m**0.5 * (x - test_statistic) for x in psi_list_boot], 2.5)
c_upper = np.percentile([m**0.5 * (x - test_statistic) for x in psi_list_boot], 97.5)

CI_lower = test_statistic - c_upper / np.sqrt(n) # 0.609744 (CI lower bound)
CI_upper = test_statistic - c_lower / np.sqrt(n) # 0.615689 (CI upper bound)
                                                 # 0.612636 (Test statistic)

sns.distplot([m**0.5 * (x - test_statistic) for x in psi_list_boot], hist=True, kde=True, 
             bins= 30, color = 'skyblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, "color": "k"})

plt.axvline(x=c_lower, color='orange', linestyle='--')
plt.axvline(x=c_upper, color='orange', linestyle='--')
#plt.savefig("boot_dist_3_m-out-of-n.pdf")
plt.show()

