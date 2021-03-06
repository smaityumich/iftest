import numpy as np
import pulp
import scipy
import json
import multiprocessing as mp
from functools import partial

def auditor_problem(p, C, l, delta, infty_equiv = 1000):
    K = p.shape[0]
    Sequence = range(K)
    Rows = Sequence
    Cols = Sequence

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)
    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')
    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])
    prob += pulp.lpSum([C[i, j] * Pi[i][j] for i in range(K) for j in range(K)]) <= delta

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == p[i]

    for i in range(K):
        for j in range(K):
            if C[i, j] == infty_equiv:
                prob += Pi[i][j] == 0
    solver = pulp.getSolver('PULP_CBC_CMD', msg = 0)
    prob.solve(solver)


    return pulp.value(prob.objective) - sum([l[_]*p[_] for _ in range(K)])


def data_process(x, y, c_distance, classifier, infty_equiv = 1000):
    n = y.shape[0]
    z = np.array([json.dumps({'x': list(x[i, :].astype('float64')), 'y': y[i].astype('float64')}) for i in range(n)])
    Z, count = np.unique(z, return_counts=True)
    p_n = count/n

    K = p_n.shape[0]
    # print(f'\n---------------\nNumber of discrete sample elements: {K}\n-----------And they are:\n')
    # for z in Z:
    #     print(z + '\n')

    C = np.zeros(shape = (K, K))
    for i in range(K):
        for j in range(K):
            if i < j:
                zi = x[i,:], y[i]
                zj = x[j, :], y[j]
                C[i, j] = c_distance(zi, zj, infty_equiv = infty_equiv)
                C[j, i] = C[i, j]
            else:
                continue

    def error(z):
        d = json.loads(z)
        x, y = np.array(d['x']).astype('float32'), d['y']
        y_hat = classifier(x)
        return int(y != y_hat)


    l = [error(_) for _ in Z]
    l = np.array(l)


    return Z, p_n, C, K, n, l


def faith_test(K, p_n, C, n, l, infty_equiv = 1000, B = 1000, alpha = 0.05, \
 random_state = 0, m = None, delta = 0.1, cpus = 1):
    
    sample_audit = auditor_problem(p = p_n, l = l, C = C, delta= delta, infty_equiv= infty_equiv)
    print(f'\n----------------\nValue of sample: {sample_audit}\n------------------------\n')
    
    if isinstance(m, type(None)):
        m = np.floor(2 * np.sqrt(K) * np.sqrt(n))
        

    np.random.seed(random_state)


    def bootstrap(i, l, C, delta, m, auditor_problem, infty_equiv, sample_audit):
        p = np.random.multinomial(m, p_n)/m
        r =  np.sqrt(m) * (auditor_problem(p = p, l = l, C = C, delta= delta, infty_equiv=infty_equiv)-sample_audit)
        #print(f'\n----------------\nValue of {i}-th bootstrap sample: {r}\n------------------------\n')
        return r

    bootstrap_partial = partial(bootstrap, l = l, C = C, delta = delta, m = m,\
         auditor_problem = auditor_problem, infty_equiv = infty_equiv, sample_audit = sample_audit)
    # if cpus == 1:
    audit_list = list(map(bootstrap_partial, range(B)))
    # elif cpus > 1:
    #     with mp.Pool(cpus) as pool:
    #         bootstrap_partial = partial(bootstrap, l = l, C = C, delta = delta, m = m,\
    #             auditor_problem = auditor_problem, infty_equiv = infty_equiv, sample_audit = sample_audit)
    #         audit_list = pool.map(bootstrap_partial, range(B))
    
    audit_list = np.array(audit_list)
    c = np.quantile((audit_list - sample_audit), 1-alpha)
    return sample_audit - c/np.sqrt(n)
    

    



    

