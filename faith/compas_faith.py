import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import compas_data as compas
import tensorflow as tf
import json, sys, itertools

def SimpleDense(variable):
    w, b = variable
    w = tf.cast(w, dtype = tf.float32)
    b = tf.cast(b, dtype = tf.float32)
    return lambda x: tf.matmul(x, w) + b

# 0.2 AIF360
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

def sensitive_dir(x, gender, race):
    d = x.shape[1]
    print(d)
    sensetive_directions = []
    protected_regression = LogisticRegression(fit_intercept = True)
    protected_regression.fit(x[:, 2:], gender)
    a = protected_regression.coef_.reshape((-1,))
    a = np.concatenate(([0, 0], a), axis=0)
    sensetive_directions.append(a)
    protected_regression.fit(x[:,2:], race)
    a = protected_regression.coef_.reshape((-1,))
    a = np.concatenate(([0, 0], a), axis=0)
    sensetive_directions.append(a)
    a, b = np.zeros((d,)), np.zeros((d,))
    a[0], b[1] = 1, 1
    sensetive_directions.append(a)
    sensetive_directions.append(b)
    sensetive_directions = np.array(sensetive_directions)

    # Extrancting orthornormal basis for sensitive directions
    sensetive_basis = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_basis):
        #while np.linalg.norm(s) != 1:
        s = s/ np.linalg.norm(s)
        sensetive_basis[i] = s

    return sensetive_directions, sensetive_basis


def fair_distance(x, y,  sensetive_directions):
    _, d = sensetive_directions.shape
    proj = np.identity(d) - sensetive_directions.T @ sensetive_directions
    Cp =  x @ proj @ x.T
    inf = (2*y-1).reshape((-1, 1)) @ (2*y-1).reshape((1, -1))
    return (1 - inf) * Cp + inf * 1000



def data_modify(random_state = 0):

    _, x_test, _, y_test, _, y_sex_test,\
        _, y_race_test, _ = compas.get_compas_train_test(random_state = random_state)
    y_sex_test, y_race_test = np.copy(y_sex_test), np.copy(y_race_test)
    print(x_test.shape)
    _, sensetive_directions = sensitive_dir(x_test, y_sex_test, y_race_test)


    # Feature names:
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

    # Label name: 'two_year_recid'          # f = 0, 1


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
    K = len(space_Z)
    p_n = list(0 for _ in range(K))
    Z_dict = dict(zip(space_Z, list(range(K))))
    
    n = len(y_test)
    for _ in range(n):
        z = tuple(x_test[_]) + tuple(y_test[_])
        #z = tuple(int(x) for x in z)
        p_n[Z_dict[z]] += 1

    space_Z_np = np.array(space_Z, dtype = 'float64')

    C = fair_distance(space_Z_np[:, :-1], space_Z_np[:, -1], sensetive_directions)
    
    return p_n, C, K, n, space_Z_np

def loss_vector(hard_classifier, space_Z_np):
    l = (space_Z_np[:, -1] - hard_classifier(space_Z_np[:, :-1])) ** 2
    return l


def faith_test(p_n, C, l, K, n, B = 1000, delta = 0):
    Sequence = range(K)
    Rows = Sequence
    Cols = Sequence

    prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)
    Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')
    prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])
    prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= delta

    for i in range(K):
        prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == p_n[i]

    for i in range(K):
        for j in range(K):
            if C[i][j] == 1000:
                prob += Pi[i][j] == 0
    
    solver = pulp.getSolver('PULP_CBC_CMD', msg = 0)
    prob.solve(solver)

    test_statistic = pulp.value(prob.objective) - sum([l[_]*p_n[_] for _ in range(K)])


    # Bootstrap sample size m -----------------------------------------------------
    m = np.floor(2 * np.sqrt(K) * np.sqrt(n))  #  1743

    # Do B times ------------------------------------------------------------------
    psi_list_boot = []
    np.random.seed(2019)
    for _ in range(B):
        p_boot = (np.random.multinomial(m, p_n, size = 1)/m).tolist()[0]
        prob = pulp.LpProblem("Fairness Testing", pulp.LpMaximize)
        Pi = pulp.LpVariable.dicts("pi", (Rows, Cols), lowBound=0, cat='Continuous')
        prob += pulp.lpSum([l[i] * Pi[j][i] for i in range(K) for j in range(K)])
        prob += pulp.lpSum([C[i][j] * Pi[i][j] for i in range(K) for j in range(K)]) <= delta

        for i in range(K):
            prob += pulp.lpSum([Pi[i][j] for j in range(K)]) == p_boot[i]
        
        for i in range(K):
            for j in range(K):
                if C[i][j] == 1000:
                    prob += Pi[i][j] == 0
            
        solver = pulp.getSolver('PULP_CBC_CMD', msg = 0)
        prob.solve(solver)
        psi_list_boot.append(pulp.value(prob.objective) - sum([l[_]*p_boot[_] for _ in range(K)]))
    
        if _ % 20 == 0:
            print(_)

    for _ in range(B):  # Due to computational error, we do this correction
        if psi_list_boot[_] < 1e-8:
            psi_list_boot[_] = 0

    c_upper = np.quantile([m**0.5 * (x - test_statistic) for x in psi_list_boot], 0.95)
    return test_statistic - c_upper/np.sqrt(n) , test_statistic, c_upper



def load_model(seed_data, seed_model, method = 'sensr'):

    if method == 'baseline':
        graph = tf.keras.models.load_model(f'./baseline/graphs/graph_{seed_data}_{seed_model}') 
    elif method == 'project':
        graph = tf.keras.models.load_model(f'./project/graphs/graph_{seed_data}_{seed_model}') 
    elif method == 'reduction':
        with open(f'./reduction/models/data_{seed_data}.txt', 'r') as f:
            data = json.load(f)
    
        coef = data['coefs']
        intercept = data['intercepts']
        weight = data['ens_weights']
        coefs = [tf.cast(c, dtype = tf.float32) for c in coef]
        intercepts = [tf.cast(c, dtype = tf.float32) for c in intercept]
        weights = [tf.cast(c, dtype = tf.float32) for c in weight]

        def graph(x):
            global data
            n, _ = x.shape
            prob = tf.zeros([n, 1], dtype = tf.float32)
            for coef, intercept, weight in zip(coefs, intercepts, weights):
                coef = tf.reshape(coef, [-1, 1])
                model_logit = x @ coef + intercept
                model_prob = tf.exp(model_logit) / (1 + tf.exp(model_logit))
                prob += model_prob * weight

            return tf.concat([1-prob, prob], axis = 1)

    else:
        with open(f'./sensr/models/data_{seed_data}_{seed_model}.txt', 'r') as f:
            weight = json.load(f)

        weights = [np.array(w) for w in weight]

        def graph(x):
            layer1 = SimpleDense([weights[0], weights[1]])
            layer2 = SimpleDense([weights[2], weights[3]])
            out = tf.nn.relu(layer1(x))
            out = layer2(out)
            prob = tf.nn.softmax(out)
            return prob


    def hard_classifier(x):
        x = tf.cast(x, dtype = tf.float32)
        prob = graph(x)[:, 1]
        return (prob > 0.5).numpy().astype('float64')

    return hard_classifier


if __name__ == '__main__':
    i = int(float(sys.argv[1]))
    expts = ['sensr', 'reduction', 'baseline', 'project'] 
    iteration = range(10)
    delta = 0
    a = list(itertools.product(expts, iteration))
    exp, iters = a[i]
    seeds = np.load('./seeds.npy')
    data_seed = seeds[iters, 0]
    expt_seed = seeds[iters, 1]

    p_n, C, K, n, space_Z_np = data_modify(data_seed)
    hard_classifier = load_model(data_seed, expt_seed, method=exp)
    l = loss_vector(hard_classifier, space_Z_np)
    psi, test_statistic, c_upper = faith_test(p_n, C, l, K, n, delta = delta)
    d = {'psi': psi, 'test-stat': test_statistic, 'c-upper': c_upper, 'exp': exp, 'iter': iters}
    with open(f'./faith_results/res_{i}.txt', 'w') as f:
        json.dump(d, f)


