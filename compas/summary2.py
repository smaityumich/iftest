import tensorflow as tf
import json, sys
import numpy as np

from compas_data import get_compas_train_test
from sklearn import linear_model
#import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
import metrics, itertools
from scipy.stats import norm


def SimpleDense(variable):
    w, b = variable
    w = tf.cast(w, dtype = tf.float32)
    b = tf.cast(b, dtype = tf.float32)
    return lambda x: tf.matmul(x, w) + b

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


    

    return graph


def summary_all(seed_data, seed_model, lr, graph, exp = 'sensr'):



    x_train, x_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train,\
          y_race_test, feature_names = get_compas_train_test(random_state = seed_data)



    x_test = tf.cast(x_test, dtype = tf.float32)
    prob = graph(x_test)
    y_pred = tf.argmax(prob, axis = 1)
    y_pred = y_pred.numpy()
    gender = y_sex_test
    race = y_race_test
     #y_test = y_test.numpy()[:, 1]
     
    print('\n\nMeasures for gender\n')
    accuracy, bal_acc, \
            gap_rms_gen, mean_gap_gen, max_gap_gen, \
            average_odds_difference_gen, equal_opportunity_difference_gen,\
                 statistical_parity_difference_gen = metrics.group_metrics(y_test, y_pred, gender, label_good=1)

    print('\n\n\nMeasures for race\n')
    accuracy, bal_acc, \
            gap_rms_race, mean_gap_race, max_gap_race, \
            average_odds_difference_race, equal_opportunity_difference_race,\
                 statistical_parity_difference_race = metrics.group_metrics(y_test, y_pred, race, label_good=1)

     
    filename = exp + f'/outcome/perturbed_ratio_0_to_1000_seed_{seed_data}_{seed_model}_lr_{lr}.npy'
    all_val = np.load(filename)
    a = all_val[:, 0]
    b = all_val[:, 1:]
    a = a[np.isfinite(a)]
    lb = np.mean(a) - 1.645*np.std(a)/np.sqrt(a.shape[0])
    t = (np.mean(a)-1.25)/np.std(a)
    t *= np.sqrt(a.shape[0])
    pval = 1- norm.cdf(t)

    mean_b = np.mean(b, axis = 0)
    cov_b = np.cov(b, rowvar=False)
    var_ratio = (cov_b[0, 0]* mean_b[1] ** 2 + cov_b[1, 1] * mean_b[0] ** 2 \
          - 2 * cov_b[0, 1] * mean_b[0] * mean_b[1])/(mean_b[0] ** 4)

    t_ratio = mean_b[1]/mean_b[0]
    lb_t2 = t_ratio - 1.645 * np.sqrt(var_ratio)/np.sqrt(all_val.shape[0])


    save_dict = {'algo': exp, 'seed': (seed_data, seed_model), 'lr': lr, 'accuracy': accuracy}
    save_dict['lb'] = lb
    save_dict['pval'] = pval
    save_dict['bal_acc'], \
            save_dict['gap_rms_gen'], save_dict['mean_gap_gen'], save_dict['max_gap_gen'], \
            save_dict['average_odds_difference_gen'], save_dict['equal_opportunity_difference_gen'],\
                 save_dict['statistical_parity_difference_gen'] = bal_acc, \
            gap_rms_gen, mean_gap_gen, max_gap_gen, \
            average_odds_difference_gen, equal_opportunity_difference_gen,\
                 statistical_parity_difference_gen

    save_dict['bal_acc'], \
            save_dict['gap_rms_race'], save_dict['mean_gap_race'], save_dict['max_gap_race'], \
            save_dict['average_odds_difference_race'], save_dict['equal_opportunity_difference_race'],\
                 save_dict['statistical_parity_difference_race'] = bal_acc, \
            gap_rms_race, mean_gap_race, max_gap_race, \
            average_odds_difference_race, equal_opportunity_difference_race,\
                 statistical_parity_difference_race

    save_dict['lb-t2'] = lb_t2

    return save_dict

if __name__ == '__main__':
    
    expts = ['baseline', 'project', 'sensr', 'reduction'] 
    iteration = range(10)
    lrs =  [4e-3, 2e-3, 6e-3]

    a = list(itertools.product(expts, iteration, lrs))
    i = int(float(sys.argv[1]))
    exp, iters, lr = a[i]
    seeds = np.load('./seeds.npy')
    seed_data = seeds[i, 0]
    seed_model = seeds[i, 1]

    graph = load_model(seed_data, seed_model, method = exp)

    d = summary_all(seed_data, seed_model, lr, graph, exp = exp)
    with open(f'summaries/f_{i}.txt', 'w') as f:
        json.dump([d], f)
    