import numpy as np
import itertools
import multiprocessing as mp
from functools import partial
import sys
from scipy.stats import norm


def entropy(y, p):
    """
    Calculates entropy

    parameters:
        y (float): label between 0 and 1
        p (float): probability taking value in [0, 1]

    return: 
        float; entropy loss
    """
    a = p if y else 1-p
    return -np.log(a+1e-16)



def bisection(der_loss, b_start, b_end, tol = 1e-7):
    """
    Bisection method for finding root for loss derivative 

    parameters:

        der_loss (function): given bias returns derivative of loss for logistic regression
        b_start (float): starting point
        b_end (float): end point
        tol (float): tolerence level

    return: 
        root of derivative

    """


    fs = der_loss(b_start)
    fe = der_loss(b_end)
    ITER = 0
    while np.sign(fs * fe) > 0 and ITER < 200:
        b_end += 1
        fe = der_loss(b_end)
    if ITER == 200 and np.sign(fs * fe) > 0:
        raise TypeError('Both of them have same sign')
    else:
        count = 0
        while np.absolute(b_start - b_end) > tol:
            b_mid = (b_start + b_end)/2
            fm = der_loss(b_mid)
            if np.sign(fs * fm) > 0 :
                b_start = b_mid
                fs = der_loss(b_start)
            else:
                b_end = b_mid
                fe = der_loss(b_end)
            count += 1
        print(f'Number of iterations {count}\nWith derivative {fs}')
        return b_start




def claculate_bias(theta, x, y):
    """
    Given weight vector theta, finds best possible bias value

    parameters:

        theta (list of floats): a length 2 list of weights

    return:
        best possible bias for logistic regression
    """
    theta = np.array(theta)
    a = x @ (theta.reshape((-1, 1)))
    def der_loss(b):
        logits =  a.reshape((-1, )) + b
        return - np.sum(y) + np.sum(np.exp(logits)/(1+np.exp(logits)))
    bias = bisection(der_loss, -30, 30)
    return bias




def soft_linear_classifier(theta, bias):
    """
    Given weights and bias, returns a soft classifier for logistic regression

    parameters:

        theta (list of floats): a length 2 list of weights
        bias (float): intercept value for logistic regression

    return: 
        a function that returns probability of success for feature vector of shape (2, )
    """
    
    
    theta = np.array(theta)
    

    #theta = theta/(np.linalg.norm(theta) + 1e-16)
    def classifier(x):
        logits = np.sum(x * theta + bias)
        if logits < 0:
            logits = np.log(np.exp(logits) + 1e-16)
        prob = 1 / (1 + np.exp(-logits))
        return prob
    return classifier

def hard_linear_classifier(theta, bias):

    soft_classifier = soft_linear_classifier(theta, bias)
    def hard_classifier(x):
        return (soft_classifier(x) > 0.5).astype('int32')
    
    return hard_classifier




def get_gradient(x, x_start, y,  theta, bias, fair_direction, regularizer):
    """
    Calculates gradient for entropy_loss - lambda * fair_distance(x, x_start)

    parameters:
        x : numpy array of shape (2, ); current position for gradient flow
        x_start:  numpy array of shape (2, ); starting position for gradient flow
        y (float): label between 0 and 1
        theta (list of floats): list of length 2 containing weights
        classifier: a function returnining probability of success for a numpy array of shape (2,)
        fair_direction (list of floats): list of length 2 for fair direction in 2-dimensional Euclidean space; must be of l2 norm 1
        regularizer (float): regularizer for fair metric 

    return:
        float; gradient
    """
    prob = 1/(1+np.exp(-np.sum(x * theta)-bias))
    # scalar =  2 * regularizer * np.sum(fair_direction * (x - x_start))
    # grad =  (prob - y) * theta + scalar * fair_direction
    # return grad / np.linalg.norm(grad)

    # prob = 1/(1+np.exp(-np.dot(x, theta)-bias))
    # return (prob - y) * theta + 2 * regularizer * np.array([(x[0] - x_start[0]), 0], dtype = 'float64')

    return (prob - y) * np.array([0, theta[1]], dtype='float64')



def sample_perturbation(data, theta, bias, fair_direction, regularizer = 5, learning_rate = 2e-2, num_steps = 200):
    """
    Returns loss ratio, original and final 0-1 loss for perturbed sample

    parameters:
        data: tuple containing x and y
            x: numpy array of shape (2, )
            y: label between 0 and 1
        theta (list of floats): list of length 2 containing weights
        bias (float): bias value for logistic classifier 
        fair_direction (list of floats): list of length 2 for fair direction in 2-dimensional Euclidean space; must be of l2 norm 1
        regularizer (float): regularizer for fair metric 
        learining_rate (float): step size for gradient flow is learning_rate/(step_number**(2/3))
        num_steps (int): number of steps for gradient flow attack

    return:
        float; ratio of two losses
    """


    #global orth_fair
    x, y = data
    x_start = x
    x_fair = x
    # soft_classifier = soft_linear_classifier(theta, bias)
    # hard_classifier = hard_linear_classifier(theta, bias)
    fair_direction = np.array(fair_direction)
    theta = np.array(theta)
    
    for i in range(num_steps):
        gradient = get_gradient(x_fair, x_start, y, theta, bias,   fair_direction, regularizer)
        x_fair = x_fair + learning_rate/((i+1) ** (2/3)) * gradient

    p_start = 1/(1+np.exp(-np.dot(theta, x_start)-bias))
    p_fair = 1/(1+np.exp(-np.dot(theta, x_fair)-bias))

    ratio = entropy(y, p_fair) / entropy(y, p_start)
    err_start = (y - (p_start>0.5).astype('float32')) ** 2
    err_fair = (y - (p_fair>0.5).astype('float32')) ** 2
    return ratio, err_start, err_fair








def lower_bound(x, y, theta, bias, fair_direction, regularizer = 1, learning_rate = 5e-3, num_steps = 200, cpus = 3, alpha = 0.05):
    """
    Calculates lower bound for expected loss ratio

    parameters:
        theta (list of floats): list of length 2 containing weights
        fair_direction (list of floats): list of length 2 for fair direction in 2-dimensional Euclidean space; must be of l2 norm 1
        regularizer (float): regularizer for fair metric 
        learining_rate (float): step size for gradient flow is learning_rate/(step_number**(2/3))
        num_steps (int): number of steps for gradient flow attack
        cpus (int): number of cpus for parallel processing; if cpus > 1, uses multiprocessing.Pool for parallel processing
    
    return: 
        float; lower bound for expected loss ratio
    """
    #x, y = np.load('data/x.npy'), np.load('data/y.npy')
    #bias = claculate_bias(theta)
    if cpus > 1:
        with mp.Pool(cpus) as pool:
            summaries = pool.map(partial(sample_perturbation, theta = theta, bias = bias, fair_direction = fair_direction,\
                regularizer = regularizer, learning_rate = learning_rate, num_steps = num_steps), zip(x, y))
    else:
        summaries = map(partial(sample_perturbation, theta = theta, bias = bias, fair_direction = fair_direction,\
                regularizer = regularizer, learning_rate = learning_rate, num_steps = num_steps), zip(x, y))
        summaries = list(summaries)
    summaries = np.array(summaries, dtype='float64')
    #print(summaries)
    ratios = summaries[:, 0].astype('float64')
    ratios = ratios[~np.isnan(ratios)]
    n = ratios.shape[0]
    mean = np.mean(ratios)
    std = np.std(ratios)
    T_n = mean -  norm.ppf(1-alpha)* std / np.sqrt(n)

    s = summaries[:, 1]
    e = summaries[:, 2]
    mean_s = np.mean(s)
    mean_e = np.mean(e)
    covar = np.cov(summaries[:, 1:], rowvar=False)
    # print(covar)
    T_n_tilde = mean_e/mean_s - (norm.ppf(1-alpha) / (np.sqrt(n) * mean_s ** 2))\
         * np.sqrt(mean_s ** 2 * covar[1, 1] + mean_e ** 2 * covar[0, 0] - 2 * mean_s * mean_e * covar[0, 1])
    

    print(f'Done for mean ratio of {theta} with  T_n T_n_tilde {T_n} {T_n_tilde}')
    return T_n, T_n_tilde




# if __name__ == "__main__":
    

#     theta1 = np.arange(-4, 4.1, step = 0.4)
#     theta2 = np.arange(-4, 4.1, step = 0.4)

#     thetas = itertools.product(theta1, theta2)
#     theta = [list(i) for i in thetas]

#     ang = int(float(sys.argv[1]))
#     angle = np.radians(ang*10)
#     c, s = np.cos(angle), np.sin(angle)
#     R = np.array(((c, -s), (s, c)))
#     fair_direction = np.array([[0], [1]])
#     fair_direction = R @ fair_direction
#     while np.linalg.norm(fair_direction) != 1:
#         fair_direction = fair_direction/np.linalg.norm(fair_direction)
#     fair_direction = fair_direction.reshape((-1,))
#     orth_fair = np.array([fair_direction[1], -fair_direction[0]])

#     mean_ratio_theta = []
#     if len(sys.argv) > 2:
#         reg, lr, num_steps = float(sys.argv[2]), float(sys.argv[3]), int(float(sys.argv[4]))
#         filename = f'data/test_stat_ang_{ang}_reg_{reg}_lr_{lr}_step_{num_steps}.npy'
#     else: 
#         reg, lr, num_steps = 100, 2e-2, 400
#         filename = f'data/test_stat_{ang}.npy'

#     for t1 in theta1:
#         mean_ratio_theta_row = []
    
#         for t2 in theta2:
#             r = lower_bound([t1, t2], fair_direction, regularizer= reg, learning_rate=lr, num_steps=num_steps)
#             mean_ratio_theta_row.append(r)
        
#         mean_ratio_theta.append(mean_ratio_theta_row)
    




#     np.save(filename, np.array(mean_ratio_theta))

