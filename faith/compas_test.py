# trying this https://github.com/IBM/AIF360/blob/master/examples/demo_reweighing_preproc.ipynb

import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from common_utils import compute_metrics

def gap(preds, y, protected_y, class_names=None, protected_names=None):
    y_names = np.unique(y)
    if protected_names is None:
        protected_names = np.unique(protected_y)
    protected_values = np.unique(protected_y)
    p_0_name = protected_names[0]
    p_1_name = protected_names[1]
    all_gaps = []
    all_tpr = []
    for c in y_names:
        if class_names is not None:
            c_name = class_names[int(c)]
        idx_c = np.where(y==c)[0]
        if len(idx_c) < 10:
            print('Nothing in test for %s' % c_name)
            continue
        idx_0 = np.where(protected_y[idx_c]==protected_values[0])[0]
        idx_1 = np.where(protected_y[idx_c]==protected_values[1])[0]
        if len(idx_0) < 10:
            print('Nothing in test for %s %s' % (p_0_name, c_name))
            continue
        if len(idx_1) < 10:
            print('Nothing in test for %s %s' % (p_1_name, c_name))
            continue
        tpr_c = preds[idx_c]==c
        all_tpr.append(tpr_c.mean())
        print('For class %s number of protected %s is %d; %s is %d' % (c_name, p_0_name, len(idx_0), p_1_name, len(idx_1)))
        tpr_0 = (preds[idx_c]==c)[idx_0]
        tpr_1 = (preds[idx_c]==c)[idx_1]
        gap_c = tpr_0.mean() - tpr_1.mean()
        all_gaps.append(gap_c)
        print('For class %s TPR for protected %s is %.3f; %s is %.3f' % (c_name, p_0_name, tpr_0.mean(), p_1_name, tpr_1.mean()))
        print('Class %s gap is %.3f\n' % (c_name, gap_c))
    total_gap = np.sqrt((np.array(all_gaps)**2).mean())
    print('Gap RMS %.3f; balanced TPR %.3f; max gap %.3f\n' % (total_gap, np.mean(all_tpr), np.abs(all_gaps).max()))
    return

## import dataset
dataset_used = "compas" # "adult", "german", "compas"
protected_attribute_used = 2 # 1, 2


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

all_metrics =  ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]

#random seed for calibrated equal odds prediction
np.random.seed(1)

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# Logistic regression classifier and predictions
X_train = dataset_orig_train.features
X_test = dataset_orig_test.features
y_train = dataset_orig_train.labels.ravel()
y_test = dataset_orig_test.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train)
#y_train_pred = lmod.predict(X_train)
y_test_pred = lmod.predict(X_test)

# Need to figure out metrics from the toolkit
#dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
#dataset_orig_test_pred.labels = y_test_pred.reshape(-1,1)
#
#metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
#                                  unprivileged_groups, privileged_groups,
#                                  disp = True)

protected_y = dataset_orig_test.protected_attributes.flatten()
gap(y_test_pred, y_test, protected_y, class_names=['no_recid','two_year_recid'], protected_names=['black', 'white'])


## Reweighting approach
RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
RW.fit(dataset_orig_train)
dataset_transf_train = RW.transform(dataset_orig_train)

# Train classifier on transformed data
lmod_t = LogisticRegression()
lmod_t.fit(X_train, y_train,
        sample_weight=dataset_transf_train.instance_weights)
y_test_pred_t = lmod_t.predict(X_test)

gap(y_test_pred_t, y_test, protected_y, class_names=['no_recid','two_year_recid'], protected_names=['black', 'white'])
