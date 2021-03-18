import numpy as np
import pandas as pd

results = np.load('tensor.npy')

summary_mean = results.mean(axis=0)
summary_std = results.std(axis=0)

methods = ['LR', 'Adv. debiasing', 'Reweighting', 'LFR']
metrics =  ['FaiTH statistic', "CI lb (two-sided)", "CI ub (two_sided)", "CI lb (one-sided)", "Accuracy", "Average Odds Difference", "Equal Opportunity Difference", "Statistical Parity Difference"]

table = pd.DataFrame(summary_mean, columns=metrics, index=methods)
table_std = pd.DataFrame(summary_std, columns=metrics, index=methods)

for c in metrics:
    table[c] = table[c].apply("{:.3f}".format)+ '$\pm$' + table_std[c].apply("{:.3f}".format)

print(table.to_latex(escape=False))