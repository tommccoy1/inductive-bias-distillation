
import statistics
import os

directory = "targeted_evaluations/"

import numpy as np
import scipy.stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Minimal pair dataset to compute results for. Options: zorro, blimp, scamp_plausible, scamp_implausible", type=str, default="zorro")
parser.add_argument("--same", help="For the standard network, use the version trained with the same hyperparameters as the prior-trained network", action='store_true')
args = parser.parse_args()



files = os.listdir(directory)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return h


yespre_values = {}
nopre_values = {}


n_runs = 40

for pre in ["nopre", "yespre"]:

    wrong_length = False

    for index in range(n_runs):
        if pre == "nopre":
            if args.same:
                model_name = "adapt_params1_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_" + args.dataset + ".log"
            else:
                model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_" + args.dataset + ".log"
        else:
            model_name = "bestparams_adapt_hidden1024_pretraining_full_yespre" + str(index) + "_0_eval_" + args.dataset + ".log"

        fi = open(directory + model_name, "r")

        for line in fi:
            if "MINIMAL PAIR RESULTS: OVERALL:" in line:
                acc = float(line.strip().split()[-1])
                if pre == "nopre":
                    if "overall" not in nopre_values:
                        nopre_values["overall"] = []
                    nopre_values["overall"].append(acc)
                else:
                    if "overall" not in yespre_values:
                        yespre_values["overall"] = []
                    yespre_values["overall"].append(acc)
            elif "MINIMAL PAIR RESULTS:" in line:
                name = line.strip().split()[-4][8:]
                acc = float(line.strip().split()[-1])
                        
                if pre == "nopre":
                    if name not in nopre_values:
                        nopre_values[name] = []
                    nopre_values[name].append(acc)
                else:
                    if name not in yespre_values:
                        yespre_values[name] = []
                    yespre_values[name].append(acc)


yes_better = 0
no_better = 0
total = 0
yes_sig_better = 0
no_sig_better = 0
for category in nopre_values:
   
    if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
        print("WRONG COUNT")
        15/0

    yes_mean = statistics.mean(yespre_values[category])
    no_mean = statistics.mean(nopre_values[category])
    pvalue = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])).pvalue

    print(category)
    print("Nopre", sorted(nopre_values[category]))
    print("Yespre", sorted(yespre_values[category]))
    print("Nopre", no_mean) 
    print("Yespre", yes_mean)
    print(scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])))
    print("Pvalue", pvalue) 
    print("")


    if category != "overall":
        total += 1

        if yes_mean > no_mean:
            yes_better += 1
        if no_mean > yes_mean:
            no_better += 1
        
        if pvalue < 0.05:
            if yes_mean > no_mean:
                yes_sig_better += 1
            if no_mean > yes_mean:
                no_sig_better += 1


print("Yes better:", yes_better, total)
print("No better:", no_better, total)
print("Yes sig better:", yes_sig_better, total)
print("No sig better:", no_sig_better, total)


