
import statistics
import os

directory = "targeted_evaluations/"

import numpy as np
import scipy.stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Minimal pair dataset to compute results for. Options: zorro, blimp, scamp_plausible, scamp_implausible", type=str, default="zorro")
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
pre_values = {}


n_runs = 40

for pre in ["nopre", "yespre", "pre"]:

    wrong_length = False

    for index in range(n_runs):
        if pre == "nopre":
            model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_" + args.dataset + ".log"
        elif pre == "pre":
            model_name = "_".join(["besthyps_hidden1024_pseudometa", str(index), "0_eval", args.dataset]) + ".log"
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
                elif pre == "pre":
                    if "overall" not in pre_values:
                        pre_values["overall"] = []
                    pre_values["overall"].append(acc)
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
                elif pre == "pre":
                    if name not in pre_values:
                        pre_values[name] = []
                    pre_values[name].append(acc)
                else:
                    if name not in yespre_values:
                        yespre_values[name] = []
                    yespre_values[name].append(acc)


yes_better = 0
no_better = 0
total = 0
yes_sig_better = 0
no_sig_better = 0

yes_vs_pre_sig_better = 0
pre_vs_yes_sig_better = 0

no_vs_pre_sig_better = 0
pre_vs_no_sig_better = 0
for category in nopre_values:
   
    if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs or len(pre_values[category]) != n_runs:
        print("WRONG COUNT")
        15/0

    yes_mean = statistics.mean(yespre_values[category])
    no_mean = statistics.mean(nopre_values[category])
    pre_mean = statistics.mean(pre_values[category])
    pvalue_yes_no = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category]), equal_var=False).pvalue
    pvalue_yes_pre = scipy.stats.ttest_ind(a=np.array(pre_values[category]), b=np.array(yespre_values[category]), equal_var=False).pvalue
    pvalue_no_pre = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(pre_values[category]), equal_var=False).pvalue

    print(category)
    print("Standard", sorted(nopre_values[category]))
    print("Meta", sorted(yespre_values[category]))
    print("Pre", sorted(pre_values[category]))
    print("Standard", no_mean) 
    print("Meta", yes_mean)
    print("Pre", pre_mean)
    print("Pvalue yes no", pvalue_yes_no)
    print("Pvalue yes pre", pvalue_yes_pre)
    print("Pvalue no pre", pvalue_no_pre)
    print("")




