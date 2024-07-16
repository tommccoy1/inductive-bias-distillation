
import statistics
import os


import numpy as np
import scipy.stats
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--r", help="Print outputs for ggplot in R", action='store_true')
args = parser.parse_args()


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return h

n_runs = 20

yespre_values = {}
nopre_values = {}
yespre_nosync_values = {}
yespre_norec_values = {}

keys = ["priming_short", "priming_long", "priming_short_implausible", "priming_long_implausible"]

for key in keys:
    yespre_values[key] = []
    nopre_values[key] = []
    yespre_nosync_values[key] = []
    yespre_norec_values[key] = []

for pre in ["nopre", "yespre", "yespre_nosync", "yespre_norec"]:

    for index in range(n_runs):
        
        if pre == "nopre":
            directory = "../natural_language_results/targeted_evaluations/"
            model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_priming.log"
        elif pre == "yespre_nosync":
            directory = "no_synchrony/"
            model_name = "adapt_no_sync_hidden1024_" + str(index) + "_0_eval_priming.log"
        elif pre == "yespre_norec":
            directory = "no_recursion/"
            model_name = "adapt_no_recursion_hidden1024_" + str(index) + "_0_eval_priming.log"
        else:
            directory = "../natural_language_results/targeted_evaluations/"
            model_name = "bestparams_adapt_hidden1024_pretraining_full_yespre" + str(index) + "_0_eval_priming.log"

        fi = open(directory + model_name, "r")

        for line in fi:
            if "AVG SINGLE PERPLEXITY" in line:
                single = float(line.strip().split()[-1])

            if "AVG DOUBLE PERPLEXITY" in line:
                double = float(line.strip().split()[-1])
                ratio = double/single

                for key in keys:
                    if key + " " in line:
                        if pre == "yespre":
                            yespre_values[key].append(ratio)
                        elif pre == "yespre_nosync":
                            yespre_nosync_values[key].append(ratio)
                        elif pre == "yespre_norec":
                            yespre_norec_values[key].append(ratio)
                        else:
                            nopre_values[key].append(ratio)


                    


yes_better = 0
no_better = 0
yes_nosync_better = 0
yes_norec_better = 0
total = 0
yes_sig_better = 0
no_sig_better = 0
for category in nopre_values:

    if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
        print("WRONG COUNT")
        15/0

    if args.r:
        print("yes_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(yespre_values[category])]) + ")")
        print("yes_nosync_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(yespre_nosync_values[category])]) + ")")
        print("yes_norec_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(yespre_norec_values[category])]) + ")")
        print("no_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(nopre_values[category])]) + ")")
        print("")
    else:
        print(category)
        print("Nopre         ", sorted(nopre_values[category]))
        print("Yespre        ", sorted(yespre_values[category]))
        print("Yespre no sync", sorted(yespre_nosync_values[category]))
        print("Yespre no rec ", sorted(yespre_norec_values[category]))
    
        print("Nopre         ", statistics.mean(nopre_values[category]))
        print("Yespre        ", statistics.mean(yespre_values[category]))
        print("Yespre no sync", statistics.mean(yespre_nosync_values[category]))
        print("Yespre no rec ", statistics.mean(yespre_norec_values[category]))
        print("Pvalue (yes/no)", scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])).pvalue)
        print("Pvalue (yes/yes no sync)", scipy.stats.ttest_ind(a=np.array(yespre_nosync_values[category]), b=np.array(yespre_values[category])).pvalue)
        print("Pvalue (yes/yes no rec)", scipy.stats.ttest_ind(a=np.array(yespre_norec_values[category]), b=np.array(yespre_values[category])).pvalue)
        print("Pvalue (no/yes no sync)", scipy.stats.ttest_ind(a=np.array(yespre_nosync_values[category]), b=np.array(nopre_values[category])).pvalue)
        print("Pvalue (no/yes no rec)", scipy.stats.ttest_ind(a=np.array(yespre_norec_values[category]), b=np.array(nopre_values[category])).pvalue)
        print("Pvalue (yes no rec/yes no sync)", scipy.stats.ttest_ind(a=np.array(yespre_nosync_values[category]), b=np.array(yespre_norec_values[category])).pvalue)
        print("")

    if category != "overall":
        total += 1
        yes_mean = statistics.mean(yespre_values[category])
        no_mean = statistics.mean(nopre_values[category])
        pvalue = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])).pvalue

        if yes_mean < no_mean:
            yes_better += 1
        if no_mean < yes_mean:
            no_better += 1
        
        if pvalue < 0.05:
            if yes_mean < no_mean:
                yes_sig_better += 1
            if no_mean < yes_mean:
                no_sig_better += 1


print("Yes better:", yes_better, "out of", total)
print("No better:", no_better, "out of", total)
print("Yes significantly better:", yes_sig_better, "out of", total)
print("No significantly better:", no_sig_better, "out of", total)


