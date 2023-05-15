
import statistics
import os

directory = "targeted_evaluations/"

import numpy as np
import scipy.stats
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--r", help="Print outputs for ggplot in R", action='store_true')
parser.add_argument("--same", help="For the standard network, use the version trained with the same hyperparameters as the prior-trained network", action='store_true')
args = parser.parse_args()


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return h

n_runs = 40

yespre_values = {}
nopre_values = {}

keys = ["priming_short", "priming_long", "priming_short_implausible", "priming_long_implausible"]

for key in keys:
    yespre_values[key] = []
    nopre_values[key] = []

yes_mean = 0
no_mean = 0

for pre in ["nopre", "yespre"]:

    for index in range(n_runs):
        
        if pre == "nopre":
            if args.same:
                model_name = "adapt_params1_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_priming.log"
            else:
                model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_priming.log"
        else:
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
                        else:
                            nopre_values[key].append(ratio)


                    


yes_better = 0
no_better = 0
total = 0
yes_sig_better = 0
no_sig_better = 0
for category in nopre_values:

    if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
        print("WRONG COUNT")
        15/0

    if args.r:
        print("yes_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(yespre_values[category])]) + ")")
        print("no_" + category + " <- c(" + ", ".join([str(round(x, 4)) for x in sorted(nopre_values[category])]) + ")")
        print("")
    else:
        print(category)
        print("Nopre", sorted(nopre_values[category]))
        print("Yespre", sorted(yespre_values[category]))
    
        print("Nopre", statistics.mean(nopre_values[category]))
        print("Yespre", statistics.mean(yespre_values[category]))
        print("Pvalue", scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])).pvalue)
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


