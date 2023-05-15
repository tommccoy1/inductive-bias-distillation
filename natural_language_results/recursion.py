
import statistics
import os


import numpy as np
import scipy.stats

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


yespre_values = {}
nopre_values = {}

yes_mean = 0
no_mean = 0

n_runs = 40
directory = "targeted_evaluations/"

for pre in ["nopre", "yespre"]:

    for index in range(n_runs):
        if pre == "nopre":
            if args.same:
                model_name = "adapt_params1_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_recursion.log"
            else:
                model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_recursion.log"
        else:
            model_name = "bestparams_adapt_hidden1024_pretraining_full_yespre" + str(index) + "_0_eval_recursion.log" 

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

for implausible in [True, False]:
    for general_category in ["recursion_intensifier_adv", "recursion_intensifier_adj", "recursion_poss_transitive", "recursion_poss_ditransitive", "recursion_pp_is", "recursion_pp_verb"]:
    
        yespre_means = []
        nopre_means = []

        yes_confidences = []
        no_confidences = []
    
        for count in range(11):
            category = general_category + "_" + str(count)
            if implausible:
                category = category + "_implausible"



            if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
                print("WRONG COUNT")
                15/0

            total += 1
            yes_mean = statistics.mean(yespre_values[category])
            no_mean = statistics.mean(nopre_values[category])
            pvalue = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category])).pvalue

            #print("Yes values", sorted(yespre_values[category]))
            #print("No values", sorted(nopre_values[category]))

            yespre_means.append(yes_mean)
            nopre_means.append(no_mean)

            if yes_mean > no_mean:
                yes_better += 1
            if no_mean > yes_mean:
                no_better += 1
        
            if pvalue < 0.05:
                if yes_mean > no_mean:
                    yes_sig_better += 1
                if no_mean > yes_mean:
                    no_sig_better += 1

        if args.r:
            if implausible:
                print("yes_" + general_category + "_implausible <- c(" + ", ".join([str(round(x, 4)) for x in yespre_means]) + ")")
                print("no_" + general_category + "_implausible <- c(" + ", ".join([str(round(x, 4)) for x in nopre_means]) + ")")
                print("")
            else:
                print("yes_" + general_category + "_plausible <- c(" + ", ".join([str(round(x, 4)) for x in yespre_means]) + ")")
                print("no_" + general_category + "_plausible <- c(" + ", ".join([str(round(x, 4)) for x in nopre_means]) + ")")
                print("")
        else:
            if implausible:
                print(general_category + "_implausible " + "yespre:", [round(x, 3) for x in yespre_means])
                print(general_category + "_implausible " + "nopre:", [round(x, 3) for x in nopre_means])
                print("")
            else:
                print(general_category + "_plausible " + "yespre:", [round(x, 3) for x in yespre_means])
                print(general_category + "_plausible " + "nopre:", [round(x, 3) for x in nopre_means])
                print("")



print("Yes better:", yes_better, "out of", total)
print("No better:", no_better, "out of", total)
print("Yes significantly better:", yes_sig_better, "out of", total)
print("No significantly better:", no_sig_better, "out of", total)

