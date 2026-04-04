
import statistics
import os


import numpy as np
import scipy.stats

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

def print3(my_list):
    return ", ".join(["%.3f" % x for x in my_list])

yespre_values = {}
nopre_values = {}
pre_values = {}


n_runs = 40
directory = "targeted_evaluations/"

for pre in ["nopre", "yespre", "pre"]:

    for index in range(n_runs):
        if pre == "nopre":
            directory = "../natural_language_results/targeted_evaluations/"
            model_name = "bestparams_adapt_hidden1024_pretraining_full_nopre_" + str(index) + "_eval_recursion.log"
        elif pre == "yespre":
            directory = "../natural_language_results/targeted_evaluations/"
            model_name = "bestparams_adapt_hidden1024_pretraining_full_yespre" + str(index) + "_0_eval_recursion.log" 
        else:
            directory = "../natural_language_results/targeted_evaluations/"
            model_name = "_".join(["besthyps_hidden1024_pseudometa", str(index), "0_eval", "recursion"]) + ".log"

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


yespre_totals = [0,0,0,0,0,0,0,0,0,0,0]
pre_totals = [0,0,0,0,0,0,0,0,0,0,0]
nopre_totals = [0,0,0,0,0,0,0,0,0,0,0]

for implausible in [True, False]:
    for general_category in ["recursion_intensifier_adv", "recursion_intensifier_adj", "recursion_poss_transitive", "recursion_poss_ditransitive", "recursion_pp_is", "recursion_pp_verb"]:
    
        yespre_means = []
        nopre_means = []
        pre_means = []

        yes_confidences = []
        no_confidences = []
        pre_confidences = []
    
        for count in range(11):
            category = general_category + "_" + str(count)
            if implausible:
                category = category + "_implausible"



            if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs or len(pre_values[category]) != n_runs:
                print("WRONG COUNT")
                15/0

            yes_mean = statistics.mean(yespre_values[category])
            no_mean = statistics.mean(nopre_values[category])
            pre_mean = statistics.mean(pre_values[category])

            yespre_means.append(yes_mean)
            nopre_means.append(no_mean)
            pre_means.append(pre_mean)

        for index, elt in enumerate(yespre_means):
            yespre_totals[index] += elt

        for index, elt in enumerate(pre_means):
            pre_totals[index] += elt

        for index, elt in enumerate(nopre_means):
            nopre_totals[index] += elt


        if args.r:
            if implausible:
                print("yes_" + general_category + "_implausible <- c(" + ", ".join([str(round(x, 4)) for x in yespre_means]) + ")")
                print("pre_" + general_category + "_implausible <- c(" + ", ".join([str(round(x, 4)) for x in pre_means]) + ")")
                print("no_" + general_category + "_implausible <- c(" + ", ".join([str(round(x, 4)) for x in nopre_means]) + ")")
                print("")
            else:
                print("yes_" + general_category + "_plausible <- c(" + ", ".join([str(round(x, 4)) for x in yespre_means]) + ")")
                print("pre_" + general_category + "_plausible <- c(" + ", ".join([str(round(x, 4)) for x in pre_means]) + ")")
                print("no_" + general_category + "_plausible <- c(" + ", ".join([str(round(x, 4)) for x in nopre_means]) + ")")
                print("")
        else:
            if implausible:
                print(general_category + "_implausible " + "yespre:        " + print3([round(x, 3) for x in yespre_means]))
                print(general_category + "_implausible " + "yespre norec:  " + print3([round(x, 3) for x in pre_means]))
                print(general_category + "_implausible " + "nopre:         " + print3([round(x, 3) for x in nopre_means]))
                print("")
            else:
                print(general_category + "_plausible " + "yespre:        " + print3([round(x, 3)  for x in yespre_means]))
                print(general_category + "_plausible " + "yespre nosync: " + print3([round(x, 3) for x in pre_means]))
                print(general_category + "_plausible " + "nopre:         " + print3([round(x, 3) for x in nopre_means]))
                print("")


if args.r:
    for name, lst in [("yespre", yespre_totals), ("pre", pre_totals), ("nopre", nopre_totals)]:
        for index in list(range(len(lst))):
            lst[index] = lst[index]/12.0
        print(name + " <- c(" + ", ".join([str(round(x, 4)) for x in lst]) + ")")

