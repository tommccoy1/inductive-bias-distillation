
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



total_scores_yes = []
total_scores_no = []

for index in range(n_runs):
    total_score_yes = 0
    total_score_no = 0
    n_categories = 0
    for category in nopre_values:

        if category.startswith("recursion"):
            continue
   
        if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
            print("WRONG COUNT")
            15/0

        total_score_yes += yespre_values[category][index]
        total_score_no += nopre_values[category][index]
        n_categories += 1


    total_scores_yes.append(total_score_yes*1.0/n_categories)
    total_scores_no.append(total_score_no*1.0/n_categories)

print("No scores", total_scores_no)
print("Yes scores", total_scores_yes)
print("")
print("Mean yes", statistics.mean(total_scores_yes))
print("Mean no", statistics.mean(total_scores_no))
print(scipy.stats.ttest_ind(a=np.array(total_scores_no), b=np.array(total_scores_yes), equal_var=False))
pvalue = scipy.stats.ttest_ind(a=np.array(total_scores_no), b=np.array(total_scores_yes), equal_var=False).pvalue
print("P-value", pvalue)
