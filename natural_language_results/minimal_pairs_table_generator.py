
import statistics
import os

import jsonlines

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

def example_from_file(filename):
    fi = open(filename, "r")

    for line in fi:
        example = line.strip().split("\t")
        break
    fi.close()

    return example[0], example[1]

def example_from_jsonl(filename):
    with jsonlines.open(filename) as reader:
        for obj in reader:
            pair = [obj["sentence_good"], obj["sentence_bad"]]
            break

    return pair[0], pair[1]

def example_from_pairs(filename):
    fi = open(filename, "r")

    first = True
    second = False
    for line in fi:
        if first:
            bad_example = line.strip()
            first = False
            second = True
        elif second:
            good_example = line.strip()
            break
    fi.close()

    return good_example, bad_example



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

count_printed = 0
max_per_table = 10
for category in nopre_values:
   
    if len(nopre_values[category]) != n_runs or len(yespre_values[category]) != n_runs:
        print("WRONG COUNT")
        15/0

    yes_mean = statistics.mean(yespre_values[category])
    no_mean = statistics.mean(nopre_values[category])
    pvalue = scipy.stats.ttest_ind(a=np.array(nopre_values[category]), b=np.array(yespre_values[category]), equal_var=False).pvalue
   
    yes_conf = confidence_interval(yespre_values[category])
    no_conf = confidence_interval(nopre_values[category])

    yesmax = max(yespre_values[category])
    yesmin = min(yespre_values[category])
    nomax = max(nopre_values[category])
    nomin = min(nopre_values[category])
    yesstdev = statistics.stdev(yespre_values[category])
    nostdev = statistics.stdev(nopre_values[category])

    #print(category)
    #print("Nopre", sorted(nopre_values[category]))
    #print("Yespre", sorted(yespre_values[category]))
    #print("Nopre", no_mean) 
    #print("Yespre", yes_mean) 
    #print("Pvalue", pvalue) 
    #print("")

    if count_printed == 0:
        header1 = "\\begin{table}[]"
        header1b = "\\footnotesize"
        header2 = "\\centering"
        header3 = "\\begin{tabular}{llll} \\toprule"
        header4 = "Category & Standard & Prior-trained & $p$-value \\\\ \\midrule"

        print(header1)
        print(header1b)
        print(header2)
        print(header3)
        print(header4)


    line = " & ".join([category, '{0:.2f}'.format(no_mean) + " $\pm$ " + '{0:.2f}'.format(no_conf), '{0:.2f}'.format(yes_mean) + " $\pm$ " + '{0:.2f}'.format(yes_conf), '{0:.2f}'.format(pvalue)]) + "\\\\"
    print(line.replace("_", "\_"))

    if category != "overall":
        if args.dataset == "zorro":
            filename = "../Zorro/sentences/babyberta/" + category + ".txt"
            good_example, bad_example = example_from_pairs(filename)
        elif args.dataset == "blimp":
            filename = "../blimp_childes/" + category + ".jsonl" 
            good_example, bad_example = example_from_jsonl(filename)
        elif args.dataset == "scamp_plausible":
            filename = "../scamp/scamp_plausible/" + category + ".tsv"
            good_example, bad_example = example_from_file(filename)
        elif args.dataset == "scamp_implausible":
            filename = "../scamp/scamp_implausible/" + category + ".tsv"
            good_example, bad_example = example_from_file(filename)
    
        line2 = "\\multicolumn{4}{l}{\\phantom{*}\\textit{" + good_example + "}}\\\\"
        line3 = "\\multicolumn{4}{l}{*\\textit{" + bad_example + "}}\\\\"
        line4 = "\\\\"
        print(line2)
        print(line3)
        print(line4)

    count_printed += 1

    if count_printed == max_per_table:
        count_printed = 0
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{SCaMP$_{\\text{implausible}}$ results (1/3). For each model, the table shows the mean and a 95\\% confidence interval. For each category, there is one example of a minimal pair used to evaluate the category; a model is considered correct on an example if it assigns a higher probability to the sentence without an asterisk than the sentence with an asterisk.}")
        print("\\label{tab:my_label}")
        print("\\end{table}")
        print("")


if count_printed != 0:
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{SCaMP$_{\\text{implausible}}$ results (1/3). For each model, the table shows the mean and a 95\\% confidence interval. For each category, there is one example of a minimal pair used to evaluate the category; a model is considered correct on an example if it assigns a higher probability to the sentence without an asterisk than the sentence with an asterisk.}")
    print("\\label{tab:my_label}")
    print("\\end{table}")
    print("")




