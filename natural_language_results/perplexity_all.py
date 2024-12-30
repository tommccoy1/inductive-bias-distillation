
import statistics
import os

directory = "perplexity/"

import numpy as np
import scipy.stats

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)

    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return h


for model_size in ["16", "32", "64", "128", "256", "512", "1024"]:
    for fraction in ["sixtyfourth", "thirtysecond", "sixteenth", "eighth", "quarter", "half", "full"]:
        yes_mean = 0
        no_mean = 0

        for pre in ["nopre", "yespre"]:

            wrong_length = False
            values1 = []
            recorded_values = []

            for index in range(20):

                if pre == "nopre":
                    model_name = "_".join(["bestparams", "adapt", "hidden" + model_size, "pretraining", fraction, pre, str(index), "eval"]) + ".log"
                else:
                    model_name = "_".join(["bestparams", "adapt", "hidden" + model_size, "pretraining", fraction, pre + str(index), "0", "eval"]) + ".log"


                fi = open(directory + model_name, "r")

                for line in fi:
                    if "Test (strided) perplexity:" in line:
                        perplexity = float(line.strip().split()[-1])
                        recorded_values.append(perplexity)

            if len(recorded_values) != 20:
                print("WRONG LENGTH", model_name)
                15/0

            print(model_size, fraction, pre, round(statistics.mean(recorded_values),1), "+-", round(confidence_interval(recorded_values), 2))
            if pre == "yespre":
                yes_mean = statistics.mean(recorded_values)
                yes_values = recorded_values[:]
            else:
                no_mean = statistics.mean(recorded_values)
                no_values = recorded_values[:]


        print(round((no_mean - yes_mean) / no_mean, 3))
        print(scipy.stats.ttest_ind(a=np.array(yes_values), b=np.array(no_values), equal_var=False))

        print("")




