
import statistics
import math
import os

def get_average_fscore(filename):
    #print(filename)
    fscore = None

    try:
        for line in open(filename, "r"):
            if "Average LM Y&P F-score:" in line:
                parts = line.strip().split()
                fscore = float(parts[-1])
    except:
        pass

    if fscore is None:
        print(filename)

    return fscore

def get_average_memorization_fscore(filename):
    for line in open(filename, "r"):
        if "Average memorization Y&P F-score:" in line:
            parts = line.strip().split()
            fscore = float(parts[-1])

    return fscore


train_sizes = ["1", "10", "100", "1000", "10000"]

# Get stats for prior-trained models
print("Prior-trained: All primitives")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(20):
        fi = "all_primitives/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))
    #print(statistics.stdev(fscores[train_size]))
    #print(sorted(fscores[train_size]))

print("")

# Get stats for prior-trained models minus synchrony
print("Prior-trained: Minus synchrony")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(20):
        if train_size in ["1", "100", "10000"]:
            fi = "no_synchrony/meta_lm_no_sync_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_for_paper.log"
        else:
            fi = "no_synchrony/meta_lm_no_sync_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_matched.log"
        fscore = get_average_fscore(fi)
        if fscore is not None:
            fscores[train_size].append(fscore)

for train_size in train_sizes:
    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))
    #print(statistics.stdev(fscores[train_size]))
    #print(sorted(fscores[train_size]))

print("")

# Get stats for prior-trained models minus synchrony
print("Prior-trained: Minus recursion")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(20):
        if train_size in ["10"]:
            fi = "no_recursion/meta_lm_no_recursion_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_for_paper.log"
        else:
            fi = "no_recursion/meta_lm_no_recursion_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_matched.log"
        fscore = get_average_fscore(fi)
        if fscore is not None:
            fscores[train_size].append(fscore)

for train_size in train_sizes:
    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))
    #print(statistics.stdev(fscores[train_size]))
    #print(sorted(fscores[train_size]))

print("")



# Get stats for standard models
print("Standard")
fscores = {}

for train_size, topp in [("1", "0.99"), ("10", "0.99"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(20):
        fi = "standard/random_meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))

print("")


# Get stats for memorization
print("Memorization")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(20):
        fi = "all_primitives/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_recursion_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_memorization_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    print(train_size, statistics.mean(fscores[train_size]))

print("")




