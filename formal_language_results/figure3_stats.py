
import statistics
import math

def get_average_fscore(filename):
    for line in open(filename, "r"):
        if "Average LM Y&P F-score:" in line:
            parts = line.strip().split()
            fscore = float(parts[-1])
    
    return fscore

def get_average_memorization_fscore(filename):
    for line in open(filename, "r"):
        if "Average memorization Y&P F-score:" in line:
            parts = line.strip().split()
            fscore = float(parts[-1])

    return fscore


train_sizes = ["1", "10", "100", "1000", "10000"]

# Get stats for Yang & Piantadosi's Bayesian model
print("Bayesian learner (Yang & Piantadosi)")
fi = open("yandp/yp_results.txt", "r")

results = {}
names = {}

posteriors = {}
fscores = {}

for line in fi:
    parts = line.strip().split()

    name = parts[0]
    factors = parts[1]
    training_size = parts[2]
    prec = float(parts[3])
    rec = float(parts[4])
    posterior = float(parts[5])

    if prec+rec == 0:
        fscore = 0
    else:
        fscore = 2*prec*rec / (prec+rec)

    names[name] = 1

    key = (name, training_size, factors)
    posteriors[key] = posterior
    fscores[key] = fscore


for training_size in train_sizes:

    total_fscore = 0
    count_fscore = 0


    for name in names:

        best_factors = None
        best_posterior = -1*math.inf
        for factors in ["1", "2", "3", "4"]:
            if posteriors[(name, training_size, factors)] > best_posterior:
                    best_factors = factors
                    best_posterior = posteriors[(name, training_size, factors)]

        this_fscore = fscores[(name, training_size, best_factors)]

        total_fscore += this_fscore
        count_fscore += 1

    print(training_size, total_fscore*1.0/count_fscore)

print("")


# Get stats for prior-trained models
print("Prior-trained")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(40):
        fi = "prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    if len(fscores[train_size]) != 40:
        print("Prior trained", train_size, len(fscores[train_size]), 40)
        15/0
    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))
    #print(statistics.stdev(fscores[train_size]))
    #print(sorted(fscores[train_size]))

print("")

# Get stats for standard models
print("Standard")
fscores = {}

for train_size, topp in [("1", "0.99"), ("10", "0.99"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(40):
        fi = "standard/random_meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    if len(fscores[train_size]) != 40:
        print("Standard", train_size, len(fscores[train_size]), 40)
        15/0

    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))

print("")


# Get stats for memorization
print("Memorization")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(40):
        fi = "prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_memorization_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    if len(fscores[train_size]) != 40:
        print("Memorization", train_size, len(fscores[train_size]), 40)
        15/0

    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))

print("")




# Get stats for pre-trained models
print("Pre-trained")
fscores = {}

for train_size, topp in [("1", "1.0"), ("10", "0.99"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores[train_size] = []
    for index in range(40):
        fi = "pre_trained/pseudo_meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscore = get_average_fscore(fi)
        fscores[train_size].append(fscore)

for train_size in train_sizes:
    if len(fscores[train_size]) != 40:
        print("Pre-trained", train_size, len(fscores[train_size]), 40)
        15/0

    print(train_size, statistics.mean(fscores[train_size]), min(fscores[train_size]), max(fscores[train_size]))

print("")




