
import math
import statistics

fi = open("yp_results.txt", "r")

results = {}
names = {}
training_sizes = ["1", "10", "100", "1000", "10000"]

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


for training_size in training_sizes:
    
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

    print(training_size, total_fscore*1.0/count_fscore, count_fscore)


