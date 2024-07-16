
import statistics
import math

def get_average_fscore(filename):
    fscore = None
    for line in open(filename, "r"):
        if "Average LM Y&P F-score:" in line:
            parts = line.strip().split()
            fscore = float(parts[-1])

    return fscore


def get_fscore_by_language(filename):
    fscore_dict = {}
    for line in open(filename, "r"):
        if "Language:" in line:
            language_name = line.strip().split()[-1]
        
        if "LM precision, recall, fscore:" in line:
            fscore = float(line.strip().split()[-1])
    
            fscore_dict[language_name] = fscore

    return fscore_dict


def get_memorization_fscore_by_language(filename):
    fscore_dict = {}
    for line in open(filename, "r"):
        if "Language:" in line:
            language_name = line.strip().split()[-1]
        
        if "Memorization precision, recall, fscore:" in line:
            fscore = float(line.strip().split()[-1])
    
            fscore_dict[language_name] = fscore

    return fscore_dict




def get_average_memorization_fscore(filename):
    for line in open(filename, "r"):
        if "Average memorization Y&P F-score:" in line:
            parts = line.strip().split()
            fscore = float(parts[-1])

    return fscore

all_languages = ["An", "AB", "ABn", "AAA", "AAAA", "AnBm", "GoldenMean", "Even",
           "ApBAp", "ApBApp", "AsBAsp", "CountA2", "CountAEven", "aABb", "AnBn", "Dyck", 
           "AnB2n", "AnCBn", "AnABn", "ABnABAn", "AnBmCn", "AnBmA2n", "AnBnC2n", "AnBmCm", 
           "AnBmCnpm", "AnBmCnm", "AnBk", "AnBmCmAn", "AnB2nC3n", "AnBnp1Cnp2", "AnUBn", "AnUAnBn", 
           "ABnUBAn", "XX", "XXX", "XY", "XXR", "XXI", "XXRI", "An2", 
           "AnBmCnDm", "AnBmAnBm", "AnBmAnBmCCC", "AnBnCn", "AnBnCnDn", "AnBnCnDnEn", "A2en", "ABnen", 
            "Count", "ChineseNumeral", "ABAnBn", "ABaaaAB", "Unequal", "Bach2", "Bach3", "WeW"]


ypname2name = {'an': "An", 'sigmaplus': "AB", 'abn': "ABn", 'aaa': "AAA", 'aaaa': "AAAA", 'anbm': "AnBm", 'goldenmean': "GoldenMean", 'even': "Even", 'apbap': "ApBAp", 'apbapp': "ApBApp", 'asbasp': "AsBAsp", 'sigmaaaasigma': "ABaaaAB", 'counta2': "CountA2", 'countaeven': "CountAEven", 'asigmab': "aABb", 'anbn': "AnBn", 'dyck': "Dyck", 'anb2n': "AnB2n", 'ancbn': "AnCBn", 'anabn': "AnABn", 'abnaban': "ABnABAn", 'anbmcn': "AnBmCn", 'anbma2n': "AnBmA2n", 'anbnc2n': "AnBnC2n", 'sigmaanbn': "ABAnBn", 'anbmcm': "AnBmCm", 'anbmcnpm': "AnBmCnpm", 'anbmcnm': "AnBmCnm", 'anbnpm': "AnBk", 'anbmcman': "AnBmCmAn", 'anb2nc3n': "AnB2nC3n", 'anbnp1cnp2': "AnBnp1Cnp2", 'anubn': "AnUBn", 'anuanbn': "AnUAnBn", 'abnuban': "ABnUBAn", 'xx': "XX", 'xxx': "XXX", 'xy': "XY", 'xxr': "XXR", 'xxi': "XXI", 'xxri': "XXRI", 'unequal': "Unequal", 'bach2': "Bach2", 'bach3': "Bach3", 'xmagx': "WeW", 'anpower2': "An2", 'anbmcndm': "AnBmCnDm", 'anbmanbm': "AnBmAnBm", 'anbmanbmccc': "AnBmAnBmCCC", 'anbncn': "AnBnCn", 'anbncndn': "AnBnCnDn", 'anbncndnen': "AnBnCnDnEn", 'a2tothen': "A2en", 'abnsquared': "ABnen", 'count': "Count", 'chinesenumeral': "ChineseNumeral"}

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


yandp_fscores_overall = {}
for training_size in train_sizes:

    yandp_fscores_overall[training_size] = {}

    for name in names:

        best_factors = None
        best_posterior = -1*math.inf
        for factors in ["1", "2", "3", "4"]:
            if posteriors[(name, training_size, factors)] > best_posterior:
                    best_factors = factors
                    best_posterior = posteriors[(name, training_size, factors)]

        this_fscore = fscores[(name, training_size, best_factors)]
        yandp_fscores_overall[training_size][ypname2name[name]] = this_fscore



yp_scores = []
for train_size in train_sizes:
    for language in all_languages:
        yp_scores.append(yandp_fscores_overall[train_size][language])
print(yp_scores)
#print(fscores_overall)


print("")


# Get stats for prior-trained models
print("Prior-trained")
fscores_overall = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores_overall[train_size] = {}
    fscore_list = []
    for index in range(40):
        fi = "prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscores_by_language = get_fscore_by_language(fi)
        fscore_list.append(fscores_by_language)

    for language in fscores_by_language:
        fscores = []
        for index in range(40):
            if language in fscore_list[index]:
                fscores.append(fscore_list[index][language])
        fscores_overall[train_size][language] = statistics.mean(fscores)


prior_scores = []
for train_size in train_sizes:
    for language in all_languages:
        prior_scores.append(fscores_overall[train_size][language])
print(prior_scores)
print("")

# Get stats for standard models
print("Standard")
fscores_overall = {}

for train_size, topp in [("1", "0.99"), ("10", "0.99"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores_overall[train_size] = {}
    fscore_list = []
    for index in range(40):
        fi = "standard/random_meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscores_by_language = get_fscore_by_language(fi)
        fscore_list.append(fscores_by_language)

    for language in fscores_by_language:
        fscores = []
        for index in range(40):
            if language in fscore_list[index]:
                fscores.append(fscore_list[index][language])
        fscores_overall[train_size][language] = statistics.mean(fscores)


prior_scores = []
for train_size in train_sizes:
    for language in all_languages:
        if language in fscores_overall[train_size]:
            prior_scores.append(fscores_overall[train_size][language])
print(prior_scores)

print("")



# Get stats for memorization
print("Memorization")
fscores_overall = {}

for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores_overall[train_size] = {}
    fscore_list = []
    for index in range(40):
        fi = "prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscores_by_language = get_memorization_fscore_by_language(fi)
        fscore_list.append(fscores_by_language)

    for language in fscores_by_language:
        fscores = []
        for index in range(40):
            if language in fscore_list[index]:
                fscores.append(fscore_list[index][language])
        fscores_overall[train_size][language] = statistics.mean(fscores)


prior_scores = []
for train_size in train_sizes:
    for language in all_languages:
        prior_scores.append(fscores_overall[train_size][language])
print(prior_scores)
print("")


# Get stats for pre-trained models
print("Pre-trained")
fscores_overall = {}

for train_size, topp in [("1", "1.0"), ("10", "0.99"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    fscores_overall[train_size] = {}
    fscore_list = []
    for index in range(40):
        fi = "pre_trained/pseudo_meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log"
        fscores_by_language = get_fscore_by_language(fi)
        fscore_list.append(fscores_by_language)

    for language in fscores_by_language:
        fscores = []
        for index in range(40):
            fscores.append(fscore_list[index][language])
        fscores_overall[train_size][language] = statistics.mean(fscores)


prior_scores = []
for train_size in train_sizes:
    for language in all_languages:
        prior_scores.append(fscores_overall[train_size][language])
print(prior_scores)

print("")




