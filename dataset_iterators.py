

import math
import random
from collections import Counter
import logging

import numpy as np
from utils import *
from yandp import *
from scfg import random_sync

import signal
from contextlib import contextmanager

# For setting a time limit on a process
# For some meta-grammars, you can get stuck in non-terminating
# recursion (or in recursion that will eventually terminate, but only
# after recursing much more than we want). This time limit allows us
# to cut a process short if it is taking too long, to avoid such scenarios.
# Code from here: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# Simple example of a dataset creation function
# Within a dataset, there is a fixed first token,
# a fixed last token, and a fixed length
def simple_dataset(vocab_size):

    def create_simple_dataset(seed, remembered_languages=None):
        random.seed(seed)

        # The seq length is really 2 plus this,
        # after we've added in the first and
        # last elements
        seqsize = random.choice(list(range(5))) + 1

        first = random.choice(list(range(vocab_size)))
        last = random.choice(list(range(vocab_size)))

        seqs = []
        for i in range(101):

            seq = [first]
            for _ in range(seqsize):
                seq.append(random.choice(list(range(vocab_size))))

            seq.append(last)

            seqs.append(" ".join([str(elt) for elt in seq]))

            dataset = {}
        dataset["train"] = seqs[:1]
        dataset["test"] = seqs[1:]

        dataset["train_batch_size"] = 1
        dataset["eval_batch_size"] = 100

        return dataset

    return create_simple_dataset

# Create a dataset sampled from Yang & Piantadosi's meta-grammar
def yandp_dataset(param_file, n_test=10, batch_size=100, eval_batch_size=10, max_batches_per_language=1):

    param_dict, basic_primitives = primitives_from_file(param_file)

    def create_yandp_dataset(seed, max_batches_per_language=max_batches_per_language):

        # Use a random seed to ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Whether the dataset has been successfully created
        dataset_created = False

        # Number of batches to include in this dataset
        num_batches = np.random.randint(max_batches_per_language) + 1

        # For generating a particular dataset, make this many attempts to generate
        # sequences (i.e., a max of 10 attempts per sequence needed). If you still
        # haven't generated a dataset by then, reject this dataset
        max_attempts = (num_batches*batch_size + n_test)*10

        while not dataset_created:

            # Try to generate a grammar. We need the try/except because
            # sometimes the recursion limit is exceeded when generating it
            try:
                sys.setrecursionlimit(40)
                hyp = Hypothesis(geometric_p=param_dict["GEOMETRIC_P"], terminal_w=param_dict["TERMINAL_W"], sigma_w=param_dict["SIGMA_W"],
                        prob_w_num=param_dict["PROB_W"], factor_w=param_dict["FACTOR_W"], x_w=param_dict["X_W"],
                        prob_divisions=100, basic_primitives=basic_primitives)
                sys.setrecursionlimit(1000)
            except:
                # Failed to generate a grammar. Loop back to the start of the
                # while loop and try again
                continue

            # List of sequences sampled from the grammar
            data_list = []

            # Make max_attempts attempts per batch
            for attempt_number in range(max_attempts):
                if len(data_list) >= batch_size*num_batches + n_test:
                    # We've generated all the examples we need, so we're done!
                    break

                if attempt_number > max_attempts and len(data_list) == 0:
                    # If we haven't generated a single successful example by
                    # now, it's unlikely we'll be able to make a full dataset,
                    # so we break
                    break

                # Attempt to generate a single example. We need the try/except
                # to catch errors from exceeding the recursion depth
                try:
                    sys.setrecursionlimit(40)
                    example = hyp.to_call([])
                    sys.setrecursionlimit(1000)

                    data_list.append(example)
                
                except:
                    pass

            # We have only successfully created the dataset if we have produced the number
            # of examples that we need
            if len(data_list) >= batch_size*num_batches + n_test:
                dataset_created = True
        
        # Return the dataset
        # We don't divide the training into batches at this point - this is done later
        dataset = {}
        dataset["train"] = data_list[:batch_size*num_batches]
        dataset["test"] = data_list[batch_size*num_batches:]
        dataset["hypothesis"] = hyp

        dataset["train_batch_size"] = batch_size
        dataset["eval_batch_size"] = eval_batch_size

        return dataset

    # Return the function that creates datasets
    return create_yandp_dataset


# Create a dataset from .txt files specifying a formal language (i.e., the
# evaluation languages for Yang & Piantadosi)
# - list_of_langs: .txt file in the directory formal_languages/ listing the languages to be used in this dataset
# - training_size, test_size: the training and test size for each episode in the dataset
# - batch_size, eval_batch_size: batch sizes for each episode's training and evaluation
def formal_dataset(list_of_langs, training_size=100, test_size=10, batch_size=100, eval_batch_size=10):

    # Create our list of languages (with their descriptions)
    fi = open("formal_languages/" + list_of_langs + ".txt", "r")

    langs = []
    for line in fi:
        line = line.strip()
        if len(line) == 0:
            continue

        parts = line.split("\t")
        langs.append((parts[0], parts[1]))

    # Alphabet for converting from chars to ints
    alphabet = {}
    alphabet["a"] = "0"
    alphabet["b"] = "1"
    alphabet["c"] = "2"
    alphabet["d"] = "3"
    alphabet["e"] = "4"

    alphabet["("] = "0"
    alphabet[")"] = "1"


    def create_formal_dataset(seed):

        # Use a seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        filename, description = langs[seed%len(langs)]

        # From this file, retrieve all strings in the language
        fi = open("formal_languages/Fleet/Models/FormalLanguageTheory-Complex/data/" + filename + ".txt", "r")

        strings = []
        counts = []
        for line in fi:
            parts = line.strip().split()
            
            if len(parts) == 1:
                seq = ""
                count = int(parts[0])
            else:
                seq = parts[0]
                count = int(parts[1])

            strings.append(seq)
            counts.append(count)

        # Convert the strings from chars to ints
        new_strings = []

        for string in strings:
            new_string = []
            for char in string:
                new_string.append(alphabet[char])
            new_strings.append(" ".join(new_string))
            

        # Get the training set from the relevant file
        fi_train = open("formal_languages/Fleet/Models/FormalLanguageTheory-Complex/data/" + filename + "-" + str(training_size) + ".txt", "r")

        training_list = []
        for line in fi_train:
            parts = line.strip().split()
                
            seq = parts[0]
            count = int(parts[1])

            new_seq = []
            for char in seq:
                new_seq.append(alphabet[char])
            new_seq = " ".join(new_seq)

            for _ in range(count):
                training_list.append(new_seq)

        random.shuffle(training_list)

        dataset = {}
        dataset["train"] = training_list

        # Produce the test set
        corpus_test = new_strings[:25]
        dataset["test"] = corpus_test
            
        dataset["name"] = filename
        dataset["description"] = description
        dataset["all_strings"] = list(zip(new_strings, counts))

        dataset["train_batch_size"] = batch_size
        dataset["eval_batch_size"] = eval_batch_size

        return dataset

    return create_formal_dataset


# Dataset produced from our version of a synchronized CFG meta-grammar
def scfg_dataset(n_test=10, batch_size=100, eval_batch_size=10, max_batches_per_language=1, withheld_languages=None, withheld_seq_dict=None):

    def create_scfg_dataset(seed, max_batches_per_language=max_batches_per_language, remembered_languages=None):
        
        # Use a seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Whether we have successfully produced a dataset
        dataset_created = False

        # Number of batches in this dataset
        num_batches = np.random.randint(max_batches_per_language) + 1

        max_attempts = (batch_size*max_batches_per_language + n_test)*10
     
        attempts = 0
        while not dataset_created:
            attempts += 1

            try:
                # Avoid getting stuck in a very slow-to-generate language
                # by setting a time limit of 10 seconds
                with time_limit(10):

                    try:
                        # Attempt to generate a grammar for this language
                        sys.setrecursionlimit(40)
                        hyp = random_sync()
                        sys.setrecursionlimit(1000)
                    except:
                        # Failed to generate a grammar
                        continue

                    # Sample sentences from this grammar
                    data_list = []
                    attempts = 0

                    for attempt_number in range(max_attempts):

                        if len(data_list) >= batch_size*num_batches + n_test:
                            # We've produced all the sentences we need, so we are done!
                            break

                        if attempt_number > max_attempts and len(data_list) == 0:
                            # If we've gone this long without producing any sentences, we are unlikely
                            # to be able to produce a complete dataset, so we move on to the next grammar
                            break

                        # Produce a sequence
                        sys.setrecursionlimit(100)
                        example = hyp.to_call([])
                        sys.setrecursionlimit(1000)

                        data_list.append(example)

                    if len(data_list) >= batch_size*num_batches + n_test:
                        # We have as many examples as we need

                        if withheld_languages is None:
                            # If we're not withholding languages...we're done
                            dataset_created = True
                            
                        else:
                            # If we are withholding languages: Need to check if this
                            # one needs to be withheld

                            # Produce a name for this hypothesis, so that we can remember it in the future without
                            # needing to re-compute the F-score
                            hyp_name = ",".join([str(x) for x in hyp.sync_pattern]) + "\t" + "\t".join(hyp.rule_names)
                            if remembered_languages is not None:
                                # Check if this language has been evaluated for withholding before. If so, apply
                                # the same decision that was remembered for it
                                if hyp_name in remembered_languages:
                                    if remembered_languages[hyp_name]:
                                        dataset_created = True
                                        continue
                                    else:
                                        continue

                            # Produce 100,000 sentences from the grammar as our set of possible sentences
                            language_counter = Counter()
                            sys.setrecursionlimit(100)
                            for _ in range(100000):
                                example = hyp.to_call([])
                                language_counter.update([example])
                            sys.setrecursionlimit(1000)

                            # Check whether there is an F-score of 1.0 between the proposed language and any
                            # of the languages to be withheld. If so, withhold it; if not, use it.
                            if not withhold_based_on_fscore(language_counter, withheld_languages, data_list, withheld_seq_dict):
                                dataset_created = True
                                if remembered_languages is not None:
                                    remembered_languages[hyp_name] = True
                            else:
                                if remembered_languages is not None:
                                    remembered_languages[hyp_name] = False

            except:
                pass


        # Return the dataset
        dataset = {}

        # We don't divide the training into batches at this point - this is done later
        dataset["train"] = data_list[:batch_size*num_batches]
        dataset["test"] = data_list[batch_size*num_batches:]
        dataset["hypothesis"] = hyp
        dataset["n_batches"] = num_batches
        dataset["train_batch_size"] = batch_size
        dataset["eval_batch_size"] = eval_batch_size

        return dataset

    return create_scfg_dataset


if __name__ == "__main__":
    print("SIMPLE DATASET")
    create_simple_dataset = simple_dataset(10)
    for i in range(5):
        print(create_simple_dataset(i))
        print(create_simple_dataset(i))
        print("")

    print("")
    print("")
    print("Y&P DATASET")
    create_yandp_dataset = yandp_dataset("yandp_weights/yandp_params_uniform.txt", n_test=10, batch_size=10, eval_batch_size=10, max_batches_per_language=3)
    for i in range(5):
        print(create_yandp_dataset(i))
        print(create_yandp_dataset(i))
        print("")


    print("")
    print("")
    print("SCFG DATASET")
    create_scfg_dataset = scfg_dataset(n_test=5, batch_size=10, eval_batch_size=10, max_batches_per_language=3)
    for i in range(5):
        print(create_scfg_dataset(i))
        print(create_scfg_dataset(i))
        print("")


    print("")
    print("")
    print("FORMAL DATASET")
    create_formal_dataset = formal_dataset("language_list", training_size=10, test_size=5, batch_size=10, eval_batch_size=10)
    for i in range(5):
        print(create_formal_dataset(i))
        print(create_formal_dataset(i))
        print("")




