
import os
import random

dataset_fractions = [("pretraining_full", 1025),
                     ("pretraining_half", 512),
                     ("pretraining_quarter", 256),
                     ("pretraining_eighth", 128),
                     ("pretraining_sixteenth", 64),
                     ("pretraining_thirtysecond", 32),
                     ("pretraining_sixtyfourth", 16)
                    ]

for name, subfile_count in dataset_fractions:
    print(name)
    for index in range(40):

        if subfile_count == 1025:
            all_indices = list(range(1025))
        else:
            # Don't include small final subdivision
            # unless we need to
            all_indices = list(range(1024))

        random.shuffle(all_indices)
        indices_to_use = all_indices[:subfile_count]

        directory = name + "_" + str(index)
        os.makedirs(directory)

        fo_train = open(directory + "/train.txt", "w")
        for index_to_use in indices_to_use:
            fi_train = open("pretraining_divided/train_" + str(index_to_use) + ".txt", "r")
            for line in fi_train:
                fo_train.write(line.strip() + "\n")
            fi_train.close()
        fo_train.close()

        fo_valid = open(directory + "/valid.txt", "w")
        fi_valid = open("pretraining_divided/valid.txt", "r")
        for line in fi_valid:
            fo_valid.write(line.strip() + "\n")
        fo_valid.close()
        fi_valid.close()

        fo_test = open(directory + "/test.txt", "w")
        fi_test = open("pretraining_divided/test.txt", "r")
        for line in fi_test:
            fo_test.write(line.strip() + "\n")
        fo_test.close()
        fi_test.close()

        fo_vocab = open(directory + "/vocab.txt", "w")
        fi_vocab = open("pretraining_divided/vocab.txt", "r")
        for line in fi_vocab:
            fo_vocab.write(line.strip() + "\n")
        fo_vocab.close()
        fi_vocab.close()











