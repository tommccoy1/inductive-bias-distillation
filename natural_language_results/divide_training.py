
n_files = 1024

fi = open("pretraining_divided/train.txt", "r")
n_lines = 0
for line in fi:
    n_lines += 1

fi.close()

lines_per_file = n_lines // n_files


fi = open("pretraining_divided/train.txt", "r")
current_file_number = 0
current_fo = open("pretraining_divided/train_" + str(current_file_number) + ".txt", "w")

for index, line in enumerate(fi):

    if index % lines_per_file == 0 and index != 0:
        current_fo.close()
        current_file_number += 1
        current_fo = open("pretraining_divided/train_" + str(current_file_number) + ".txt", "w")

    current_fo.write(line)

current_fo.close()



