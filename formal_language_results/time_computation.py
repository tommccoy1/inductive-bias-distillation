

import math

#2023-04-22 08:11:17,132 INFO     Description: Sigma+ A^ B^n (except it seems to be missing some short ones, like aab)


def timestamp_to_seconds(timestamp, correction=False):
    parts = timestamp.split(":")
    hours = float(parts[0])
    if correction and hours == 0:
        hours = 24.0
    minutes = float(parts[1])
    seconds = float(parts[2].replace(",", "."))

    total = 3600*hours + 60*minutes + seconds
    return total

def compute_seconds(time1, time2):

    start = timestamp_to_seconds(time1)
    end = timestamp_to_seconds(time2)

    if end < start:
        #print("BACKWARDS", time1, time2, start, end)
        start = timestamp_to_seconds(time1, correction=True)
        end = timestamp_to_seconds(time2, correction=True)

    if end < start:
        print("BACKWARDS", time1, time2, start, end)

    return end - start


min_time = math.inf
max_time = -1*math.inf

total_count = 0
for train_size, topp in [("1", "1.0"), ("10", "1.0"), ("100", "0.99"), ("1000", "0.99"), ("10000", "0.99")]:
    for index in range(40):
        fi = open("prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log", "r")

        inner_count = 0
        for line in fi:
            if "Description" in line:
                parts = line.strip().split()
                start_time = parts[1]
            elif "DONE TRAINING" in line:
                parts = line.strip().split()
                end_time = parts[1]

                time = compute_seconds(start_time, end_time)

                if time < min_time:
                    min_time = time
                if time > max_time:
                    max_time = time
                total_count += 1
                inner_count += 1
        if inner_count != 56:
            print("prior_trained/meta_lm_hidden1024_" + str(index) + "_eval_formal_" + train_size + "_language_list_topp" + topp + "_nsamples1000000_for_paper.log")
            print(inner_count)
            print("")

print("Total count:", total_count, 40*56*5)

print("Minimum:", min_time)
print("Maximum:", max_time)

