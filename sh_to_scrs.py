



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--command_file", help="file of commands to turn into MARCC scripts", type=str, default=None)
parser.add_argument("--commands_per_script", help="number of commands to include in each script file", type=int, default=1)
parser.add_argument("--scr_type", help="script type: cpu or gpu", type=str, default="gpu")
parser.add_argument("--repetitions", help="number of times to repeat each command", type=int, default=1)
parser.add_argument("--cd_up", action="store_true", help="Whether to cd to the parent directory for running the command")
parser.add_argument("--tasks", help="ntasks per node", type=int, default=1)
parser.add_argument("--hours", help="hours for the job to run", type=str, default="72")
args = parser.parse_args()



fi = open(args.command_file, "r")
prefix = args.command_file[:-3]

to_run = open(prefix + "_run_all.sh", "w")


commands = []
for index, line in enumerate(fi):
    command = line.strip()
    if command != "":
        commands.append(command)

command_lists = []
current_list = []


for command in commands:
    current_list.append(command)
    if len(current_list) == args.commands_per_script:
        command_lists.append(current_list[:])
        current_list = []


if len(current_list) > 0:
    command_lists.append(current_list[:])


for index, command_list in enumerate(command_lists):
    this_name = prefix + "_" + str(index+1)
    fo = open(this_name + ".scr", "w")
    to_run.write("sbatch " + this_name + ".scr\n")

    fo.write("#!/bin/bash\n")
    fo.write("#SBATCH --job-name=" + this_name + "\n")

    fo.write("#SBATCH --time=" + args.hours + ":00:00\n")
    if args.scr_type == "gpu":
        fo.write("#SBATCH --gres=gpu:1\n")
    fo.write("#SBATCH --ntasks=1\n")
    fo.write("#SBATCH --cpus-per-task=1\n")
    fo.write("#SBATCH --mem-per-cpu=4G\n")

    fo.write("#SBATCH --mail-type=begin\n")
    fo.write("#SBATCH --mail-type=end\n")
    fo.write("#SBATCH --mail-type=fail\n")
    fo.write("#SBATCH --mail-user=tm4633@princeton.edu\n")
    fo.write("#SBATCH --output=" + this_name + ".log\n")
    fo.write("#SBATCH --error=" + this_name + ".err\n\n")
    fo.write("module purge\nmodule load anaconda3/2022.5\nsource ../mamlized_seq2seq/.venv/bin/activate\n\n")


    for command in command_list:
        for _ in range(args.repetitions):
            fo.write(command + "\n")




