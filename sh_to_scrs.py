

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--command_file", help="file of commands to turn into MARCC scripts", type=str, default=None)
parser.add_argument("--commands_per_script", help="number of commands to include in each script file", type=int, default=1)
parser.add_argument("--partition", help="partition to run commands in", type=str, default="shared")
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
    fo.write("#SBATCH --partition=" + args.partition + "\n")

    if "gpu" in args.partition:
        fo.write("#SBATCH --time=" + args.hours + ":0:0\n")
        fo.write("#SBATCH --gres=gpu:1\n")
        fo.write("#SBATCH --ntasks-per-node=" + str(args.tasks) + "\n")
    else:
        fo.write("#SBATCH --time=" + args.hours + ":0:0\n")
        fo.write("#SBATCH --nodes=1\n")
        fo.write("#SBATCH --ntasks-per-node=" + str(args.tasks) + "\n")

    fo.write("#SBATCH --mail-type=ALL\n")
    fo.write("#SBATCH --output=" + this_name + ".log\n")
    fo.write("#SBATCH --error=" + this_name + ".err\n\n")


    if args.cd_up:
        fo.write("cd ..\n")

    fo.write("module load Python/3.10.8-GCCcore-12.2.0\nsource .venv/bin/activate\n\n")

    for command in command_list:
        for _ in range(args.repetitions):
            fo.write(command + "\n")





