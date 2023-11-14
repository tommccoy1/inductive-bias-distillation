

# Getting started
1. Set the values in `config.py` to whatever you want the default directories to be.

2. Create a python venv to run code in (the code was developed with Python 3.9.12, but the specific version probably doesn't matter much, as long as it is some version of Python 3):
```
python -m venv .venv
```

3. Activate the venv (the venv should be active any time you are running code from the repo).
```
source .venv/bin/activate
```

4. Install requirements
```
pip install -U pip setuptools wheel
pip install torch
pip install transformers
pip install higher
```


# Replicating formal language results (Figure 2)

0. The numbers used for Figure 2 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following in the directory `formal_language_results/`:
```
python figure2_stats.py
```
The steps listed below describe all of the scripts whose results are aggregated by this command.

1. Bayesian learner from Yang and Piantadosi: The directory `formal_language_results/yandp/` contains files for dealing with Yang and Piantadosi's results. `yandp_results.txt` gives the results from their [supplementary materials](https://www.pnas.org/doi/abs/10.1073/pnas.2021865119#supplementary-materials); the 6 columns are: (i) language description; (ii) number of factors; (iii) number of training examples; (iv) precision; (v) recall; and (vi) posterior. If you use these data, please cite [Yang and Piantadosi 2022](https://www.pnas.org/doi/abs/10.1073/pnas.2021865119).

2. Prior-trained neural network:
- First, perform inductive bias distillation. The commands that we used are in `formal_language_results/prior_trained/meta_training.sh`. Each command performs one run; we did 40 runs of the case with size 1024 and 20 runs for the other sizes. Each one takes 1 to 2 days on an A100 GPU.
- Then, adapt the prior-trained models to the formal languages. The commands that we used are in `formal_language_results/prior_trained/adapt_to_formal_languages.sh`. Each of these only needs to be run once and takes from 12 hours to a little over 1 day on an A100 GPU.
- The results of all the commands in the previous step are the files ending with `.log` in `formal_language_results/prior_trained/`. 

3. Memorization: The memorization numbers are in the same log files produced for the prior-trained models (step 2 above).

4. Standard neural network: The commands to adapt standard (randomly-initialized) networks to the formal languages are in `formal_language_results/standard/adapt_to_formal_languages.sh`. Each of these only needs to be run once and takes from 12 hours to a little over 1 day on an A100 GPU. The results of these commands are the `.log` files in `formal_language_results/standard/`.

5. Pre-trained neural network:
- First, pre-train neural networks. The command to do this is in `formal_language_results/pre_trained/pseudo_meta_training.sh`. We performed 40 runs of this command (each taking about 1 day on an A100 GPU).
- Then, adapt the pre-trained models to the formal languages. The commands that we used are in `formal_language_results/pre_trained/adapt_to_formal_languages.sh`. Each of these only needs to be run once and takes from 12 hours to a little over 1 day on an A100 GPU.
- The results of all the commands in the previous step are the files ending with `.log` in `formal_language_results/pre_trained/`.





# Replicating perplexity results (Figure 3)

0. The numbers used in Figure 3 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following in the directory `natural_language_results`. In the output, `no` means a standard neural network, while `yes` means a prior-trained one.
```
# Figure 3A
python perplexity_1024.py 

# Figure 3B
python perplexity_all.py
```
The steps listed below describe all of the scripts whose results are aggregated by these commands.


1. Create datasets: First download the CHILDES training set (not yet publicly available; please contact us if you need it). Then, move the scripts `divide_training.py` and `make_training_sets.py` into the `CHILDEs/` directory, and then run each of them:
```
python divide_training.py
python make_training_sets.py
``` 

2. Train language models on these datasets. The commands to do this are in `natural_language_results/perplexity/adapt_besthyps_nopre.sh` and `natural_language_results/perplexity/adapt_besthyps_yespre.sh`. Each command needs to be run once to replicate all of our results; these commands vary widely in execution time (from a few minutes to about 1 day, on an A100 GPU). This yields 20 runs for each cell in the heatmap (Fig. 3B) and 40 runs for the largest-scale case (1024 hidden units and the full training set).

3. Compute perplexity on the test set. The commands for doing this are in `natural_language_results/perplexity/test_perplexity_for_heatmap.sh` and `natural_language_results/perplexity/test_perplexity_1024full.sh`. The results of running these commands are in the `.log` files in `natural_language_results/perplexity/`.


# Getting statistics about Yang & Piantadosi's meta-grammar

Running `python yandp.py` samples languages from Yang & Piantadosi's prior and logs statistics in `yandp_stats/yandp_stats.log`. The file shows the proportion of sampled languages that only generate the empty string (ONLY EPSILON), that have a maximum length of 1 (MAX LENGTH 1), that only produce one unique string (ONLY ONE UNIQUE), and that have both more than one unique string and a maximum length greater than 1 (LONG AND DIVERSE).


# Replicating targeted evaluation results (Figure 4)

0. The numbers used in Figure 4 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following in the directory `natural_language_results`. In the output, `no` means a standard neural network, while `yes` means a prior-trained one.
```
# Zorro results
python minimal_pairs.py --dataset zorro

# BLiMP_CH results
python minimal_pairs.py --dataset blimp

# SCaMP (plausible) results
python minimal_pairs.py --dataset scamp_plausible

# SCaMP (implausible) results
python minimal_pairs.py --dataset scamp_implausible

# Recursion results
python recursion.py
```
The steps listed below describe all of the scripts whose results are aggregated by these commands.


1. Download the Zorro dataset by running the following command in the main directory:
```
git clone https://github.com/phueb/Zorro.git
```

2. The BLiMP_CHILDES dataset is already present in this repo, in the folder `blimp_childes`. It was created by running the BLiMP generation pipeline from [https://github.com/alexwarstadt/data_generation/tree/blimp](https://github.com/alexwarstadt/data_generation/tree/blimp) except with the vocabulary (`vocabulary.csv` in that repo) filtered to only include words that appear at least 10 times in our CHILDES training set. If you use this dataset, please cite [the BLiMP paper](https://aclanthology.org/2020.tacl-1.25/) by Warstadt et al. 

3. The SCaMP dataset, and the recursion and priming datasets, are already present in this repo, in the folders `scamp/scamp_plausible`, `scamp/scamp_implausible`. `scamp/recursion`, and `scamp/scamp_priming`. See `scamp/README.md` for more information.

4. To evaluate on all of these datasets, run the scripts in `natural_language_results/targeted_evaluations/targeted_evals.sh`.


# Replicating targeted evaluation results in the supplement

Figure 2 in the supplement compares the prior-trained and standard network in a way that controls their hyperparameters. These numbers are produced by running the following:

```
# Zorro results
python minimal_pairs.py --dataset zorro --same

# BLiMP_CH results
python minimal_pairs.py --dataset blimp --same

# SCaMP (plausible) results
python minimal_pairs.py --dataset scamp_plausible --same

# SCaMP (implausible) results
python minimal_pairs.py --dataset scamp_implausible --same

# Recursion results
python recursion.py --same
```

# Getting formal language results by language

The supplement contains results for all of the individual formal languages that are averaged across in Figure 2 of the main paper. The numbers used to produce the individual-language figure are created by running `python stats_by_language.py`.





# Description of the pipeline

1. First, you need to create a dataloader. The dataloader should contain 3 data splits (training, validation, and test). Each split should be an iterator where, as you iterate over it, it returns one batch at a time. Dataloaders are defined in `dataloading.py`

    a. For standard training, each batch will be one input to your model. It should be a dictionary containing the input as well as the target output that this input should have. An example of a standard dataloader is `LMDatasetFromFiles`.

    b. For meta training, each batch corresponds to one episode of metatraining. Therefore, it should contain both the training set for this episode and the test set for this episode. Specifically, the batch will be a dictionary with the following keys: `training_batches`: a list of batches (set up just like a standard batch in 1a - each of these batches should contain the model's input for that batch and the target output). `test_input_ids`: The inputs for the test set for this episode. `test_labels`: The labels for the test set for this episode. An example of a meta dataloader is `MetaLMDataset`.

2. Then, you need to create a model. The model should take in a single standard batch (as defined in 1a: a dictionary containing the inputs and the target outputs for that batch). Then it should return the model's predicted output and the model's loss (when its predicted output is compared to the target output). `RNNLM` gives an example of a model. 

3. Then, you need to train the model. This is done with a trainer (or metatrainer) from `training.py`, which takes in the model and dataset (and training parameters like the learning rate or number of epochs) and trains the model on that dataset. If your model and dataset are set up as described above, then you might not need to change or add anything to `training.py`.

4. Finally, you need to evaluate your model. For this, you will have to write functions that define whatever evaluations you want to run.

5. To put it all together, you can create a single script (like `lm_train.py` or `meta_train.py`) which first instantiates the dataset and the model, then trains the model, and then evaluates the model.

# Description of files

- `lm_train.py`: Train a language model with standard training
- `meta_train.py`: Meta-train a language model
- `config.py`: Some global variables
- `dataloading.py`: Code for loading and preparing general LM data
- `dataset_iterators.py`: Functions that yield a language sampled from each meta-grammar
- `evaluations.py`: Evaluation functions (beyond the loss automatically returned by models)
- `lr_scheduler.py`: Functions for learning rate scheduling
- `models.py`: Defining model architectures
- `utils.py`: Some miscellaneous helper functions
- `scfg.py`, `yandp.py`: Specifying two metagrammars. `scfg.py` is our prior based on formal language primitives; `yandp.py` is our reimplementation of Yang & Piantadosi's prior
- `training.py`: Classes for (meta-)training models
- `yandp_weights/yandp_params_uniform.txt`: PCFG weights for use in Yang and Piantadosi's meta-grammar
- `sh_to_scrs.py`: Created to help with launching jobs on SLURM


# Download data needed for language experiments
1. Download data from Yang and Piantadosi
```
cd formal_languages/
git clone https://github.com/piantado/Fleet.git
```

2. Download CHILDES data by following [the instructions provided by the authors of the paper introducing that dataset](https://github.com/adityayedetore/lm-povstim-with-childes/tree/master/data/CHILDES).
```
# Dividing up the files
cd CHILDES
cp -r pretraining pretraining_divided
python divide_training.py
python make_training_sets.py
```

3. Download the Zorro dataset from the [repository provided by Huebner et al.](https://github.com/phueb/Zorro/tree/master/sentences/babyberta)




