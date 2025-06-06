
# Introduction

This repository contains the code for [Modeling rapid language learning by distilling Bayesian priors into artificial neural networks](https://www.nature.com/articles/s41467-025-59957-y) by Tom McCoy and Tom Griffiths. If you use this code, please cite that paper.

Bibtex:
```
@article{mccoy2025modeling,
  title={Modeling rapid language learning by distilling Bayesian priors into artificial neural networks},
  author={McCoy, R. Thomas and Griffiths, Thomas L.},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group}
}
```

Text:
```
McCoy, R. T., & Griffiths, T. L. (2025). Modeling rapid language learning by distilling Bayesian priors into artificial neural networks. Nature Communications, 16(1), 1-14.
```

# Getting started
0. We ran our experiments on the operating system Springdale Linux 8, but it should work on any system that can run Python. The installation steps below should take approximately 10 minutes.

1. Set the values in `config.py` to whatever you want the default directories to be. Assuming that you want to stick with the ones listed there already, you should create the relevant directories under the main project folder. That is, first navigate to the `inductive_bias_distillation/` folder and then run:
```
mkdir logs
mkdir weights
mkdir transformers
```

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

# The torch version that we used was 2.0.0+cu117
pip install torch

# The transformers version that we used was 4.26.1
pip install transformers

# The higher version that we used was 0.2.1
pip install higher

# The jsonlines version that we used was 4.0.0
pip install jsonlines
```


# Quickstart
Here is a quick, simple example illustrating the inductive bias distillation procedure. Running this should take about 10 minutes. In this case, we are working with a space of "languages" where each "sentence" in a language is a sequence of numbers. Within a given language, all sequences are the same length, all sequences have the same first element as each other, and all sequences have the same last element as each other - but the elements in the middle can be any numbers. Here are some example languages in this framing:
```
Language 1: sequence length = 4, first element = 7, last element = 2
Example sequences:
7 3 9 2
7 8 6 2
7 7 9 2
7 2 5 2
7 3 3 2

Language 2: sequence length = 7, first element = 8, last element = 4
Example sequences:
8 7 4 5 8 4 4
8 3 1 0 3 3 4
8 2 1 9 7 6 4
8 4 8 5 6 3 4
8 7 5 3 5 3 4
```

In principle, a learner should be able to figure out the rule defining a language after seeing only one example in that language - but only if the learner has inductive biases encoding the parameters that govern this space of languages. In this example, we will use inductive bias distillation to create neural networks that have these inductive biases.

1. As a baseline, first test out a model that has not undergone inductive bias distillation by running this line of code:
```
python meta_train.py --n_meta_train 0 --n_meta_valid 20 --n_meta_test 10 --meta_train_batch_size 1 --meta_eval_batch_size 1 --max_batches_per_language 1 --meta_test_size 10 --dataset simple --architecture LSTM --n_embd 64 --n_layer 1 --eval_every 200 --learning_rate 0.005 --inner_lr 1.0 --warmup_proportion 0.5 --model_name random --eval_generate --eval
```
This code will train a standard (not meta-trained) neural network on one example from a language, and then ask the neural network to generate additional examples from that language. Unsurprisingly, the neural network will not do well here, because it has no way to know what sorts of rules govern the languages. Below is one sample output; based on the training example, we can see that the language uses sequences of length 4 starting with 1 and ending with 7. But after training the network on this example, the samples that it produces do not adhere to these constraints (unsurprisingly). Note that you might get different outputs from the model, but you should get the same basic result of the model failing to generalize in the intended way.
```
TRAINING EXAMPLE(S):
1 5 3 7
SEQUENCES SAMPLED FROM TRAINED MODEL:
5 5 7
7 2 0 8 9 8 7 7 8 5 3 8 3 4 3 3 2 0 5 1 0 5 1 3 0 4 9
1 5 9 5 2 2 6 0 8 6 2 5 4 8
9
5 3 9
```

2. Now let's use inductive bias distillation to create a neural network that is able to make the sorts of inferences required to learn one of these "languages" from a single example:
```
python meta_train.py --n_meta_train 5000 --n_meta_valid 20 --n_meta_test 10 --meta_train_batch_size 1 --meta_eval_batch_size 1 --max_batches_per_language 1 --meta_test_size 10 --dataset simple --architecture LSTM --n_embd 64 --n_layer 1 --eval_every 200 --learning_rate 0.005 --inner_lr 1.0 --warmup_proportion 0.5 --model_name tmp --eval_generate
```
The above line of code will first have the network undergo meta-learning, where it is exposed to many languages sampled from the space of possible languages. You should see the code printing out validation losses and perplexities, which should be going down, indicating that the model is successfully meta-learning.

Once meta-learning concludes, the model will be evaluated in the same way as our baseline model was evaluated in step (1) above. But now it should do a much better job of adhering to the constraints illustrated by the training example. For instance, after being trained on the same example as above, the model now correctly produces sequences that obey the relevant constraints (having length 4, starting with 1, and ending with 7). Note that the specific outputs that you get might vary, but you should get the same basic finding that the model now generalizes well.
```
TRAINING EXAMPLE(S):
1 5 3 7
SEQUENCES SAMPLED FROM TRAINED MODEL:
1 0 7 7
1 2 2 7
1 2 4 7
1 2 0 7
1 8 4 7
```



# Replicating formal language results (Figure 3)

0. The numbers used for Figure 3 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following in the directory `formal_language_results/`:
```
python figure3_stats.py
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





# Replicating perplexity results (Figure 4)

0. The numbers used in Figure 4 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following in the directory `natural_language_results`. In the output, `no` means a standard neural network, while `yes` means a prior-trained one.
```
# Figure 4a
python perplexity_1024.py 

# Figure 4b
python perplexity_all.py
```
The steps listed below describe all of the scripts whose results are aggregated by these commands.


1. Create datasets: First download the CHILDES training set (not yet publicly available; please contact us if you need it). Then, move the scripts `divide_training.py` and `make_training_sets.py` into the `CHILDEs/` directory, and then run each of them:
```
python divide_training.py
python make_training_sets.py
``` 

2. Train language models on these datasets. The commands to do this are in `natural_language_results/perplexity/adapt_besthyps_nopre.sh` and `natural_language_results/perplexity/adapt_besthyps_yespre.sh`. Each command needs to be run once to replicate all of our results; these commands vary widely in execution time (from a few minutes to about 1 day, on an A100 GPU). This yields 20 runs for each cell in the heatmap (Fig. 4b) and 40 runs for the largest-scale case (1024 hidden units and the full training set).

3. Compute perplexity on the test set. The commands for doing this are in `natural_language_results/perplexity/test_perplexity_for_heatmap.sh` and `natural_language_results/perplexity/test_perplexity_1024full.sh`. The results of running these commands are in the `.log` files in `natural_language_results/perplexity/`.


# Getting statistics about Yang & Piantadosi's meta-grammar

Running `python yandp.py` samples languages from Yang & Piantadosi's prior and logs statistics in `yandp_stats/yandp_stats.log`. The file shows the proportion of sampled languages that only generate the empty string (ONLY EPSILON), that have a maximum length of 1 (MAX LENGTH 1), that only produce one unique string (ONLY ONE UNIQUE), and that have both more than one unique string and a maximum length greater than 1 (LONG AND DIVERSE).


# Replicating targeted evaluation results (Figure 5)

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


# Replicating ablation results (Figure 6)

0. The numbers used in Figure 6 were produced by aggregating the results of many other scripts. To reproduce that aggregation step and see the numerical results, run the following.
```
cd ablation_formal_results/

# Figure 6a (left) results
python ablation_stats_recursion.py  

# Figure 6a (right) results
python ablation_stats_synchrony.py

# Figure 6b results
cd ../ablation_natural_results/
python recursion.py

# Figure 6c results
python priming.py

```

1. To create the results that the above commands aggregate, we took the following steps. First, we meta-trained models on the `no synchrony` and `no recursion` sets of primitives by running the commands in `ablation_formal_results/no_synchrony/meta_training.sh` and `ablation_formal_results/no_recursion/meta_training.sh`. We ran each of these commands 20 times, and each run took 1 to 2 days on an A100 GPU.


2. To adapt the meta-trained models to formal languages, we ran the commands in the following files. Each shows only the model with index `17`; these commands need to be run for all indices from `0` to `19` (which is done by replacing `17` with the relevant index).
```
all_primitives/adapt.sh
no_synchrony/adapt.sh
no_recursion/adapt.sh
standard/adapt.sh
```

3. To train the meta-trained models on the English corpus, we ran the commands in the following files. Each shows only the model with index `17`; these commands need to be run for all indices from `0` to `19` (which is done by replacing `17` with the relevant index).
```
ablation_natural_results/no_synchrony/adapt.sh
ablation_natural_results/no_recursion/adapt.sh
```

4. To get the evaluation results on the priming and synchrony tests, we ran the commands in the following files. Each shows only the model with index `17`; these commands need to be run for all indices from `0` to `19` (which is done by replacing `17` with the relevant index).
```
ablation_natural_results/no_synchrony/evaluate.sh
ablation_natural_results/no_recursion/evaluate.sh
``` 




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


# Download model weights

The trained model weights for our main experiments are available at [this OSF link](https://osf.io/59rgm/files/osfstorage). Below is a list of folders along with the types of models stored in each folder. Note that the "standard" model weights from Figure 3 of the paper are not included here; we used PyTorch's default random initialization approach to generate those weights:

- `meta_all_primitives`: Models meta-trained on formal languages (40 reruns), with our full set of primitives. These are the models underlying Figure 3 in the paper ("Prior-trained neural network").
- `meta_no_recursion`: Models meta-trained on formal languages but with the recursion primitive withheld (20 reruns). These are the "no recursion" cases in Figure 6 in the paper.
- `meta_no_sync`: Models meta-trained on formal languages but with the synchrony primitive withheld (20 reruns). These are the "no synchrony" cases in Figure 6 in the paper.
- `natural_language_meta`: The `meta_all_primitives` models further trained on an English corpus. These are the prior-trained networks in Figure 4 and Figure 5 in the paper.
- `natural_language_meta_no_recursion`: The `meta_no_recursion` models further trained on an English corpus. These are the "prior-trained (no recursion)" models in Figure 6 in the paper.
- `natural_language_meta_no_sync`: The `meta_no_sync` models further trained on an English corpus. These are the "prior-trained (no synchrony)" models in Figure 6 in the paper.
- `natural_language_standard`: Randomly-initialized models trained on the English corpus. These are the standard networks in Figure 4 and Figure 5 in the paper.
- `pretrained`: These models are like `meta_all_primitives` but pretrained rather than prior-trained. These are the "pre-trained neural networks" in Figure S2 in the supplementary materials.

Here is some basic code for loading one of the meta-trained models:
```
from models import *

model_name = "meta_lm_hidden1024_17"
model = RNNLM(rnn_type="LSTM", vocab_size=15, emb_size=1024, hidden_size=1024, n_layers=2, dropout=0.1, tie_weights=True, save_dir=".", model_name=model_name).to(device)

# If you want a randomly-initialized model, just give it a new name 
# (not the name of a saved model) and omit the loading command
# shown below
model.load()
```

And here is some basic code for loading one of the natural language models:
```
from models import *

model_name = "bestparams_adapt_hidden1024_pretraining_full_yespre23_0"
model = RNNLM(rnn_type="LSTM", vocab_size=17096, emb_size=1024, hidden_size=1024, n_layers=2, dropout=0.1, tie_weights=True, save_dir=".", model_name=model_name).to(device)

# If you want a randomly-initialized model, just give it a new name 
# (not the name of a saved model) and omit the loading command
# shown below
model.load()
```

Note, however, that for full functionality you will probably want to also load a dataset so that you can then use its tokenizer (the models load in the code above work directly with token IDs, not with textual input, so the tokenizer is necessary for using actual text). For illustrations of the whole pipeline, refer to `lm_train.py`, which instantiates a dataset and a model (the model can either be from scratch or from a checkpoitn) and can then evaluate the model by running an evaluation function such as `scamp_eval`.



