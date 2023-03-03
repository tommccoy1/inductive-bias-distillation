



# Getting started
1. Set the values in `config.py` to whatever you want the default directories to be.

2. Create a python venv to run code in (the code was developed with Python 3.9.12, but the specific version probably doesn't matter much, as long as it is some version of Python 3):
```
python -m venv .venv
```

3. Activate the venv
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
- `scfg.py`, `yandp.py`: Specifying two metagrammars
- `training.py`: Classes for (meta-)training models
- `yandp_weights/yandp_params_uniform.txt`: PCFG weights for use in Yang and Piantadosi's meta-grammar
- `sh_to_scrs.py`: Created to help with launching jobs on SLURM


# Download data needed for language experiments
1. Download data from Yang and Piantadosi
```
cd formal_languages/
git clone https://github.com/piantado/Fleet.git
```

2. Download CHILDES data. TODO: Get this uploaded somehow/somewhere.
```
# Dividing up the files
cd CHILDES
cp -r pretraining pretraining_divided
python divide_training.py
python make_training_sets.py
```

3. Download Zorro. TODO: Describe




