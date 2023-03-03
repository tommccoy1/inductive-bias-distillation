
import random
import math
import os
import logging
from collections import Counter

import config
os.environ['TRANSFORMERS_CACHE'] = config.TRANSFORMERS_PATH

import numpy as np
import torch

from transformers import GPT2LMHeadModel, AutoConfig

from training import *
from dataloading import *
from dataset_iterators import *
from models import *

import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser()

# Corpus arguments
parser.add_argument("--n_meta_train", help="Episodes in the meta-training set. Just one epoch will be run.", type=int, default=20000)
parser.add_argument("--n_meta_valid", help="Episodes in the meta-validation set", type=int, default=100)
parser.add_argument("--n_meta_test", help="Episodes in the meta-test set", type=int, default=100)

parser.add_argument("--meta_train_batch_size", help="Sequences per episode of meta training", type=int, default=100)
parser.add_argument("--meta_eval_batch_size", help="Batch size during meta evaluation", type=int, default=None)
parser.add_argument("--max_batches_per_language", help="Max number of training batches for each episode of meta training", type=int, default=1)
parser.add_argument("--meta_train_size", help="Training sequences per episode of a meta dataset; only used when evaluating with formal languages", type=int, default=10)
parser.add_argument("--meta_test_size", help="Evaluation sequences per episode of meta training", type=int, default=10)

parser.add_argument("--dataset", help="Data to meta-train on. Options: simple, yp, regex, scfg, abn, emb, cross", type=str, default="simple")
parser.add_argument("--yandp_param_file", help="File with production weights for the Yang & Piantadosi model", type=str, default=None)
parser.add_argument("--formal_train_size", help="Training size for using the formal languages as an evaluation set", type=int, default=100)
parser.add_argument("--formal_test_size", help="Test size for using the formal languages as an evaluation set", type=int, default=10)
parser.add_argument("--language_list", help="Name for file listing formal languages to use", type=str, default="language_list")
parser.add_argument("--withheld_languages", help="List of languages that should be withheld from the meta-training set", type=str, default=None) 

# Architecture arguments
parser.add_argument("--architecture", help="Type of architecture. Options: GPT2, LSTM", type=str, default="GPT2")
parser.add_argument("--n_embd", help="Embedding size; also used as the hidden size", type=int, default=768)
parser.add_argument("--n_positions", help="Max context length the model can take", type=int, default=128)
parser.add_argument("--n_head", help="Number of attention heads", type=int, default=12)
parser.add_argument("--n_layer", help="Number of layers", type=int, default=12)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.1)

# Training arguments
parser.add_argument("--n_epochs", help="Number of training epochs", type=int, default=1)
parser.add_argument("--eval_every", help="Number of training steps to go between evaluations", type=int, default=100)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-1)
parser.add_argument("--learning_rate", help="Outer-loop learning rate", type=float, default=5e-4)
parser.add_argument("--inner_lr", help="Inner-loop learning rate", type=float, default=1e-1)
parser.add_argument("--lr_scheduler_type", help="Learning rate scheduler type (cosine or constant)", type=str, default="cosine")
parser.add_argument("--warmup_proportion", help="Proportion of total steps that are warmup", type=float, default=0.05)
parser.add_argument("--patience", help="Patience", type=int, default=None)
parser.add_argument("--lr_decay_patience", help="Learning rate decay paatience", type=int, default=None)
parser.add_argument("--multi_step_loss", help="use multi-step loss", action='store_true')

# Saving arguments
parser.add_argument("--model_name", help="Model name prefix", type=str, default=None)
parser.add_argument("--weight_dir", help="Directory to save model weights in", type=str, default=config.WEIGHT_DIR)
parser.add_argument("--log_dir", help="Directory to save logs in", type=str, default=config.LOG_DIR)

# Evaluation arguments
parser.add_argument("--eval", help="Just evaluate, don't train", action='store_true')
parser.add_argument("--eval_formal", help="evaluate on formal languages", action='store_true')
parser.add_argument("--eval_valid", help="evaluate on the validation set", action='store_true')
parser.add_argument("--top_p", help="Probability mass to truncate the probability distribution to when sampling from an LM for precision and recall", type=float, default=1.00)
parser.add_argument("--prec_rec_n_samples", help="Number of samples to generate for precision and recall", type=int, default=10000)
parser.add_argument("--sgd_epochs", help="Number of epochs to do with SGD during adaptation", type=int, default=1)
parser.add_argument("--adam_epochs", help="Epochs with Adam, after SGD", type=int, default=10)
parser.add_argument("--eval_suffix", help="Suffix to add in filename for eval output", type=str, default="")
parser.add_argument("--random_normalized", help="Compare to normalized probabilities", action='store_true')
parser.add_argument("--return_last", help="No early stopping: Just use last values", action='store_true')
args = parser.parse_args()


if args.meta_eval_batch_size is None:
    args.meta_eval_batch_size = args.meta_train_batch_size



################################################################################################
# Set up logging
################################################################################################

if args.eval_suffix != "":
    args.eval_suffix = "_" + args.eval_suffix

if args.eval:
    if args.eval_formal:
        log_file_name = args.model_name + "_eval_formal_" + str(args.formal_train_size) + "_" + args.language_list + "_topp" + str(args.top_p) + "_nsamples" + str(args.prec_rec_n_samples) + args.eval_suffix
    else:
        log_file_name = args.model_name + args.eval_suffix + "_eval"
else:
    model_name = args.model_name
    model_index = 0
    args.model_name = model_name + "_" + str(model_index)
    while args.model_name + ".log" in os.listdir(args.log_dir):
        model_index += 1
        args.model_name = model_name + "_" + str(model_index)

    log_file_name = args.model_name

    random.seed(model_index)
    np.random.seed(model_index)
    torch.manual_seed(model_index)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + log_file_name + ".log")])


logging.info(args)


################################################################################################
# Set up the data
################################################################################################

withheld_languages = None
withheld_seq_dict = None
if args.withheld_languages is not None:
    withheld_languages, withheld_seq_dict = load_withheld_languages(args.withheld_languages)

if args.dataset == "simple":
    create_dataset = simple_dataset(10)
    integer_vocab_size = 10
elif args.dataset == "yandp":
    create_dataset = yandp_dataset(args.yandp_param_file, n_test=args.meta_test_size, batch_size=args.meta_train_batch_size, eval_batch_size=args.meta_eval_batch_size, max_batches_per_language=args.max_batches_per_language)
    integer_vocab_size = 10
elif args.dataset == "scfg":
    create_dataset = scfg_dataset(n_test=args.meta_test_size, batch_size=args.meta_train_batch_size, eval_batch_size=args.meta_eval_batch_size, max_batches_per_language=args.max_batches_per_language, withheld_languages=withheld_languages, withheld_seq_dict=withheld_seq_dict)
    integer_vocab_size = 10
elif args.dataset == "formal":
    create_dataset = formal_dataset(args.language_list, training_size=args.meta_train_size, test_size=args.meta_test_size, batch_size=args.meta_train_batch_size, eval_batch_size=args.meta_eval_batch_size, max_batches_per_language=args.max_batches_per_language)
    integer_vocab_size = 10
else:
    raise ValueError('The specified dataset is not implemented')

meta_dataset = MetaLMDataset(create_dataset=create_dataset, meta_train_size=args.n_meta_train, meta_valid_size=args.n_meta_valid, meta_test_size=args.n_meta_test, integer_vocab_size=integer_vocab_size, context_size=args.n_positions)



################################################################################################
# Set up the model
################################################################################################

if args.architecture == "GPT2":

    model = GPT2LM(vocab_size=len(meta_dataset.tokenizer)+1, emb_size=args.n_embd, n_positions=args.n_positions, n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout, save_dir=args.weight_dir, model_name=args.model_name).to(device)

elif args.architecture == "LSTM":
    
    model = RNNLM(rnn_type="LSTM", vocab_size=len(meta_dataset.tokenizer)+1, emb_size=args.n_embd, hidden_size=args.n_embd, n_layers=args.n_layer, dropout=args.dropout, tie_weights=True, save_dir=args.weight_dir, model_name=args.model_name).to(device)
    model.rnn.flatten_parameters()


model_size = sum(t.numel() for t in model.parameters())
logging.info(f"Model size: {model_size/1000**2:.1f}M parameters")



################################################################################################
# Meta-train
################################################################################################

warmup_steps = math.ceil(args.warmup_proportion*args.n_epochs*len(meta_dataset.train))
trainer = MetaTrainer(
        model=model,
        train_datasplit=meta_dataset.train,
        eval_datasplit=meta_dataset.valid,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr_decay_patience=args.lr_decay_patience,
        weight_decay=args.weight_decay,   
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        eval_every=args.eval_every,
        log_every=args.eval_every,
        tokenizer=meta_dataset.tokenizer,
        inner_lr=args.inner_lr,
        multi_step_loss=args.multi_step_loss,
        )

if not args.eval: 
    trainer.train()

if not args.model_name.startswith("random"):
    trainer.model.load()


################################################################################################
# Evaluate
################################################################################################

if (not args.eval) or args.eval_valid:
    trainer.evaluate(eval_datasplit=meta_dataset.valid, name="Validation")

if args.eval_formal:

    eval_formal(trainer.model, args.language_list, formal_train_size=args.formal_train_size, formal_test_size=args.formal_test_size, meta_train_batch_size=args.meta_train_batch_size, n_positions=args.n_positions, top_p=args.top_p, prec_rec_n_samples=args.prec_rec_n_samples, inner_lr=args.inner_lr, sgd_epochs=args.sgd_epochs, adam_epochs=args.adam_epochs, return_last=args.return_last, random_normalized=args.random_normalized)






